"""Local frame analysis via Ollama (vision LLM) running on the worker's GPU.

For low-motion videos (`no_person` / `no_significant_motion`), `detect_motion` extracts a
representative middle frame per motion event. Instead of spending scarce Gemini quota on these, we
send those frames to a local vision model (Ollama) for a brief description of what likely triggered
the recording.

Two analysis modes (toggle with `OLLAMA_MULTI_FRAME`, default on):
- **Multi-frame (default):** when a video has ≥2 event frames, all frames go to Ollama in ONE
  multi-image call with a multi-frame prompt. The model sees the whole short clip at once, so it can
  narrate progression ("first someone crosses, then it goes still") and dedup naturally — more
  coherent and, for reasoning models, much faster than analyzing each frame separately (one
  reasoning pass instead of N + a merge). A 1-frame video is just a single-image call.
- **Legacy per-frame + combine:** with `OLLAMA_MULTI_FRAME=false`, each frame is analyzed
  separately and a text-only follow-up call merges the sentences into one.
- **Two-stage (set `OLLAMA_REFINE_MODEL`):** a fast vision model (`OLLAMA_MODEL`, per-frame,
  single-image) describes frames in English, then a dedicated language model (`OLLAMA_REFINE_MODEL`,
  e.g. lapa for Ukrainian) merges them into one cheeky sentence in the target language. Best for
  clean, russism-free Ukrainian; see the two-stage section in CLAUDE.md.

Model notes:
- qwen3-vl: use the `-instruct` variant. The base `qwen3-vl:2b` has mandatory chain-of-thought
  that emits hundreds of hidden tokens before any visible text — combined with `num_predict` that
  yields empty responses. Set `OLLAMA_THINK` unset (instruct doesn't reason).
- gemma4 (reasoning model): set `OLLAMA_THINK=true` + `OLLAMA_NUM_PREDICT=-1` for grounded Ukrainian
  output (it reads timestamps and rarely fabricates), at ~50-100s/frame on a CPU-bound worker.

This module is self-contained: no Gemini dependency, and degrades gracefully (returns ``None``)
whenever Ollama is unavailable or disabled, so the caller can fall back to a placeholder message.
It is called synchronously from `analyze_video`, which already runs in the master's I/O executor.
"""

import os
import time
import base64
import logging

import httpx

logger = logging.getLogger()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Prompt files by role: (env override, default). Each is resolved fresh per call so edits and env
# changes take effect without a restart; a relative path (env or default) is resolved against this
# module's directory, NOT the process CWD, so it works regardless of where the master was launched.
_PROMPT_FILES = {
    "single":  ("OLLAMA_PROMPT_FILE",         "config/prompt_frame.txt"),
    "multi":   ("OLLAMA_MULTI_PROMPT_FILE",   "config/prompt_frame_multi.txt"),
    "combine": ("OLLAMA_COMBINE_PROMPT_FILE", "config/prompt_frame_combine.txt"),
    "refine":  ("OLLAMA_REFINE_PROMPT_FILE",  "config/prompt_frame_refine_uk.txt"),
}


def _prompt_path(kind):
    env_var, default = _PROMPT_FILES[kind]
    path = os.getenv(env_var, default)
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def _load_text(path):
    # Read fresh each call so edits take effect without restart; the file is tiny next to the call.
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _parse_keep_alive(env="OLLAMA_KEEP_ALIVE", default="-1"):
    # Ollama accepts an int (seconds; -1 = forever) or a duration string ("30m"). Default -1 keeps
    # the model resident. Set OLLAMA_KEEP_ALIVE=30m to free RAM/GPU during idle hours.
    v = str(os.getenv(env, default)).strip()
    try:
        return int(v)
    except ValueError:
        return v


def _encode_frame(frame_path, file_basename):
    """Read a JPEG and return base64, or None on failure (logged).

    Downscales to OLLAMA_IMAGE_MAX_DIM (longest side, default 1280) before encoding: a 4K frame has
    a huge vision-patch count that the model spends ~30s encoding, while ~1280px is ~2.6x faster
    (~896px ~5x) with grounding intact. The original on-disk frame is untouched. Set 0 to disable.
    """
    try:
        max_dim = int(os.getenv("OLLAMA_IMAGE_MAX_DIM", "1280"))
        if max_dim > 0:
            try:
                import cv2
                img = cv2.imread(frame_path)
                if img is not None:
                    h, w = img.shape[:2]
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        return base64.b64encode(buf.tobytes()).decode("ascii")
            except Exception as e:
                # Any decode/resize problem → fall back to sending the original bytes.
                logger.debug("[%s] Frame downscale failed for %s (%s); sending original.",
                             file_basename, os.path.basename(frame_path), e)
        with open(frame_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError as e:
        logger.warning("[%s] Could not read frame %s for local analysis: %s",
                       file_basename, os.path.basename(frame_path), e)
        return None


def _sample_frames(frame_paths, cap):
    """Cap the frame count, sampling evenly across the list (keeping first and last) if needed.

    Returns (sampled_paths, dropped_count). Logging of any drop is the caller's job.
    """
    if cap <= 0 or len(frame_paths) <= cap:
        return frame_paths, 0
    n = len(frame_paths)
    idxs = sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})
    sampled = [frame_paths[i] for i in idxs]
    return sampled, n - len(sampled)


def _ollama_generate(client, base_url, model, timeout, prompt, images_b64,
                     options, attempts, max_chars, think, keep_alive, label, file_basename):
    """One /api/generate call (text-only if images_b64 is empty), with retry + ramble guard.

    Retries up to `attempts` times on a transient failure, an empty response, or a rambling
    (over-long) response. Returns the text, or None once exhausted.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
    }
    if images_b64:
        payload["images"] = images_b64
    if think is not None:
        payload["think"] = think  # e.g. true for gemma4 reasoning, omitted for non-reasoning models

    for attempt in range(1, attempts + 1):
        try:
            resp = client.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
            if not text:
                logger.warning("[%s] %s: empty response (attempt %d/%d).",
                               file_basename, label, attempt, attempts)
            # We expect ONE sentence; much longer means the model is looping on a phrase
            # ("...and it's so still, and the shadows are long, and it's so still...") — retry it.
            elif len(text) > max_chars:
                logger.warning("[%s] %s: rambled (%d chars) (attempt %d/%d).",
                               file_basename, label, len(text), attempt, attempts)
            else:
                return text
        except Exception as e:
            logger.warning("[%s] %s: request failed (attempt %d/%d): %s",
                           file_basename, label, attempt, attempts, e)
    return None


def _refine_descriptions(client, base_url, refine_model, timeout, attempts, max_chars,
                         keep_alive, descriptions, file_basename):
    """Stage 2 of the 2-stage pipeline: a dedicated language model (e.g. lapa for Ukrainian) merges
    the per-frame English descriptions into ONE cheeky sentence in the target language. Text-only.

    Set OLLAMA_REFINE_NUM_GPU=0 to force this (large) model fully onto CPU, leaving the GPU to the
    fast vision model. OLLAMA_REFINE_KEEP_ALIVE overrides keep-alive for the refine model only —
    set it to -1 so the (CPU-resident, 0-VRAM) refiner never unloads and avoids its ~50s cold load.
    Returns the refined sentence, or None (caller falls back to the raw stage-1).
    """
    try:
        template = _load_text(_prompt_path("refine"))
    except OSError as e:
        logger.warning("[%s] Could not load refine prompt %s: %s", file_basename, _prompt_path("refine"), e)
        return None
    prompt = template.replace("{descriptions}", "\n".join(f"- {d}" for d in descriptions))

    options = {
        "temperature": float(os.getenv("OLLAMA_REFINE_TEMPERATURE", "0.8")),
        "num_predict": int(os.getenv("OLLAMA_REFINE_NUM_PREDICT", "120")),
        "repeat_penalty": float(os.getenv("OLLAMA_REFINE_REPEAT_PENALTY", "1.1")),
    }
    rg = os.getenv("OLLAMA_REFINE_NUM_GPU")
    if rg not in (None, ""):
        options["num_gpu"] = int(rg)
    # The refiner can be kept resident independently (it's CPU/0-VRAM, so pinning it is ~free).
    if os.getenv("OLLAMA_REFINE_KEEP_ALIVE") not in (None, ""):
        keep_alive = _parse_keep_alive("OLLAMA_REFINE_KEEP_ALIVE")

    # think=None: refine models (lapa) aren't reasoning models.
    return _ollama_generate(client, base_url, refine_model, timeout, prompt, [],
                            options, attempts, max_chars, None, keep_alive,
                            "Stage2 refine", file_basename)


def analyze_frames_local(frame_paths, file_basename):
    """Describe a video's event frames with the local vision model.

    Args:
        frame_paths (list[str]): Master-perspective paths to saved event-frame JPEGs.
        file_basename (str): For log prefixing.

    Returns:
        str | None: A one-sentence description, or None if the feature is disabled, there are no
            usable frames, or every request failed (caller falls back to a placeholder).
    """
    enabled = os.getenv("LOCAL_FRAME_ANALYSIS_ENABLED", "true").lower() == "true"
    if not enabled:
        logger.debug("[%s] Local frame analysis disabled via env.", file_basename)
        return None
    if not frame_paths:
        return None

    base_url = os.getenv("OLLAMA_URL", "http://10.0.0.2:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3-vl:2b-instruct")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "60"))
    attempts = max(1, int(os.getenv("OLLAMA_ATTEMPTS", "2")))  # total tries per call (incl. first)
    keep_alive = _parse_keep_alive()
    # Sampling controls. A high temperature gives livelier output, but on its own it makes a small
    # model ramble/loop until it fills num_ctx and gets aborted (empty response). The tight top_k/top_p
    # clamp blocks the incoherent long-tail tokens. num_predict caps length; repeat_penalty breaks loops.
    options = {
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.8")),
        "top_k": int(os.getenv("OLLAMA_TOP_K", "20")),
        "top_p": float(os.getenv("OLLAMA_TOP_P", "0.92")),
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "120")),
        "repeat_penalty": float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.3")),
    }
    # We expect one sentence; a much longer response is a rambling/repetition loop → retry it.
    max_chars = int(os.getenv("OLLAMA_MAX_CHARS", "300"))
    # Reasoning toggle. Omitted by default (instruct models don't reason). Set OLLAMA_THINK=true for
    # gemma4 (with OLLAMA_NUM_PREDICT=-1) so its reasoning isn't truncated into an empty response.
    _think_env = os.getenv("OLLAMA_THINK")
    think = None if _think_env is None else (_think_env.lower() == "true")

    multi_frame = os.getenv("OLLAMA_MULTI_FRAME", "true").lower() == "true"
    refine_model = os.getenv("OLLAMA_REFINE_MODEL", "").strip()
    max_frames = int(os.getenv("OLLAMA_MULTI_MAX_FRAMES", "4"))

    with httpx.Client() as client:
        # --- Two-stage mode (OLLAMA_REFINE_MODEL set): vision model describes each frame (English),
        #     then a dedicated language model (e.g. lapa) merges them into one cheeky sentence in the
        #     target language. Per-frame single-image because small vision models (qwen-2B) mishandle
        #     multi-image input. ---
        if refine_model:
            try:
                v_prompt = _load_text(_prompt_path("single"))
            except OSError as e:
                logger.warning("[%s] Could not load frame prompt %s: %s", file_basename, _prompt_path("single"), e)
                return None
            used_paths, dropped = _sample_frames(frame_paths, max_frames)
            if dropped:
                logger.info("[%s] Local frame analysis: %d frame(s) over cap of %d — sampling %d evenly.",
                            file_basename, len(frame_paths), max_frames, len(used_paths))
            t_stage1 = time.time()
            descriptions = []
            for frame_path in used_paths:
                b64 = _encode_frame(frame_path, file_basename)
                if not b64:
                    continue
                d = _ollama_generate(client, base_url, model, timeout, v_prompt, [b64],
                                     options, attempts, max_chars, think, keep_alive,
                                     "Stage1 vision", file_basename)
                if d:
                    descriptions.append(d)
            if not descriptions:
                return None
            logger.info("[%s] Stage1 vision (%s) described %d frame(s) in %.1fs: %s",
                        file_basename, model, len(descriptions), time.time() - t_stage1,
                        " / ".join(descriptions))

            t_stage2 = time.time()
            refined = _refine_descriptions(client, base_url, refine_model, timeout, attempts,
                                           max_chars, keep_alive, descriptions, file_basename)
            stage2_s = time.time() - t_stage2
            if refined:
                logger.info("[%s] Stage2 refine (%s) in %.1fs: %s",
                            file_basename, refine_model, stage2_s, refined)
                return refined
            logger.warning("[%s] Stage2 refine (%s) failed in %.1fs; returning raw stage-1 description(s).",
                           file_basename, refine_model, stage2_s)
            return " / ".join(descriptions)

        # --- Multi-frame mode (default): one multi-image call over the whole clip ---
        if multi_frame:
            used_paths, dropped = _sample_frames(frame_paths, max_frames)
            if dropped:
                logger.info("[%s] Local frame analysis: %d frame(s) over cap of %d — sampling %d evenly.",
                            file_basename, len(frame_paths), max_frames, len(used_paths))
            images = [b for b in (_encode_frame(p, file_basename) for p in used_paths) if b]
            if not images:
                return None
            # 1 frame → single-frame prompt; 2+ → multi-frame prompt.
            path = _prompt_path("multi") if len(images) > 1 else _prompt_path("single")
            try:
                prompt = _load_text(path)
            except OSError as e:
                logger.warning("[%s] Could not load frame prompt %s: %s", file_basename, path, e)
                return None
            t0 = time.time()
            result = _ollama_generate(client, base_url, model, timeout, prompt, images,
                                      options, attempts, max_chars, think, keep_alive,
                                      "Local frame analysis", file_basename)
            if result:
                logger.info("[%s] Local frame analysis (%s, prompt=%s) described %d frame(s) in one call in %.1fs: %s",
                            file_basename, model, os.path.basename(path), len(images), time.time() - t0, result)
            return result

        # --- Legacy per-frame + combine mode (OLLAMA_MULTI_FRAME=false) ---
        try:
            prompt = _load_text(_prompt_path("single"))
        except OSError as e:
            logger.warning("[%s] Could not load frame prompt %s: %s", file_basename, _prompt_path("single"), e)
            return None

        t0 = time.time()
        descriptions = []
        for frame_path in frame_paths:
            b64 = _encode_frame(frame_path, file_basename)
            if not b64:
                continue
            desc = _ollama_generate(client, base_url, model, timeout, prompt, [b64],
                                    options, attempts, max_chars, think, keep_alive,
                                    "Local frame analysis", file_basename)
            if desc:
                descriptions.append(desc)

        if not descriptions:
            return None

        # Merge redundant per-frame sentences into one via a text-only follow-up call.
        if len(descriptions) > 1 and os.getenv("OLLAMA_COMBINE_FRAMES", "true").lower() == "true":
            try:
                template = _load_text(_prompt_path("combine"))
                combine_prompt = template.replace("{sentences}", "\n".join(f"- {d}" for d in descriptions))
                combined = _ollama_generate(client, base_url, model, timeout, combine_prompt, [],
                                            options, attempts, max_chars, think, keep_alive,
                                            "Combine call", file_basename)
            except OSError as e:
                logger.warning("[%s] Could not load combine prompt %s: %s", file_basename, _prompt_path("combine"), e)
                combined = None
            if combined:
                logger.info("[%s] Local frame analysis (%s) combined %d/%d frame(s) into one sentence in %.1fs: %s",
                            file_basename, model, len(descriptions), len(frame_paths), time.time() - t0, combined)
                return combined

        joined = " / ".join(descriptions)
        logger.info("[%s] Local frame analysis (%s) described %d/%d frame(s) in %.1fs: %s",
                    file_basename, model, len(descriptions), len(frame_paths), time.time() - t0, joined)
        return joined
