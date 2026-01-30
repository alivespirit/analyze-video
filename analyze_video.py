import os
import datetime
import random
import time
import logging

from google import genai
from google.genai import types
from dotenv import dotenv_values

logger = logging.getLogger()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USERNAME = os.getenv("TELEGRAM_NOTIFY_USERNAME")

NO_ACTION_RESPONSES = [
    "Нема шо дивитись",
    "Ніц цікавого",
    "Йойки, геть ніц",
    "Все спокійно",
    "Німа нічо",
    "Журбинка якась",
    "Сумулька лиш",
    "Чортівні нема",
    "Всьо чотко",
    "Геть нема екшину"
]

try:
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=120000)
    )
    logger.info("Gemini configured successfully.")
except Exception as e:
    logger.critical(f"Failed to configure Gemini: {e}", exc_info=True)
    raise


def analyze_video(motion_result, video_path):
    """
    Generates a descriptive text for a video using Google's Gemini AI and formats the final response.

    This function runs in the I/O executor. Its behavior depends on the motion detection result:
    -   **No Motion**: Returns a random "nothing to see" message.
    -   **Gate Crossing**: Returns a formatted message indicating the crossing event, skipping Gemini.
    -   **Off-Peak Hours**: Skips Gemini and returns a summary based on object counts to save API calls.
    -   **Significant Motion (Peak Hours)**: Sends the highlight clip to Gemini for analysis.
    -   **Error/Insignificant Motion (Peak Hours)**: Sends the full video to Gemini for analysis.

    It handles model selection (Pro vs. Flash), API retries, and error handling.

    Args:
        motion_result (dict): The result dictionary from the `detect_motion` function.
        video_path (str): The path to the original video file.

    Returns:
        dict: A dictionary containing the formatted text response, paths to any insignificant frames,
              and the path to the highlight clip if one was generated.
    """
    file_basename = os.path.basename(video_path)
    timestamp = f"_{video_path.split(os.path.sep)[-2][-2:]}H{file_basename[:3]}:_ "
    use_files_api = False
    now = datetime.datetime.now()

    if motion_result is None or not isinstance(motion_result, dict):
        logger.warning(f"[{file_basename}] Motion detection returned an unexpected value: {motion_result}. Analyzing full video.")
        motion_result = {'status': 'error', 'clip_path': None, 'insignificant_frames': []}

    # Append ReID result if available and positive
    reid = motion_result.get('reid')
    reid_text = ""
    if isinstance(reid, dict):
        score = float(reid.get('score', 0.0))
        percent = int(round(score * 100))
        reid_matched = reid.get('matched')
        if reid_matched is True:
            reid_text = f" / \U0001FAC6 *{percent}%*\n{USERNAME}"
        elif reid_matched is False:
            reid_text = f" / \U0001FAC6 {percent}%"

    detected_motion_status = motion_result['status']
    
    # --- Skip Gemini analysis for no motion events ---
    if detected_motion_status == "no_motion":
        logger.info(f"[{file_basename}] Skipping Gemini analysis (no motion).")
        return {'response': timestamp + "\u2714\uFE0F " + random.choice(NO_ACTION_RESPONSES), 'insignificant_frames': [], 'clip_path': None}

    # --- Skip Gemini analysis for gate crossing events ---
    if detected_motion_status == "gate_crossing":
        logger.info(f"[{file_basename}] Skipping Gemini analysis (gate crossing).")
        direction = motion_result.get('crossing_direction')
        persons_up = motion_result.get('persons_up', 0)
        persons_down = motion_result.get('persons_down', 0)

        direction_text = ""
        if direction == 'up':
            direction_text = "\U0001F6A7" +"\U0001F6B6\u200D\u27A1\uFE0F" * persons_up
            if USERNAME in reid_text:
                reid_text += " - *Юху!*"
                logger.info(f"[{file_basename}] AUTO Reaction detected: object went away.")
        elif direction == 'down':
            direction_text = "\U0001F6B6\u200D\u27A1\uFE0F" * persons_down + " \U0001F6A7"
            if USERNAME in reid_text:
                reid_text += " - *Ех...*"
                logger.info(f"[{file_basename}] AUTO Reaction detected: object came back.")
        elif direction == 'both':
            direction_text = "\U0001F6B6\u200D\u27A1\uFE0F" * persons_down + " \U0001F6A7" + "\U0001F6B6\u200D\u27A1\uFE0F" * persons_up

        analysis_result = direction_text + reid_text

        return {
            'response': timestamp + analysis_result,
            'insignificant_frames': motion_result.get('insignificant_frames', []),
            'clip_path': motion_result.get('clip_path')
        }
    # -----------------------------------------

    # --- Skip Gemini analysis during off-peak hours to keep under rate limits ---
    if now.hour < 9 or now.hour > 18:
        logger.info(f"[{file_basename}] Skipping Gemini analysis (off-peak hours).")
        if detected_motion_status == "error":
            return {'response': timestamp + "\U0001F4A2 Шось неясно", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': None}
        elif detected_motion_status == "no_significant_motion":
            return {'response': timestamp + "\U0001F518 Шось там цейво...", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': None}
        elif detected_motion_status == "no_person":
            return {'response': timestamp + "\U0001F532 Шось нікого...", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
        elif detected_motion_status == "significant_motion":
            # --- MODIFIED: Append detected object counts to the off-peak message ---
            persons = motion_result.get('persons_detected', 0)
            cars = motion_result.get('cars_detected', 0)
            details = []
            if persons > 0:
                details.append(f"{persons} \U0001F9CD")
            if cars > 0:
                details.append(f"{cars} \U0001F699")

            if details:
                return {'response': timestamp + f"\u2611\uFE0F Шось там {', '.join(details)}" + reid_text, 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
            else:
                return {'response': timestamp + "\u2611\uFE0F Виявлено капець рух." + reid_text, 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}

    video_to_process = None
    video_bytes_obj = None
    if detected_motion_status == "error":
        logger.warning(f"[{file_basename}] Error during motion detection. Analyzing full video.")
        video_to_process = video_path
    elif detected_motion_status == "no_significant_motion" or detected_motion_status == "no_person":
        logger.info(f"[{file_basename}] Analyzing full video as there was no significant motion or no person detected.")
        video_to_process = video_path
    elif detected_motion_status == "significant_motion":
        logger.info(f"[{file_basename}] Running Gemini analysis for detected motion at {motion_result['clip_path']}")
        video_to_process = motion_result['clip_path']

    if not video_to_process:
        logger.info(f"[{file_basename}] No video to analyze, but insignificant frames may exist.")
        return {'response': timestamp + "\U0001F4A2 Нема значного руху.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': None}

    try:
        if use_files_api:
            video_bytes_obj = client.files.upload(file=video_to_process)
            max_wait_seconds = 120
            wait_interval = 10
            waited = 0
            while video_bytes_obj.state == "PROCESSING":
                if waited >= max_wait_seconds:
                    logger.error(f"[{file_basename}] Video processing timed out after {max_wait_seconds} seconds.")
                    return {'response': timestamp + "\u274C Відео не вдалося обробити (timeout).", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
                logger.info(f"[{file_basename}] Waiting for video to be processed ({waited}/{max_wait_seconds}s).")
                time.sleep(wait_interval)
                waited += wait_interval
                video_bytes_obj = client.files.get(name=video_bytes_obj.name)

            if video_bytes_obj.state == "FAILED":
                logger.error(f"[{file_basename}] Video processing failed: {video_bytes_obj.error_message}")
                return {'response': timestamp + "\u274C Відео не вдалося обробити.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
        else:
            try:
                with open(video_to_process, 'rb') as f:
                    video_data = f.read()
            except Exception as e:
                logger.error(f"[{file_basename}] Error reading video file: {e}")
                return {'response': timestamp + "\u274C Відео не вдалося прочитати.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}

        # --- Load model names from gemini_models.env on each call ---
        models_env_path = os.path.join(SCRIPT_DIR, "gemini_models.env")
        try:
            models_env = dotenv_values(models_env_path) or {}
        except Exception as e_env:
            logger.warning(f"[{file_basename}] Failed to load gemini_models.env: {e_env}. Using defaults.")
            models_env = {}

        MODEL_PRO = (models_env.get("MODEL_PRO") or "").strip() or None
        MODEL_MAIN = (models_env.get("MODEL_MAIN") or "").strip() or "gemini-2.5-flash"
        MODEL_FALLBACK = (models_env.get("MODEL_FALLBACK") or "").strip() or "gemini-2.5-flash-lite"
        MODEL_FINAL_FALLBACK = (models_env.get("MODEL_FINAL_FALLBACK") or "").strip() or None

        # Codename mapping for known models
        MODEL_CODENAMES = {
            "gemini-3-flash-preview": "3FP",
            "gemini-2.5-flash": "2.5F",
            "gemini-2.5-flash-lite": "2.5FL",
            "gemini-2.5-pro": "2.5Pro",
        }

        # Select main and fallback based on PRO presence and time window
        if MODEL_PRO and (9 <= now.hour <= 13):
            model_main = MODEL_PRO
            model_fallback = MODEL_MAIN
            model_fallback_text = f'_[{MODEL_CODENAMES.get(MODEL_MAIN, "FB")}]_ '
        else:
            model_main = MODEL_MAIN
            model_fallback = MODEL_FALLBACK
            model_fallback_text = f'_[{MODEL_CODENAMES.get(MODEL_FALLBACK, "FB")}]_ '
        # Final fallback
        final_fallback_enabled = bool(MODEL_FINAL_FALLBACK)
        model_final_fallback = MODEL_FINAL_FALLBACK
        model_final_fallback_text = f'_[{MODEL_CODENAMES.get(MODEL_FINAL_FALLBACK, "FF")}]_ '

        sampling_rate = 5
        max_retries = 3

        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
                prompt = prompt_file.read().strip()
            logger.debug("[%s] Prompt loaded successfully from %s.", file_basename, prompt_file_path)
        except FileNotFoundError:
            logger.error(f"[{file_basename}] Prompt file not found: {prompt_file_path}")
            return {'response': timestamp + "Prompt file not found.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
        except Exception as e:
            logger.error(f"[{file_basename}] Error reading prompt file: {e}", exc_info=True)
            return {'response': timestamp + "Error reading prompt file.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}

        if use_files_api:
            contents = [video_bytes_obj, prompt]
        else:
            contents = types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=video_data,
                            mime_type='video/mp4'
                        ),
                        video_metadata=types.VideoMetadata(fps=sampling_rate)
                    ),
                    types.Part(text=prompt)
                ]
            )
        gen_config = types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))

        analysis_result = ""

        for attempt in range(max_retries):
            try:
                logger.info(f"[{file_basename}] Generating content ({model_main}), attempt {attempt+1}...")
                response = client.models.generate_content(
                    model=model_main,
                    contents=contents,
                    config=gen_config
                )
                logger.info(f"[{file_basename}] {model_main} response received.")
                if response.text is None or response.text.strip() == "":
                    raise ValueError(f"{model_main} returned an empty response with reason {response.candidates[0].finish_reason.name}.")
                analysis_result = response.text
                break
            except Exception as e_main:
                try:
                    logger.warning(f"[{file_basename}] {model_main} failed. Falling back to {model_fallback}. Message: {e_main}")
                    response = client.models.generate_content(
                        model=model_fallback,
                        contents=contents,
                        config=gen_config
                    )
                    logger.info(f"[{file_basename}] {model_fallback} response received.")
                    if response.text is None or response.text.strip() == "":
                        raise ValueError(f"{model_fallback} returned an empty response with reason {response.candidates[0].finish_reason.name}.")
                    analysis_result = model_fallback_text + response.text
                    break
                except Exception as e_fallback:
                    logger.warning(f"[{file_basename}] {model_fallback} also failed: {e_fallback}")
                    if '429' in str(e_fallback) and '429' in str(e_main) and (detected_motion_status == "no_significant_motion" or detected_motion_status == "no_person"):
                        logger.info(f"[{file_basename}] Rate limit exceeded on both models, skipping retries.")
                        attempt = max_retries - 1  # Skip retries for rate limit on non-critical analyses
                    if attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt)
                        logger.info(f"[{file_basename}] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        if final_fallback_enabled:
                            try:
                                logger.info(f"[{file_basename}] Attempting {model_final_fallback} as final fallback...")
                                response = client.models.generate_content(
                                    model=model_final_fallback,
                                    contents=contents,
                                    config=gen_config
                                )
                                logger.info(f"[{file_basename}] {model_final_fallback} response received.")
                                analysis_result = model_final_fallback_text + response.text
                                break
                            except Exception as e_fallback_final:
                                logger.error(f"[{file_basename}] {model_final_fallback} failed as well: {e_fallback_final}")
                                logger.error(f"[{file_basename}] Giving up after retries.")
                                raise
                        else:
                            logger.error(f"[{file_basename}] Giving up after retries.")
                            raise

        analysis_result = (analysis_result[:512] + '...') if len(analysis_result) > 1023 else analysis_result

        logger.info(f"[{file_basename}] Response: {analysis_result}")

        # Append detected object counts to the Gemini result
        persons = motion_result.get('persons_detected', 0)
        cars = motion_result.get('cars_detected', 0)
        details = []
        if persons > 0:
            details.append(f"{persons} \U0001F9CD")
        if cars > 0:
            details.append(f"{cars} \U0001F699")

        if details:
            analysis_result += f" ({', '.join(details)})"

        if detected_motion_status == "significant_motion":
            timestamp += "\u2705 *Отакої!* "
            analysis_result += reid_text
        elif detected_motion_status == "no_person":
            timestamp += "\u274E "
        else:
            timestamp += "\u2747\uFE0F "

        return {'response': timestamp + analysis_result, 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}

    except Exception as e_analysis:
        logger.error(f"[{file_basename}] Video analysis failed: {e_analysis}", exc_info=False)
        if '429' in str(e_analysis):
            return {'response': timestamp + "\u26A0\uFE0F Ти забагато питав...", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
        else:
            return {'response': timestamp + "\u274C Відео не вдалося проаналізувати: " + str(e_analysis)[:512] + '...', 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
    finally:
        if use_files_api and 'video_bytes_obj' in locals() and hasattr(video_bytes_obj, 'name'):
            try:
                client.files.delete(name=video_bytes_obj.name)
                logger.info(f"[{file_basename}] Successfully deleted uploaded file from Gemini API.")
            except Exception as e_delete:
                logger.warning(f"[{file_basename}] Failed to delete uploaded file from Gemini API: {e_delete}", exc_info=False)
