#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging

# Ensure we can import detect_motion from the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from detect_motion import detect_motion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run detect_motion on a single video and save outputs.")
    parser.add_argument("input_video", help="Path to the input .mp4 file")
    parser.add_argument("output_dir", nargs="?", default=os.path.join(REPO_ROOT, "temp"), help="Directory to save outputs (default: repo/temp)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for console output (default: INFO)",
    )
    args = parser.parse_args()

    input_video = os.path.abspath(args.input_video)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Configure console logging so detect_motion logs are visible
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    print(f"Running detect_motion on: {input_video}")
    res = detect_motion(input_video, output_dir)

    # Pretty print result summary
    reid = res.get("reid") or None
    summary = {
        "status": res.get("status"),
        "clip_path": res.get("clip_path"),
        "persons_detected": res.get("persons_detected"),
        "cars_detected": res.get("cars_detected"),
        "crossing_direction": res.get("crossing_direction"),
        "persons_up": res.get("persons_up"),
        "persons_down": res.get("persons_down"),
        "insignificant_frames_saved": len(res.get("insignificant_frames", [])),
        "reid": (
            {
                "matched": reid.get("matched"),
                "score": reid.get("score"),
                "neg_score": reid.get("neg_score"),
                "threshold": reid.get("threshold"),
                "margin": reid.get("margin"),
                "samples": reid.get("samples"),
                "best_path": reid.get("best_path"),
            }
            if isinstance(reid, dict)
            else None
        ),
    }
    print(json.dumps(summary, indent=2))

    if summary.get("clip_path"):
        print(f"Highlight clip: {summary['clip_path']}")
    else:
        print("No highlight clip generated.")


if __name__ == "__main__":
    main()
