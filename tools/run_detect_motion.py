#!/usr/bin/env python3
import os
import sys
import json
import argparse

# Ensure we can import detect_motion from the repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from detect_motion import detect_motion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run detect_motion on a single video and save outputs.")
    parser.add_argument("input_video", help="Path to the input .mp4 file")
    parser.add_argument("output_dir", nargs="?", default=os.path.join(REPO_ROOT, "temp"), help="Directory to save outputs (default: repo/temp)")
    args = parser.parse_args()

    input_video = os.path.abspath(args.input_video)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running detect_motion on: {input_video}")
    res = detect_motion(input_video, output_dir)

    # Pretty print result summary
    summary = {
        "status": res.get("status"),
        "clip_path": res.get("clip_path"),
        "persons_detected": res.get("persons_detected"),
        "cars_detected": res.get("cars_detected"),
        "crossing_direction": res.get("crossing_direction"),
        "persons_up": res.get("persons_up"),
        "persons_down": res.get("persons_down"),
        "insignificant_frames_saved": len(res.get("insignificant_frames", [])),
    }
    print(json.dumps(summary, indent=2))

    if summary.get("clip_path"):
        print(f"Highlight clip: {summary['clip_path']}")
    else:
        print("No highlight clip generated.")


if __name__ == "__main__":
    main()
