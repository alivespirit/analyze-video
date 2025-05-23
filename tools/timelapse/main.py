#!/usr/bin/env python3

import argparse
import cv2 # OpenCV library
import sys
import re
from pathlib import Path
import shutil
import os # For sorting

# --- Configuration (can be overridden by command-line arguments) ---
DEFAULT_BASE_DIR = Path("/mnt/nas/xiaomi_camera_videos")
DEFAULT_CAMERA_ID = "04cf8c6b201d" # Use '*' for all cameras (requires globbing)
DEFAULT_DATE_PATTERN = "2025*" # Use '*' for all dates/hours
DEFAULT_TMP_DIR_NAME = "xiaomi_frames_tmp_cv2"
DEFAULT_FRAMERATE = 60
DEFAULT_FRAME_FORMAT = ".jpg" # Format for temporary frames (jpg is usually smaller)
# --- End Configuration ---

def extract_timestamp(filename):
    """Extracts the numerical timestamp part from the filename."""
    match = re.search(r'_(\d+)\.mp4$', filename)
    if match:
        return match.group(1)
    return None

def sort_key_xiaomi(path: Path):
    """Generate a sort key based on parent directory name and filename timestamp."""
    timestamp = extract_timestamp(path.name)
    # Use parent dir name (YYYYMMDDHH) and timestamp for sorting
    # Return large number if timestamp is missing to sort them last (or handle as error)
    return (path.parent.name, int(timestamp) if timestamp else float('inf'))


def main():
    parser = argparse.ArgumentParser(description="Create a timelapse from Xiaomi camera videos using OpenCV.")
    parser.add_argument("-b", "--base-dir", type=Path, default=DEFAULT_BASE_DIR,
                        help=f"Base directory containing camera ID folders (default: {DEFAULT_BASE_DIR})")
    parser.add_argument("-c", "--camera-id", type=str, default=DEFAULT_CAMERA_ID,
                        help=f"Camera ID subfolder name, or '*' for all (default: {DEFAULT_CAMERA_ID})")
    parser.add_argument("-d", "--date-pattern", type=str, default=DEFAULT_DATE_PATTERN,
                        help=f"Date/hour pattern (e.g., '20250422*', '2025042210', '*') (default: {DEFAULT_DATE_PATTERN})")
    parser.add_argument("-t", "--tmp-dir", type=Path, default=Path(".") / DEFAULT_TMP_DIR_NAME,
                        help=f"Temporary directory for frames (default: ./{DEFAULT_TMP_DIR_NAME})")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output timelapse file path (default: timelapse_CAMID_DATE.mp4)")
    parser.add_argument("-f", "--framerate", type=int, default=DEFAULT_FRAMERATE,
                        help=f"Framerate for the output timelapse video (default: {DEFAULT_FRAMERATE})")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Do not delete the temporary frame directory after completion.")

    args = parser.parse_args()

    # Determine search path pattern
    search_path = args.base_dir / args.camera_id / args.date_pattern

    # Determine output filename if not specified
    output_file = args.output
    if not output_file:
        date_part = args.date_pattern.replace('*', 'all')
        output_file = Path(f"timelapse_{args.camera_id}_{date_part}_cv2.mp4")

    # --- Create temporary directory ---
    try:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Temporary directory: {args.tmp_dir.resolve()}")
    except OSError as e:
        print(f"Error: Failed to create temporary directory {args.tmp_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Find and sort video files ---
    print(f"Searching for videos in: {search_path.parent} matching '{search_path.name}/**/*.mp4'")

    # Build the glob pattern relative to the camera ID directory
    # Example: If date_pattern is '20250422*', glob becomes '20250422*/**/*.mp4'
    # Example: If date_pattern is '2025042210', glob becomes '2025042210/*.mp4'
    if '*' in args.date_pattern or len(args.date_pattern) == 8: # YYYYMMDD or YYYYMMDD*
        glob_pattern = f"{args.date_pattern}/**/*.mp4"
    else: # Specific hour YYYYMMDDHH
        glob_pattern = f"{args.date_pattern}/*.mp4"

    search_root = args.base_dir / args.camera_id
    try:
        # Find all matching files
        all_files = list(search_root.glob(glob_pattern))
        # Sort files chronologically using directory name and timestamp
        video_files = sorted(all_files, key=sort_key_xiaomi)

    except Exception as e:
         print(f"Error finding video files using pattern '{glob_pattern}' in '{search_root}': {e}", file=sys.stderr)
         sys.exit(1)

    if not video_files:
        print(f"Warning: No video files found matching the pattern '{glob_pattern}' in '{search_root}'.", file=sys.stderr)
        if not args.keep_frames:
            try:
                args.tmp_dir.rmdir() # Remove empty dir
            except OSError:
                pass # Ignore if already removed or not empty
        sys.exit(0) # Exit gracefully if no files found

    print(f"Found {len(video_files)} video files to process.")

    # --- Extract Frames ---
    print("Extracting frame #30 from each video...")
    frames_extracted_count = 0
    for video_file in video_files:
        timestamp = extract_timestamp(video_file.name)
        if not timestamp:
            print(f"  Warning: Could not extract timestamp from '{video_file.name}'. Skipping.", file=sys.stderr)
            continue

        # Naming convention ensures chronological sorting later
        output_frame_path = args.tmp_dir / f"frame_{timestamp}{DEFAULT_FRAME_FORMAT}"

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"  Warning: Could not open video file '{video_file}'. Skipping.", file=sys.stderr)
            continue

        # Set the video position to frame #30
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        # Read the frame at position #30
        ret, frame = cap.read()
        cap.release() # Release the video capture object immediately

        if ret:
            # Save the frame
            try:
                # Ensure the frame data is valid before writing
                if frame is not None and frame.size > 0:
                     success = cv2.imwrite(str(output_frame_path), frame)
                     if success:
                         # print(f"  Extracted frame from {video_file.name} to {output_frame_path.name}") # Verbose
                         frames_extracted_count += 1
                     else:
                         print(f"  Warning: Failed to write frame for {video_file.name} to {output_frame_path}", file=sys.stderr)
                else:
                     print(f"  Warning: Read invalid frame data from {video_file.name}. Skipping.", file=sys.stderr)

            except Exception as e:
                print(f"  Error writing frame for {video_file.name}: {e}", file=sys.stderr)
        else:
            print(f"  Warning: Could not read first frame from '{video_file.name}'. Skipping.", file=sys.stderr)

    print(f"Finished extracting frames. Successfully extracted {frames_extracted_count} frames.")

    if frames_extracted_count == 0:
         print("Error: No frames were successfully extracted. Cannot create timelapse.", file=sys.stderr)
         if not args.keep_frames:
             try:
                 shutil.rmtree(args.tmp_dir)
                 print(f"Cleaned up empty temporary directory: {args.tmp_dir}")
             except OSError as e:
                 print(f"Warning: Could not remove temporary directory {args.tmp_dir}: {e}", file=sys.stderr)
         sys.exit(1)

    # --- Create Timelapse Video ---
    print(f"Creating timelapse video '{output_file}' at {args.framerate} fps...")

    # Get the list of extracted frames, sorted by timestamp in filename
    frame_files = sorted(list(args.tmp_dir.glob(f"frame_*{DEFAULT_FRAME_FORMAT}")))

    if not frame_files:
        print("Error: No frame image files found in the temporary directory.", file=sys.stderr)
        # Cleanup is handled below if needed
        sys.exit(1)

    # Read the first frame to get dimensions (width, height)
    first_frame_path = frame_files[0]
    try:
        first_frame_img = cv2.imread(str(first_frame_path))
        if first_frame_img is None:
             raise IOError(f"Could not read first frame image: {first_frame_path}")
        height, width, layers = first_frame_img.shape
        size = (width, height)
        print(f"Detected frame size: {width}x{height}")
    except Exception as e:
        print(f"Error reading first frame image ({first_frame_path}) to get dimensions: {e}", file=sys.stderr)
        sys.exit(1)


    # Define the codec and create VideoWriter object
    # Try common FOURCC codes for MP4. 'mp4v' is often good for compatibility.
    # Other options: 'XVID', 'MJPG' (large files), 'avc1' (H.264, might need specific backend/libs)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        out = cv2.VideoWriter(str(output_file), fourcc, args.framerate, size)
        if not out.isOpened():
            raise IOError("VideoWriter failed to open. Check codec compatibility or permissions.")
    except Exception as e:
         print(f"Error initializing VideoWriter: {e}", file=sys.stderr)
         print("You might need to install additional codecs or try a different FOURCC code like 'XVID'.")
         sys.exit(1)


    # Write frames to video
    frames_written = 0
    for frame_file in frame_files:
        try:
            img = cv2.imread(str(frame_file))
            if img is not None:
                out.write(img)
                frames_written += 1
            else:
                print(f"  Warning: Could not read frame image {frame_file}. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"  Error processing frame file {frame_file}: {e}", file=sys.stderr)


    out.release() # Release the VideoWriter
    print(f"Finished writing timelapse video. Wrote {frames_written} frames.")

    if frames_written > 0:
        print(f"Timelapse created successfully: {output_file.resolve()}")
        # Clean up temporary frames if not requested to keep
        if not args.keep_frames:
            try:
                shutil.rmtree(args.tmp_dir)
                print(f"Cleaned up temporary frames directory: {args.tmp_dir}")
            except OSError as e:
                print(f"Warning: Could not remove temporary directory {args.tmp_dir}: {e}", file=sys.stderr)
    else:
        print("Error: Failed to write any frames to the timelapse video.", file=sys.stderr)
        # Attempt to remove potentially empty output file
        try:
            output_file.unlink(missing_ok=True)
        except OSError:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()