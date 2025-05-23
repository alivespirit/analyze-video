# Timelapse Creator for Xiaomi Camera Videos

This tool creates a timelapse video by extracting a specific frame (frame #30) from each `.mp4` video file recorded by Xiaomi cameras. It uses OpenCV for frame extraction and video assembly.

## Features

- Processes video files organized by camera ID and date/hour folders.
- Extracts a single frame from each video for the timelapse.
- Supports batch processing for multiple cameras and dates.
- Cleans up temporary files automatically (optional).
- Customizable output framerate and paths.

## Requirements

- Python 3.7+
- [OpenCV](https://pypi.org/project/opencv-python/) (`pip install opencv-python`)

## Usage

```bash
python3 main.py [options]
```

### Options

| Option                | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `-b`, `--base-dir`    | Base directory containing camera folders (default: `/mnt/nas/xiaomi_camera_videos`)           |
| `-c`, `--camera-id`   | Camera ID subfolder name, or `*` for all cameras (default: `04cf8c6b201d`)                   |
| `-d`, `--date-pattern`| Date/hour pattern, e.g. `20250422*`, `2025042210`, `*` (default: `2025*`)                    |
| `-t`, `--tmp-dir`     | Temporary directory for extracted frames (default: `./xiaomi_frames_tmp_cv2`)                |
| `-o`, `--output`      | Output timelapse file path (default: `timelapse_CAMID_DATE_cv2.mp4`)                         |
| `-f`, `--framerate`   | Framerate for the output video (default: `60`)                                               |
| `--keep-frames`       | Do not delete the temporary frame directory after completion                                 |

### Example

```bash
python3 main.py -c 04cf8c6b201d -d 20250422* -o my_timelapse.mp4
```

This command processes all videos from camera `04cf8c6b201d` for the date `2025-04-22` and creates `my_timelapse.mp4`.

## Directory Structure

Expected input directory layout:

```
/mnt/nas/xiaomi_camera_videos/
  └── <camera_id>/
      └── <YYYYMMDDHH>/
          └── *.mp4
```

## How It Works

1. **Finds video files** matching the given camera ID and date pattern.
2. **Extracts frame #30** from each video and saves it as a temporary image.
3. **Sorts frames chronologically** based on folder and filename.
4. **Assembles frames** into a timelapse video using OpenCV.
5. **Cleans up** temporary frames unless `--keep-frames` is specified.

## Notes

- The script expects Xiaomi video files to be named with a timestamp suffix (e.g., `..._1234567890.mp4`).
- If no frames are extracted, no timelapse is created.
- For best results, ensure all videos have the same resolution.

## Troubleshooting

- If you encounter codec errors, try installing additional codecs or changing the FOURCC code in the script (e.g., to `XVID`).
- Make sure you have read/write permissions for the specified directories.

## License

MIT License
