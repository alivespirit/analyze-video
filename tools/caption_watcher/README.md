# Windows Video Watcher & Viewer

This Python script monitors a specified folder (and its subfolders) on Windows for newly created `.mp4` image files. When a new image appears, it is automatically opened in the default app (preferably VLC). If multiple files appear simultaneously, they are processed one by one.

## Features

*   **Recursive Folder Monitoring:** Watches a directory and all its subdirectories.
*   **Specific File Type:** Targets `.mp4` files by default (configurable).
*   **Default Viewer:** Opens images using the system's default application for MP4.
*   **Sequential Processing:** Handles multiple new files one after another from a queue.

## Requirements

*   Windows Operating System
*   Python 3.10+
*   The following Python libraries:
    *   `watchdog` (for file system monitoring)
    *   `pygetwindow` (for checking if window is still opened)
*   VLC configured as a default app for MP4, with enabled option to close after play and optionally to open fullscreen.

## Installation

1.  Ensure you have Python installed on your Windows system.
2.  Open a command prompt or PowerShell and install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Save the script to a file, for example, `main.py`.
2.  Open a command prompt or PowerShell.
3.  Navigate to the directory where you saved the script.
4.  Run the script:
    ```bash
    python main.py
    ```
5.  The script will start monitoring. When you add new `.mp4` files to the specified folder or its subfolders, they should be opened, displayed, and then the viewer should close.
6.  To stop the script, press `Ctrl+C` in the console window where it's running.

## Configuration

The following parameters can be easily modified at the top of the `main.py` script:

*   `MONITOR_FOLDER = r"\\UX305CA\nas\xiaomi_camera_videos\04cf8c6b201d"`: Change this to point to the directory you want to watch.
*   `FILE_EXTENSIONS = ".mp4"`: Change this string to monitor different file types (e.g., `".png"`). Note that the default viewer might change depending on the file type.
