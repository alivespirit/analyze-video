# Windows Image Watcher & Viewer

This Python script monitors a specified folder (and its subfolders) on Windows for newly created `.jpg` image files. When a new image appears, it is automatically opened in the default Windows image viewer, displayed for a configurable duration (default 5 seconds), brought to the foreground (activated), and then the script attempts to close the viewer window. If multiple files appear simultaneously, they are processed one by one.

## Features

*   **Recursive Folder Monitoring:** Watches a directory and all its subdirectories.
*   **Specific File Type:** Targets `.jpg` files by default (configurable).
*   **Default Viewer:** Opens images using the system's default application for JPEGs.
*   **Timed Display:** Shows each image for a set number of seconds.
*   **Window Activation:** Brings the image viewer window to the foreground when an image is opened.
*   **Automatic Closing:** Attempts to close the image viewer window after the display duration using `pygetwindow`.
*   **Fallback Closing:** If `pygetwindow` fails to close the window by title, it can attempt to use `taskkill` on common image viewer process names (less precise).
*   **Sequential Processing:** Handles multiple new files one after another from a queue.

## Requirements

*   Windows Operating System
*   Python 3.6+
*   The following Python libraries:
    *   `watchdog` (for file system monitoring)
    *   `pygetwindow` (for finding, activating, and closing application windows)

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
5.  The script will prompt you to: `Enter the full path to the folder to monitor:`. Provide the full path to the directory you want to watch and press Enter.
6.  The script will start monitoring. When you add new `.jpg` files to the specified folder or its subfolders, they should be opened, displayed, and then the viewer should close.
7.  To stop the script, press `Ctrl+C` in the console window where it's running.

## Configuration

The following parameters can be easily modified at the top of the `main.py` script:

*   `MONITOR_FOLDER = r"\\UX305CA\nas\xiaomi_camera_videos\04cf8c6b201d"`: Change this to point to the directory you want to watch.
*   `FILE_EXTENSION = ".jpg"`: Change this string to monitor different file types (e.g., `".png"`). Note that the default viewer might change depending on the file type.
*   `VIEW_DURATION = 5`: Change this integer to set the number of seconds each image is displayed.
*   `KNOWN_VIEWER_PROCESS_FALLBACK = ["Photos.exe", "mspaint.exe", "PhotoViewer.dll", "ImageGlass.exe"]`: This is a list of executable names. If `pygetwindow` cannot identify and close the image viewer window by its title, the script will attempt to use `taskkill` with these process names as a last resort. This is less precise and might close unrelated instances of these applications. You can add or remove process names as needed for your default viewer.

## Important Notes & Limitations

*   **Window Closing Reliability:** The script's ability to find and close the image viewer window relies heavily on the behavior of your default image viewer application.
    *   It primarily tries to find the window by matching the image filename (without extension) in the window's title (e.g., "MyImage - Photos"). This might not work if your viewer uses a significantly different title format.
    *   Some viewers, especially modern UWP apps (like the Windows "Photos" app), can be tricky to control programmatically. The `pygetwindow.close()` method sends a standard close message, which is usually effective.
*   **Window Activation vs. Always-on-Top:** The script uses `pygetwindow`'s `activate()` method to bring the image viewer to the foreground. This is *not* the same as making it "always on top." If you click on another window, it can still cover the image viewer. For an "always on top" behavior, integration with `pywin32` would be necessary (see script comments in previous versions for an example).
*   **Fallback Closing (`taskkill`):** The `taskkill` fallback is a blunt instrument. If used, it will attempt to terminate any process matching the names in `KNOWN_VIEWER_PROCESS_FALLBACK`. Use with caution and configure the list appropriately for your system.
*   **Windows Specific:** While the `watchdog` library is cross-platform, the parts of the script responsible for opening files (`os.startfile`) and managing windows (`pygetwindow` in this context, `taskkill`) are specific to Windows.

## Troubleshooting

*   **Window Not Closing:**
    *   Check if the `KNOWN_VIEWER_PROCESS_FALLBACK` list includes the correct executable name for your default image viewer.
    *   The window title matching might be failing. You might need to adjust the logic in `find_activate_and_close_window` if your viewer's title doesn't include the filename.
    *   Some viewers might ignore the standard close command.
*   **Script Errors:** Ensure all dependencies are installed correctly.

---

Feel free to adapt this further if you make more changes to the script!