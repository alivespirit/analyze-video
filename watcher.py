# watcher.py
import subprocess
import time
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration ---
# SCRIPT_TO_WATCH should be just the filename if it's in the same dir as watcher.py
SCRIPT_TO_WATCH_FILENAME = "main.py"
PYTHON_EXECUTABLE = sys.executable # Uses the same python interpreter that runs watcher.py

# --- Dynamically determine paths ---
# Get the directory where THIS SCRIPT (watcher.py) is located.
# This is crucial for finding main.py correctly when double-clicked.
WATCHER_DIR = os.path.dirname(os.path.abspath(__file__)) # <--- KEY CHANGE

# Construct the absolute path to the script to watch
SCRIPT_PATH = os.path.join(WATCHER_DIR, SCRIPT_TO_WATCH_FILENAME)

# Directory containing the script to watch (which is also WATCHER_DIR in this case)
WATCH_DIRECTORY = WATCHER_DIR

class ChangeHandler(FileSystemEventHandler):
    def __init__(self, script_path_to_run, script_filename_to_monitor):
        self.script_path_to_run = script_path_to_run # Absolute path to main.py
        self.script_filename_to_monitor = script_filename_to_monitor # Just "main.py"
        self.process = None
        self.start_script()

    def start_script(self):
        if self.process and self.process.poll() is None:
            print(f"Watcher: Terminating old process (PID: {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Watcher: Old process (PID: {self.process.pid}) did not terminate, killing...")
                self.process.kill()
                self.process.wait()
            print("Watcher: Old process terminated.")

        print(f"Watcher: Starting '{self.script_path_to_run}'...")
        # When starting the script, its working directory will be inherited
        # from the watcher. If main.py needs to know its own directory,
        # it should use os.path.dirname(os.path.abspath(__file__)) itself.
        # However, for subprocess.Popen, if main.py needs to access files
        # relative to its own location, it's best if it determines its own path.
        # We pass the absolute path to Popen, so it knows what to execute.
        self.process = subprocess.Popen([PYTHON_EXECUTABLE, self.script_path_to_run]) # Removed creationflags for now
        print(f"Watcher: '{os.path.basename(self.script_path_to_run)}' started with PID {self.process.pid}.")

    def on_modified(self, event):
        # event.src_path will be the absolute path to the modified file.
        # We need to compare the filename part with SCRIPT_TO_WATCH_FILENAME
        # or the full path with SCRIPT_PATH.
        # Comparing full path is more robust if WATCH_DIRECTORY could contain other .py files.
        if not event.is_directory and event.src_path == self.script_path_to_run: # Compare absolute paths
            print(f"Watcher: File '{os.path.basename(event.src_path)}' has been modified. Restarting...")
            self.start_script()

    def stop(self):
        if self.process and self.process.poll() is None:
            print(f"Watcher: Shutting down. Terminating '{os.path.basename(self.script_path)}' (PID: {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print(f"Watcher: '{os.path.basename(self.script_path)}' terminated.")

if __name__ == "__main__":
    try:
        # The SCRIPT_PATH is now absolute and correctly determined above.
        if not os.path.exists(SCRIPT_PATH):
            print(f"CRITICAL ERROR: Script to watch '{SCRIPT_PATH}' not found.")
            print(f"Watcher script directory: {WATCHER_DIR}")
            print(f"Expected main.py filename: {SCRIPT_TO_WATCH_FILENAME}")
            input("Press Enter to exit.")
            sys.exit(1)

        print(f"Watcher: Monitoring directory '{WATCH_DIRECTORY}' for changes to '{SCRIPT_TO_WATCH_FILENAME}'.")
        print(f"Watcher: Full path to monitored script: {SCRIPT_PATH}")
        print(f"Watcher: Using Python executable: {PYTHON_EXECUTABLE} to run {SCRIPT_TO_WATCH_FILENAME}")
        print(f"Watcher: This script (watcher.py) is running with: {sys.executable}")
        print("Watcher: Press Ctrl+C in this window to stop the watcher (if run from cmd).")
        print("          (Closing this window will also stop the watcher and main.py)")


        event_handler = ChangeHandler(SCRIPT_PATH, SCRIPT_TO_WATCH_FILENAME)
        observer = Observer()
        # Watch the directory containing the script (WATCH_DIRECTORY), not recursive
        observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
        observer.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Watcher: KeyboardInterrupt received. Stopping observer...")
    finally:
        observer.stop()
        observer.join()
        event_handler.stop() # Ensure the managed script is also stopped
        print("Watcher: Exited cleanly.")