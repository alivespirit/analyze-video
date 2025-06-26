import os
import time
import queue
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pygetwindow as gw # For finding, activating, maximizing, and closing the window

# --- Configuration ---
MONITOR_FOLDER = r"\\UX305CA\nas\analyze-video\temp"
FILE_EXTENSIONS = [".mp4"]
# --- End Configuration ---

file_queue = queue.Queue()
stop_event = threading.Event()

class NewFileHandler(FileSystemEventHandler):
  def on_created(self, event):
    if stop_event.is_set():
        return
    if not event.is_directory and any(event.src_path.lower().endswith(ext) for ext in FILE_EXTENSIONS):
        print(f"New file detected: {event.src_path}")
        file_queue.put(event.src_path)

def find_window(image_filepath):
    """
    Tries to find the window that opened the image_filepath and waits until it's closed.
    """
    filename_stem = os.path.splitext(os.path.basename(image_filepath))[0].lower()
    still_open = True

    time.sleep(1) # Give app time to launch and window to appear

    while still_open:
        try:
            still_open = False
            all_windows = gw.getAllWindows()
            for w in all_windows:
                if filename_stem in w.title.lower():
                    print(f"Found window: '{w.title}' for {filename_stem}")
                    still_open = True
                    time.sleep(1)  # Wait a bit before checking again

        except Exception as e_find:
            print(f"Error while searching for window for {filename_stem}: {e_find}")
            still_open = False
            pass
    
    print(f"Window for {filename_stem} is now closed.")
    return

def file_processor_worker():
    """Processes files from the queue one by one."""
    while not stop_event.is_set():
        try:
            filepath = file_queue.get(timeout=1)
            if filepath is None:
                break
            
            print(f"\nProcessing: {filepath}")
            try:
                print(f"Opening {filepath} with default application 2 times...")
                for _ in range(2):
                    os.startfile(filepath)
                    find_window(filepath)
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
            finally:
                file_queue.task_done()
                if stop_event.is_set():
                    break
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Critical error in file_processor_worker: {e}")
            time.sleep(1)

if __name__ == "__main__":
    print(f"Monitoring folder: {MONITOR_FOLDER} for new {FILE_EXTENSIONS} files.")
    print("Press Ctrl+C to stop.")

    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITOR_FOLDER, recursive=True)
    
    processor_thread = threading.Thread(target=file_processor_worker)
    
    observer.start()
    processor_thread.start()

    try:
        while processor_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping script...")
    finally:
        stop_event.set()
        
        print("Stopping observer...")
        observer.stop()
        observer.join()
        
        print("Signaling file processor to stop and waiting for it to finish...")
        if processor_thread.is_alive():
            try:
                file_queue.put_nowait(None) 
            except queue.Full:
                pass
            processor_thread.join(timeout=5)

        if processor_thread.is_alive():
            print("Processor thread did not stop gracefully.")

        print("Script stopped.")