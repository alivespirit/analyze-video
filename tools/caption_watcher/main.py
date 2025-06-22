import os
import time
import queue
import subprocess
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pygetwindow as gw # For finding, activating, maximizing, and closing the window

# --- Configuration ---
MONITOR_FOLDER = r"\\UX305CA\nas\analyze-video\temp"
FILE_EXTENSIONS = [".jpg", ".mp4"]
VIEW_DURATION = 5  # seconds
KNOWN_VIEWER_PROCESS_FALLBACK = ["Photos.exe", "mspaint.exe", "PhotoViewer.dll", "ImageGlass.exe"]
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

def find_activate_maximize_and_close_window(image_filepath, duration_seconds): # Renamed function
    """
    Tries to find the window that opened the image_filepath, activate it, maximize it,
    and close it after duration_seconds.
    """
    opened_window = None
    filename_stem = os.path.splitext(os.path.basename(image_filepath))[0].lower()
    window_title_for_messages = f"window for {filename_stem}"

    time.sleep(1.5) # Give app time to launch and window to appear

    for _ in range(10): # Try for up to 5 seconds (10 * 0.5s)
        if stop_event.is_set(): return
        try:
            all_windows = gw.getAllWindows()
            possible_windows = [
                w for w in all_windows
                if filename_stem in w.title.lower() and w.visible and not w.isMinimized
            ]

            if possible_windows:
                active_window = gw.getActiveWindow()
                if active_window and filename_stem in active_window.title.lower() and \
                   active_window.visible and not active_window.isMinimized:
                    opened_window = active_window
                else:
                    opened_window = possible_windows[-1]
                
                window_title_for_messages = opened_window.title
                print(f"Found window: '{window_title_for_messages}' for {filename_stem}")
                
                if opened_window:
                    # --- Activate window ---
                    try:
                        opened_window.activate()
                        print(f"Window '{window_title_for_messages}' activated.")
                        time.sleep(0.1) # Small delay for activation
                    except gw.PyGetWindowException as e_activate:
                        if "Error code from Windows: 0" in str(e_activate):
                            print(f"Warning during activation for '{window_title_for_messages}': {e_activate}. OS reported success (code 0), proceeding.")
                        else:
                            print(f"Error trying to activate window '{window_title_for_messages}' with pygetwindow: {e_activate}")
                    except Exception as e_general_activate:
                        print(f"Unexpected error during window activation for '{window_title_for_messages}': {e_general_activate}")

                    # --- Maximize window ---
                    # Ensure window object is still valid before trying to maximize
                    try:
                        if opened_window.visible: # Check if still visible before maximizing
                            opened_window.maximize()
                            print(f"Window '{window_title_for_messages}' maximized.")
                            time.sleep(0.1) # Small delay for maximization
                        else:
                            print(f"Window '{window_title_for_messages}' not visible, skipping maximization.")
                    except gw.PyGetWindowException as e_maximize: # Catch errors if window closed or handle invalid
                        print(f"Error trying to maximize window '{window_title_for_messages}' with pygetwindow: {e_maximize}")
                    except Exception as e_general_maximize:
                        print(f"Unexpected error during window maximization for '{window_title_for_messages}': {e_general_maximize}")
                # --- End activate and maximize window ---
                break
        except Exception as e_find:
            print(f"Error while searching for window for {filename_stem}: {e_find}")
            pass
        time.sleep(0.5)

    print(f"Displaying '{os.path.basename(image_filepath)}' for {duration_seconds} seconds...")
    
    for _ in range(duration_seconds * 2):
        if stop_event.is_set():
            break
        time.sleep(0.5)
    
    if stop_event.is_set():
        print("Stop event received, attempting to close window early if found.")

    if opened_window:
        try:
            window_title_for_messages = opened_window.title
        except gw.PyGetWindowException:
            print(f"Window for {filename_stem} closed before explicit close attempt.")
            return
        except Exception:
            pass

        print(f"Attempting to close window: '{window_title_for_messages}'")
        try:
            # Before closing, it might be good to restore if it was maximized by us
            # However, simply closing is usually fine. If you want to restore:
            # try:
            #     if opened_window.isMaximized: # Check if it's actually maximized
            #         opened_window.restore()
            #         time.sleep(0.1)
            # except Exception: pass # Ignore errors during restore

            if opened_window.visible:
                opened_window.close()
                print("Window close command sent via pygetwindow.")
                time.sleep(0.5)
                try:
                    if opened_window.visible:
                        print(f"Warning: Window '{window_title_for_messages}' still appears to be visible.")
                    else:
                        print(f"Window '{window_title_for_messages}' is no longer visible.")
                except gw.PyGetWindowException:
                    print(f"Window '{window_title_for_messages}' confirmed closed (access failed).")
            else:
                print(f"Window '{window_title_for_messages}' was found but is no longer visible.")
        except gw.PyGetWindowException as e:
            print(f"pygetwindow error for '{window_title_for_messages}': {e}")
        except Exception as e_generic:
            print(f"Unexpected error closing window '{window_title_for_messages}' with pygetwindow: {e_generic}")
    else:
        print(f"Could not find a specific window for '{os.path.basename(image_filepath)}' to close.")
        print("As a fallback, attempting to close known image viewers (this is less precise).")
        for proc_name in KNOWN_VIEWER_PROCESS_FALLBACK:
            if stop_event.is_set(): break
            try:
                subprocess.run(["taskkill", "/IM", proc_name, "/F"], capture_output=True, check=False, creationflags=0x08000000)
                print(f"Attempted taskkill on {proc_name}")
            except FileNotFoundError:
                print("taskkill command not found.")
            except Exception as e:
                print(f"Error during taskkill for {proc_name}: {e}")
            time.sleep(0.1)


def file_processor_worker():
    """Processes files from the queue one by one."""
    while not stop_event.is_set():
        try:
            filepath = file_queue.get(timeout=1)
            if filepath is None:
                break
            
            print(f"\nProcessing: {filepath}")
            try:
                print(f"Opening {filepath} with default application...")
                os.startfile(filepath)
                find_activate_maximize_and_close_window(filepath, VIEW_DURATION) # Renamed function call
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
    print(f"Files will be displayed for {VIEW_DURATION} seconds each.")
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
            processor_thread.join(timeout=VIEW_DURATION + 5)

        if processor_thread.is_alive():
            print("Processor thread did not stop gracefully.")

        print("Script stopped.")