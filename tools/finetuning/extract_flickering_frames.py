import cv2
import os
import glob
from ultralytics import YOLO

# --- CONFIGURATION ---
INPUT_FOLDER = "videos_to_test/"
OUTPUT_FOLDER = "dataset_active_learning/"
MODEL_PATH = "runs/detect/train7/weights/best_openvino_model/"

# Detection Threshold
# We want to be strict. If confidence drops below this, we consider it a "Gap".
CONF_THRESHOLD = 0.6 

# Gap Settings (Flicker detection)
# Minimum frames to consider it a gap (to avoid micro-noise)
MIN_GAP_FRAMES = 1 
# Maximum frames for a gap. 
# If a person disappears for more than 60 frames (e.g. 2-3 seconds), 
# we assume they actually left the frame, so we DON'T save those frames.
MAX_GAP_FRAMES = 60 

# To save disk space/labeling time, only save every Nth frame from the gap
SAVE_EVERY_N_FRAMES = 4 
# ---------------------

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    video_files = glob.glob(os.path.join(INPUT_FOLDER, "*.mp4"))
    print(f"Found {len(video_files)} videos. Scanning for flickering...")
    print("-" * 60)

    total_extracted = 0
    for video_path in video_files:
        count = process_video_gaps(video_path, model)
        total_extracted += count
        
    print("-" * 60)
    print(f"Done! Total new frames extracted: {total_extracted}")
    print(f"Check folder: {OUTPUT_FOLDER}")

def process_video_gaps(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # State Variables
    person_was_present = False
    gap_buffer = [] # Stores (frame_data, frame_index)
    frames_since_last_seen = 0
    saved_count = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # Run Inference (Fast, no plotting needed)
        results = model(frame, imgsz=640, verbose=False)
        
        # Check if ANY person is detected with high confidence
        person_detected_now = False
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            # Assuming class 0 is Person
            if cls == 0 and conf >= CONF_THRESHOLD:
                person_detected_now = True
                break

        # --- LOGIC CORE ---

        if person_detected_now:
            # SCENARIO: Person is visible
            
            # 1. Did we just close a gap? (Flicker detected)
            if len(gap_buffer) > 0:
                # We had a buffer of "empty" frames, and now the person is back.
                # This means the buffer frames were FALSE NEGATIVES. Save them!
                saved_count += save_buffer(gap_buffer, video_basename)
                gap_buffer = [] # Clear buffer
            
            # 2. Reset counters
            person_was_present = True
            frames_since_last_seen = 0
            
        else:
            # SCENARIO: No person visible (or low confidence)
            
            if person_was_present:
                # We rely on "person_was_present" to ignore empty streets before anyone arrives.
                frames_since_last_seen += 1
                
                # Check if the gap is still within "Flicker" range
                if frames_since_last_seen <= MAX_GAP_FRAMES:
                    # Potential false negative, add to buffer
                    # We copy the frame to ensure we don't store reference to changing buffer
                    gap_buffer.append((frame.copy(), frame_idx))
                else:
                    # Gap is too long (person likely left).
                    # Discard buffer and reset state.
                    gap_buffer = [] 
                    person_was_present = False # Reset logic until next person arrives

    cap.release()
    if saved_count > 0:
        print(f"{video_basename:<30} | Extracted {saved_count} frames")
    return saved_count

def save_buffer(buffer, video_name):
    count = 0
    # buffer is list of tuples: (image, original_frame_index)
    
    # Filter based on MIN_GAP
    if len(buffer) < MIN_GAP_FRAMES:
        return 0

    for i, (img, idx) in enumerate(buffer):
        # Skip frames to save space (SAVE_EVERY_N_FRAMES)
        if i % SAVE_EVERY_N_FRAMES != 0:
            continue
            
        filename = f"{video_name}_frame_{idx}.jpg"
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, img)
        count += 1
        
    return count

if __name__ == "__main__":
    main()