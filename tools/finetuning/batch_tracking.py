import cv2
import os
import glob
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
INPUT_FOLDER = "videos_to_test/"
OUTPUT_FOLDER = "processed_results/"
MODEL_PATH = "runs/detect/train8/weights/best_openvino_model/"

SAVE_OUTPUT_VIDEO = True
CONF_THRESHOLD = 0.45
LINE_Y = 860  # Counting Line Y-Coordinate

# Custom Colors
COLOR_PERSON = (100, 200, 0)
COLOR_CAR = (200, 120, 0)
COLOR_DEFAULT = (255, 255, 255)
COLOR_LINE = (0, 255, 255)
# ---------------------

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return

    if SAVE_OUTPUT_VIDEO and not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 1. LOAD MODEL ONCE (Global)
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect')
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    video_files = glob.glob(os.path.join(INPUT_FOLDER, "*.mp4"))
    if not video_files:
        print("No .mp4 files found.")
        return

    print(f"Found {len(video_files)} videos. Starting batch processing...")
    print("-" * 115)
    print(f"{'Video Name':<30} | {'Time':<8} | {'FPS':<5} | {'Unique P':<9} | {'P Up':<6} | {'P Down':<7} | {'Unique C':<9}")
    print("-" * 115)

    for video_path in video_files:
        # Pass the LOADED model object
        process_single_video(video_path, model)

def process_single_video(video_path, model):
    # 2. RESET TRACKER
    # Reset internal tracker state to prevent "Ghost" tracks from previous video
    if hasattr(model, 'predictor') and model.predictor is not None:
        if hasattr(model.predictor, 'trackers') and model.predictor.trackers:
            model.predictor.trackers[0].reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    class_names = model.names

    out = None
    if SAVE_OUTPUT_VIDEO:
        filename = "tracked_" + os.path.basename(video_path)
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    id_mapping = {}
    class_counters = {} 
    
    unique_objects_detected = {
        'person': set(),
        'car': set()
    }

    # Crossing Logic
    previous_positions = {}
    persons_up = 0
    persons_down = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Run Tracking
        results = model.track(frame, imgsz=640, conf=CONF_THRESHOLD, persist=True, verbose=False, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            global_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            for box, global_id, cls, conf in zip(boxes, global_ids, clss, confs):
                label_name = class_names[cls]

                # --- MAPPING LOGIC ---
                if label_name not in class_counters: class_counters[label_name] = 1
                if global_id not in id_mapping:
                    id_mapping[global_id] = class_counters[label_name]
                    class_counters[label_name] += 1
                local_id = id_mapping[global_id]
                # ---------------------

                if label_name in unique_objects_detected:
                    unique_objects_detected[label_name].add(global_id)

                # --- CROSSING LOGIC ---
                y_center = int((box[1] + box[3]) / 2)
                
                if global_id in previous_positions:
                    prev_y = previous_positions[global_id]
                    if label_name == 'person':
                        # Moved DOWN (Y increased)
                        if prev_y < LINE_Y <= y_center: 
                            persons_down += 1
                        # Moved UP (Y decreased)
                        elif prev_y > LINE_Y >= y_center: 
                            persons_up += 1
                
                previous_positions[global_id] = y_center
                # ----------------------

                if SAVE_OUTPUT_VIDEO:
                    draw_tracked_box(frame, box, local_id, label_name, conf)

        if SAVE_OUTPUT_VIDEO:
            # Draw Yellow Line
            cv2.line(frame, (0, LINE_Y), (width, LINE_Y), COLOR_LINE, 1)
            cv2.putText(frame, f"Hvirtka Y={LINE_Y}", (10, LINE_Y - 10), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, COLOR_LINE, 1, cv2.LINE_AA)

        if out: out.write(frame)

    cap.release()
    if out: out.release()

    end_time = time.time()
    duration = end_time - start_time
    proc_fps = total_frames / duration if duration > 0 else 0

    video_name = os.path.basename(video_path)
    if len(video_name) > 28: video_name = video_name[:25] + "..."
    
    p_unique = len(unique_objects_detected['person'])
    c_unique = len(unique_objects_detected['car'])
    
    print(f"{video_name:<30} | {duration:.1f}s   | {proc_fps:.1f}  | {p_unique:<9} | {persons_up:<6} | {persons_down:<7} | {c_unique:<9}")

def draw_tracked_box(frame, box, local_id, label_name, conf):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    
    if label_name == 'person': color = COLOR_PERSON
    elif label_name == 'car': color = COLOR_CAR
    else: color = COLOR_DEFAULT

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label_text = f"{label_name} {local_id} {conf:.0%}"
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
    
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)
    cv2.putText(frame, label_text, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()