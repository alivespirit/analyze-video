import cv2
import numpy as np
import json  # ### NEW ### For saving/loading the ROI
import time  # ### NEW ### For timing the recording
import os    # ### NEW ### For creating an output directory

# --- Configuration ---
VIDEO_SOURCE = r"C:\NAS\opencv\44M16S_1763901856.mp4"
MIN_CONTOUR_AREA = 2000
ROI_CONFIG_FILE = r"C:\NAS\roi.json"
OUTPUT_DIR = "motion_clips" # ### NEW ### Directory to save clips
POST_MOTION_RECORD_SECONDS = 2 # ### NEW ### How many seconds to record after motion stops

MIN_SOLIDITY = 0.8             # A person is a very solid shape. 1.0 is a perfect rectangle.
MIN_ASPECT_RATIO = 1.0          # Height should be at least 1.2x its width for an upright person.

# --- Global variables for drawing the ROI ---
roi_points = []
drawing_complete = False

# ### NEW ### Function to save ROI points to a file
def save_roi(points, file_path):
    with open(file_path, 'w') as f:
        json.dump(points, f)
    print(f"ROI saved to {file_path}")

def is_likely_person_or_object(contour):
    """
    Checks if a contour has the shape properties of a person or solid object.
    Returns True if it passes the checks, False otherwise.
    """
    # 1. Check Area (basic filter)
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return False

    # 2. Check Aspect Ratio
    #x, y, w, h = cv2.boundingRect(contour)
    #aspect_ratio = h / float(w) if w > 0 else 0
    #if aspect_ratio < MIN_ASPECT_RATIO:
    #    return False

    # 3. Check Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    if solidity < MIN_SOLIDITY:
        return False
    
    # If all checks pass, it's likely a person/solid object
    return True

# ### NEW ### Function to load ROI points from a file
def load_roi(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            points = json.load(f)
            print(f"ROI loaded from {file_path}")
            return np.array(points, dtype=np.int32)
    return None

def draw_roi(event, x, y, flags, param):
    """Mouse callback function to draw the ROI polygon."""
    global roi_points, drawing_complete
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_complete:
        roi_points.append([x, y]) # Save as list of lists for JSON
        print(f"Added point: ({x}, {y})")

def get_roi_from_user(frame):
    """Displays the frame and lets the user draw a polygon."""
    # (This function is the same as before, but note roi_points now uses lists)
    global roi_points, drawing_complete
    window_name = "Draw ROI - Click points, press 'd' when done, 'r' to reset"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_roi)
    print("Please define the Region of Interest (ROI).")
    print("Click to add points, 'd' when done, 'r' to reset, 'q' to quit.")

    while True:
        temp_frame = frame.copy()
        if len(roi_points) > 0:
            cv2.polylines(temp_frame, [np.array(roi_points)], isClosed=False, color=(0, 255, 0), thickness=2)
            for point in roi_points:
                cv2.circle(temp_frame, tuple(point), 5, (0, 255, 0), -1)
        
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('d'):
            if len(roi_points) > 2:
                drawing_complete = True
                cv2.destroyWindow(window_name)
                # Save the ROI for next time
                save_roi(roi_points, ROI_CONFIG_FILE)
                return np.array(roi_points, dtype=np.int32)
            else:
                print("You need at least 3 points.")
        elif key == ord('r'):
             roi_points = []
             drawing_complete = False
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

def main():
    # ### NEW ### Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # ### NEW ### Try to load ROI, if it fails, ask user to draw it
    roi_poly_points = load_roi(ROI_CONFIG_FILE)
    if roi_poly_points is None:
        roi_poly_points = get_roi_from_user(first_frame)
    
    if roi_poly_points is None:
        print("ROI not defined. Exiting.")
        cap.release()
        return

    height, width, _ = first_frame.shape
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_poly_points], 255)
    
    cv2.imshow("ROI Mask", roi_mask)
    
    backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

    # ### NEW ### Variables for video recording
    is_recording = False
    last_motion_time = 0
    video_writer = None
    
    # ### NEW ### Get video properties for the writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        fg_mask = backSub.apply(roi_frame)
        
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Instead of just checking area, we use our sophisticated function
        person_like_contours = [cnt for cnt in contours if is_likely_person_or_object(cnt)]
        
        motion_detected = False
        for cnt in person_like_contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break # No need to check other contours if one is found
        
        # ### NEW ### Recording Logic
        if motion_detected:
            last_motion_time = time.time()
            if not is_recording:
                is_recording = True
                # Start a new video writer
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_filename = os.path.join(OUTPUT_DIR, f"motion_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
                video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
                print(f"Started recording to {output_filename}")

        if is_recording:
            # Write the current frame to the video file
            video_writer.write(frame)
            
            # Check if we should stop recording
            if time.time() - last_motion_time > POST_MOTION_RECORD_SECONDS:
                is_recording = False
                video_writer.release()
                video_writer = None
                print("Stopped recording.")

        # Display status text
        status_text = "Recording" if is_recording else "Monitoring"
        text_color = (0, 0, 255) if is_recording else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.polylines(frame, [roi_poly_points], isClosed=True, color=(255, 255, 0), thickness=2)

        cv2.imshow("Motion Detector", frame)
        cv2.imshow("Cleaned Foreground Mask", fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # ### NEW ### Release the writer if we're still recording when the video ends
    if video_writer is not None:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()