import os

# Set the environment variable to allow duplicate OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import torch
import pickle
import numpy as np
import time
import threading
import queue
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from data import (
    load_database, save_database, recognize_face, add_person,  DEFAULT_THRESHOLD,update_person, enhance_face_for_recognition, track_faces
)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Debug mode for troubleshooting
DEBUG = False
# Default detection threshold
DETECTION_THRESHOLD = DEFAULT_THRESHOLD 

# Load YOLO model for face detection
model = YOLO('yolov8n-face.pt')

# Load FaceNet model for face recognition/embedding
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Queue for adding new people without blocking the main thread
new_person_queue = queue.Queue()

# Async Face Processor using threading
class FaceProcessor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.results = []
        self.pending_unknown_faces = []  # Store unknown faces for UI to handle
        self.lock = threading.Lock()
        self.running = True
        self.daemon = True  # Make thread exit when main program exits
        self.face_detection_threshold = DETECTION_THRESHOLD
        self.skip_frames = 0  # Counter to process every Nth frame for better performance
        self.last_saved_time = time.time()  # For periodic automatic saving
        
    def run(self):
        while self.running:
            # Process queue for adding new people (non-blocking)
            try:
                while not new_person_queue.empty():
                    data = new_person_queue.get_nowait()
                    if len(data) == 3:  # Normal add
                        name, embedding, room = data
                        person_id = add_person(name, embedding, room)
                    elif len(data) == 4:  # Add with existing person_id
                        name, embedding, room, person_id = data
                        add_person(name, embedding, room, person_id)
                    new_person_queue.task_done()
            except queue.Empty:
                pass
            
            # Periodic automatic save (every 5 minutes)
            if time.time() - self.last_saved_time > 300:  # 5 minutes
                save_database()
                self.last_saved_time = time.time()
            
            # Process video frame
            if self.frame is not None:
                with self.lock:
                    img = self.frame.copy()
                    self.frame = None
                
                # Skip some frames to improve performance
                self.skip_frames = (self.skip_frames + 1) % 2
                if self.skip_frames != 0:
                    time.sleep(0.01)
                    continue
                
                try:
                    # Detect faces using YOLO
                    results = model(img)
                    yolo_boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract boxes
                    yolo_confs = results[0].boxes.conf.cpu().numpy()  # Extract confidence scores
                    
                    if DEBUG:
                        print(f"YOLO detected {len(yolo_boxes)} faces")
                    
                    if len(yolo_boxes) > 0:
                        boxes = []
                        probs = []
                        for i, box in enumerate(yolo_boxes):
                            conf = yolo_confs[i]
                            if conf >= self.face_detection_threshold:
                                boxes.append(box)
                                probs.append(conf)
                                
                        if len(boxes) == 0:
                            self.results = []
                            time.sleep(0.01)
                            continue

                        face_crops = []
                        valid_boxes = []
                        for box in boxes:
                            x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
                            if (x1 < x2 and y1 < y2 and x2 - x1 >= 20 and y2 - y1 >= 20):
                                # Add margins around face for better recognition
                                h_margin = int((y2 - y1) * 0.15)
                                w_margin = int((x2 - x1) * 0.15)
                                y1_m = max(0, y1 - h_margin)
                                y2_m = min(img.shape[0], y2 + h_margin)
                                x1_m = max(0, x1 - w_margin)
                                x2_m = min(img.shape[1], x2 + w_margin)

                                face = img[y1_m:y2_m, x1_m:x2_m]
                                if face.shape[0] > 0 and face.shape[1] > 0:
                                    # Enhance face image for better recognition
                                    face = enhance_face_for_recognition(face)
                                    
                                    # Resize to FaceNet input size
                                    face = cv2.resize(face, (160, 160))
                                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                                    face_crops.append(face)
                                    valid_boxes.append(box)

                        if not face_crops:
                            self.results = []
                            time.sleep(0.01)
                            continue

                        try:
                            # Convert face crops to tensors for FaceNet
                            faces_tensor = torch.stack([
                                torch.from_numpy(face.transpose((2, 0, 1))).float() / 255.0
                                for face in face_crops
                            ]).to(device)

                            # Generate embeddings using FaceNet
                            with torch.no_grad():
                                embeddings = facenet(faces_tensor).detach().cpu().numpy()
                            
                            # Normalize embeddings
                            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                            current_results = []
                            self.pending_unknown_faces = []
                            recognized_faces = []

                            # Match embeddings with database
                            for box, emb in zip(valid_boxes, embeddings):
                                # Use improved recognition with adaptive thresholding
                                name, info = recognize_face(emb, threshold=self.face_detection_threshold, top_k=3, use_adaptive_threshold=True)
                                x1 = int(box[0])
                                room = "Room One" if x1 < img.shape[1] // 2 else "Room Two"

                                if name is None:
                                    name = "Unknown"
                                    self.pending_unknown_faces.append((box.astype(int).tolist(), emb, room))
                                else:
                                    person_id = info.get("person_id")
                                    if info.get("confidence", 0) > 0.8 and info.get("room") != room:
                                        update_person(person_id, room=room)
                                
                                recognized_faces.append((name, info))
                                current_results.append((box, name, room, info))
                            
                            # Apply tracking for consistent face recognition
                            tracked_results = track_faces(img, valid_boxes, recognized_faces)
                            
                            # Update results with tracking information if needed
                            # This section can be expanded to utilize the tracking data
                            
                            self.results = current_results

                        except Exception as e:
                            print(f"Error processing face tensor: {e}")
                            self.results = []
                            if DEBUG:
                                import traceback
                                traceback.print_exc()
                    else:
                        self.results = []
                except Exception as e:
                    print(f"Error in face processing: {e}")
                    if DEBUG:
                        import traceback
                        traceback.print_exc()
                    self.results = []

            time.sleep(0.01)

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def stop(self):
        self.running = False

def register_new_face(name, embedding, room):
    """
    Register a new face in the database.

    :param name: The name of the person.
    :param embedding: The face embedding.
    :param room: The room where the face was detected.
    """
    # Add the person to the database
    new_person_queue.put((name, embedding, room))
    print(f"Adding {name} to the database...")

def main():
    # Load the face recognition database
    load_database()

    # Start the face processing thread
    processor = FaceProcessor()
    processor.start()

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30fps if possible
    
    prev_time = 0

    show_help = True
    global DEBUG
    
    # Track the last frame when we checked for unknown faces
    last_unknown_check = 0
    # Flag to show "Add Person?" UI
    show_add_person = False
    current_unknown_idx = 0
    unknown_box = None
    unknown_embedding = None
    unknown_room = None
    name_input = ""
    input_active = False
    auto_dismiss_timer = 0  # Timer to auto-dismiss the popup if no action is taken
    
    # Text prompt position
    prompt_x = 20
    prompt_y = 60
    
    # Show database stats
    show_stats = False
    
    # Main video capture loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            height, width, _ = frame.shape

            # Update frame for processing
            processor.update_frame(frame)
            
            # Handle pending unknown faces - only show the UI when we're not already adding someone
            if not show_add_person and time.time() - last_unknown_check > 2 and processor.pending_unknown_faces:
                last_unknown_check = time.time()
                unknown_box, unknown_embedding, unknown_room = processor.pending_unknown_faces[0]
                show_add_person = True
                current_unknown_idx = 0
                name_input = ""
                input_active = False
                auto_dismiss_timer = time.time()  # Start the auto-dismiss timer
            
            # Draw recognition results
            if DEBUG:
                print(f"Number of results: {len(processor.results)}")  # Debugging
                
            for box, name, room, info in processor.results:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                if DEBUG:
                    print(f"Drawing box at: {x1, y1, x2, y2}")  # Debugging
                
                # Get confidence level and embedding count from info
                confidence = info.get("confidence", 0) if info else 0
                embed_count = info.get("embedding_count", 0) if info else 0
                
                # Determine if face is known or unknown
                is_known = name != "Unknown"
                
                # Set colors for known (green) vs unknown (red) faces
                # OpenCV uses BGR color format
                if is_known:
                    # Color gradient based on confidence: from light green (low) to bright green (high)
                    conf_factor = min(1.0, confidence)
                    box_color = (0, int(100 + 155 * conf_factor), 0)  # Brighter green for higher confidence
                    text_bg_color = (0, 70, 0)
                else:
                    box_color = (0, 0, 255)  # Red for unknown faces
                    text_bg_color = (70, 0, 0)
                
                # Draw thicker bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Add corners to the bounding box for better visibility
                corner_length = 15  # Length of corner lines
                thickness = 3       # Thickness of corner lines
                
                # Top-left corner
                cv2.line(frame, (x1, y1), (x1 + corner_length, y1), box_color, thickness)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_length), box_color, thickness)
                # Top-right corner
                cv2.line(frame, (x2, y1), (x2 - corner_length, y1), box_color, thickness)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_length), box_color, thickness)
                # Bottom-left corner
                cv2.line(frame, (x1, y2), (x1 + corner_length, y2), box_color, thickness)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_length), box_color, thickness)
                # Bottom-right corner
                cv2.line(frame, (x2, y2), (x2 - corner_length, y2), box_color, thickness)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_length), box_color, thickness)

                # Add label with name and room
                label = f"{name} ({room})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_width, label_height = label_size
                cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), text_bg_color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show "Add Person?" UI - improved version
            if show_add_person:
                # Check if we should auto-dismiss after 10 seconds of no interaction
                if time.time() - auto_dismiss_timer > 10 and not input_active:
                    show_add_person = False
                    if processor.pending_unknown_faces:
                        processor.pending_unknown_faces.pop(0)
                    continue
                
                # Highlight the unknown face we're asking about
                if unknown_box and len(unknown_box) == 4:
                    x1, y1, x2, y2 = unknown_box
                    face_width = x2 - x1
                    face_height = y2 - y1
                    
                    # Create a darker semi-transparent overlay for the face highlight
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1-5, y1-5), (x2+5, y2+5), (0, 165, 255), -1)  # Filled orange
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # Apply with 30% opacity
                    
                    # Draw a more prominent outline
                    cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 165, 255), 2)
                    
                    # Position the UI above the face if there's space, otherwise below
                    ui_height = 100 if input_active else 55
                    if y1 > ui_height + 10:  # If there's enough space above the face
                        ui_y = y1 - ui_height - 10
                    else:  # Place below the face
                        ui_y = y2 + 10
                    
                    # Position UI centered on the face
                    ui_width = max(220, face_width + 40)
                    ui_x = max(5, x1 - (ui_width - face_width) // 2)
                    
                    # Make sure UI doesn't go off-screen
                    if ui_x + ui_width > frame.shape[1] - 5:
                        ui_x = frame.shape[1] - ui_width - 5
                    
                    # Draw UI background with rounded corners
                    # Since OpenCV doesn't support rounded corners directly, we'll use an overlay
                    ui_overlay = frame.copy()
                    cv2.rectangle(ui_overlay, (ui_x, ui_y), (ui_x + ui_width, ui_y + ui_height), (40, 40, 45), -1)
                    cv2.addWeighted(ui_overlay, 0.8, frame, 0.2, 0, frame)
                    
                    # Add prompt text
                    prompt_text = f"Unknown face in {unknown_room}"
                    cv2.putText(frame, prompt_text, (ui_x + 10, ui_y + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Add action text
                    cv2.putText(frame, "Add to database? [Y/N]", (ui_x + 10, ui_y + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # If name input is active, show input field
                    if input_active:
                        # Input field background
                        cv2.rectangle(frame, (ui_x + 10, ui_y + 50), (ui_x + ui_width - 10, ui_y + 80), (60, 60, 65), -1)
                        
                        # Show name input with blinking cursor
                        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
                        cv2.putText(frame, f"Name: {name_input}{cursor}", (ui_x + 15, ui_y + 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Show instructions
                        cv2.putText(frame, "Enter: confirm | Esc: cancel", (ui_x + 15, ui_y + 95), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show instructions
            if show_help:
                help_text = "Press 1 to Quit | 2 to Toggle Debug | 3 to Save | 4 to Toggle Help"
                cv2.putText(frame, help_text, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show the frame
            cv2.imshow("Face Recognition", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                break
            elif key == ord('2'):
                DEBUG = not DEBUG
                print(f"DEBUG mode set to {DEBUG}")
            elif key == ord('3'):
                save_database()
                print("Database saved.")
            elif key == ord('4'):
                show_help = not show_help
            elif show_add_person:
                if key == ord('y') or key == ord('Y'):
                    input_active = True
                    auto_dismiss_timer = time.time()  # Reset timer when user interacts
                elif key == ord('n') or key == ord('N'):
                    show_add_person = False
                    input_active = False
                    # Go to next unknown face if there are more
                    if processor.pending_unknown_faces:
                        processor.pending_unknown_faces.pop(0)
                elif input_active:
                    auto_dismiss_timer = time.time()  # Reset timer when user types
                    if key == 27:  # ESC key
                        input_active = False
                        show_add_person = False
                        # Go to next unknown face if there are more
                        if processor.pending_unknown_faces:
                            processor.pending_unknown_faces.pop(0)
                    elif key == 13:  # Enter key
                        if name_input:
                            # Register the new face
                            register_new_face(name_input, unknown_embedding, unknown_room)
                            
                            # Reset UI state
                            show_add_person = False
                            input_active = False
                            
                            # Remove from pending list
                            if processor.pending_unknown_faces:
                                processor.pending_unknown_faces.pop(0)
                    elif key == 8 or key == 127:  # Backspace
                        name_input = name_input[:-1] if name_input else ""
                    elif 32 <= key <= 126:  # Printable ASCII characters
                        name_input += chr(key)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        processor.stop()
        processor.join()
        cap.release()
        cv2.destroyAllWindows()
        save_database()
        print("Program exited cleanly.")

if __name__ == "__main__":
    main()