import cv2
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from typing import Dict, Any, List, Tuple, Optional
from scipy.spatial.distance import cosine
import time
from collections import defaultdict

# Constants
FACE_DB_PATH = 'face_db.pkl'
EMBEDDING_SIZE = 512
SIMILARITY_THRESHOLD = 0.8  # Adjusted for better balance
RECHECK_INTERVAL = 60  # in seconds
MIN_FACES_FOR_REGISTRATION = 30  # Minimum faces for reliable registration
MAX_REGISTRATION_ATTEMPTS = 100  # Max frames to capture for registration
FRAME_WIDTH = 1280  # Higher resolution for better recognition
FRAME_HEIGHT = 720

# Device and models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    thresholds=[0.6, 0.7, 0.7],  # Adjusted thresholds
    min_face_size=60,
    post_process=False
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load or create face DB with versioning
def load_face_db() -> Dict[str, Any]:
    if os.path.exists(FACE_DB_PATH):
        try:
            with open(FACE_DB_PATH, 'rb') as f:
                db = pickle.load(f)
                # Add version check if needed
                return db
        except (pickle.PickleError, EOFError, Exception) as e:
            print(f"Warning: Face DB is corrupted ({str(e)}). Creating new DB.")
            # Backup corrupted file
            try:
                os.rename(FACE_DB_PATH, FACE_DB_PATH + '.corrupted')
            except:
                pass
    return {'version': 1, 'faces': {}, 'last_updated': datetime.now().isoformat()}

def save_face_db(db: Dict[str, Any]) -> bool:
    try:
        db['last_updated'] = datetime.now().isoformat()
        with open(FACE_DB_PATH, 'wb') as f:
            pickle.dump(db, f)
        return True
    except Exception as e:
        print(f"Error saving face DB: {e}")
        return False

face_db = load_face_db()

# Face utilities
def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector to unit length."""
    return embedding / np.linalg.norm(embedding)

def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return 1.0 - cosine(emb1, emb2)

def draw_face_info(frame: np.ndarray, box: List[int], name: str, similarity: float, color: Tuple[int, int, int]) -> None:
    """Draw face bounding box and information on the frame."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw name and similarity
    label = f"{name} ({similarity:.2f})" if name else "Unknown"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw coordinates
    coord_label = f"({x1},{y1})-({x2},{y2})"
    cv2.putText(frame, coord_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Register new face with improved flow
def register_face(name: str, cap: cv2.VideoCapture) -> bool:
    """Register a new face with interactive guidance."""
    print(f"Starting registration for {name}. Please face the camera directly.")
    
    captured_faces = []
    last_capture_time = time.time()
    registration_window = "Face Registration - Press 's' to start, 'q' to cancel"
    cv2.namedWindow(registration_window, cv2.WINDOW_NORMAL)
    
    try:
        while len(captured_faces) < MIN_FACES_FOR_REGISTRATION:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Show instructions
            instruction = f"Captured {len(captured_faces)}/{MIN_FACES_FOR_REGISTRATION}. Move your head slightly"
            cv2.putText(frame, instruction, (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb_frame)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Auto-capture if face is detected and enough time has passed
                if time.time() - last_capture_time > 0.3 and boxes is not None and len(boxes) == 1:
                    aligned_face = mtcnn(rgb_frame)
                    if aligned_face is not None:
                        captured_faces.append(aligned_face)
                        last_capture_time = time.time()
                        # Visual feedback
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Captured!", (x1, y1 - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(registration_window, frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Registration cancelled")
                cv2.destroyWindow(registration_window)
                return False
            
            if len(captured_faces) >= MAX_REGISTRATION_ATTEMPTS:
                print("Reached maximum registration attempts")
                break
        
        if len(captured_faces) < MIN_FACES_FOR_REGISTRATION:
            print(f"Could only capture {len(captured_faces)} faces. Need at least {MIN_FACES_FOR_REGISTRATION}.")
            return False
        
        # Process captured faces
        faces_stack = torch.cat(captured_faces).to(device)
        embeddings = resnet(faces_stack).detach().cpu().numpy()
        
        # Calculate average embedding with outlier removal
        mean_emb = np.mean(embeddings, axis=0)
        distances = [np.linalg.norm(emb - mean_emb) for emb in embeddings]
        median_distance = np.median(distances)
        
        # Filter out outliers (faces too different from the average)
        filtered_embeddings = [emb for emb, dist in zip(embeddings, distances) 
                              if dist < 2 * median_distance]
        
        if not filtered_embeddings:
            print("No valid faces captured after filtering")
            return False
            
        averaged_embedding = np.mean(filtered_embeddings, axis=0)
        normalized_embedding = normalize_embedding(averaged_embedding)
        
        # Store in database
        if 'faces' not in face_db:
            face_db['faces'] = {}
            
        face_db['faces'][name] = {
            'embedding': normalized_embedding,
            'checkins': [],
            'registration_date': datetime.now().isoformat(),
            'sample_count': len(filtered_embeddings)
        }
        
        if save_face_db(face_db):
            print(f"Successfully registered {name} with {len(filtered_embeddings)} samples")
            return True
        else:
            print("Failed to save face database")
            return False
            
    except Exception as e:
        print(f"Registration error: {str(e)}")
        return False
    finally:
        cv2.destroyWindow(registration_window)

# Recognize faces with improved logic
def recognize_faces(frame: np.ndarray, cap: cv2.VideoCapture) -> np.ndarray:
    """Recognize faces in frame with enhanced logic."""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb_frame)
        
        if boxes is None or len(boxes) == 0:
            return frame
            
        # Get aligned faces
        aligned_faces = mtcnn(rgb_frame)
        if aligned_faces is None:
            return frame
            
        # Get embeddings for all detected faces
        embeddings = resnet(aligned_faces.to(device)).detach().cpu().numpy()
        norm_embeddings = np.array([normalize_embedding(emb) for emb in embeddings])
        
        now = datetime.now()
        recognition_results = []
        
        for i, (box, embedding) in enumerate(zip(boxes, norm_embeddings)):
            best_match = None
            best_similarity = 0.0
            
            # Compare with known faces
            for name, data in face_db.get('faces', {}).items():
                known_emb = data.get('embedding')
                if known_emb is not None:
                    similarity = calculate_similarity(embedding, known_emb)
                    if similarity > best_similarity and similarity > SIMILARITY_THRESHOLD:
                        best_similarity = similarity
                        best_match = name
            
            # Visual feedback
            color = (0, 255, 0) if best_match else (0, 0, 255)
            draw_face_info(frame, box, best_match, best_similarity if best_match else 0, color)
            
            if best_match:
                recognition_results.append((best_match, best_similarity))
                # Update check-in if needed
                checkins = face_db['faces'][best_match]['checkins']
                if not checkins or (now - datetime.fromisoformat(checkins[-1]['time'])).total_seconds() > RECHECK_INTERVAL:
                    checkins.append({
                        'time': now.isoformat(),
                        'type': 'in',
                        'similarity': float(best_similarity),
                        'location': [int(b) for b in box]
                    })
        
        # Save DB if we had any updates
        if recognition_results and save_face_db(face_db):
            pass  # Saved successfully
        
        # Display recognition stats
        stats = f"Faces: {len(boxes)} | Recognized: {len(recognition_results)}"
        cv2.putText(frame, stats, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
        
    except Exception as e:
        print(f"Recognition error: {str(e)}")
        return frame

# Database management functions
def list_registered_faces() -> List[str]:
    """List all registered faces."""
    return list(face_db.get('faces', {}).keys())

def delete_face(name: str) -> bool:
    """Delete a registered face."""
    if name in face_db.get('faces', {}):
        del face_db['faces'][name]
        return save_face_db(face_db)
    return False

def get_face_info(name: str) -> Optional[Dict[str, Any]]:
    """Get information about a registered face."""
    return face_db.get('faces', {}).get(name)

# Main application loop
def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    print("Face Recognition System - Controls:")
    print("  q: Quit")
    print("  r: Register new face")
    print("  d: Delete registered face")
    print("  l: List registered faces")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture error, trying to reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    print("Reconnection failed.")
                    break
                continue
            
            # Process frame
            processed_frame = recognize_faces(frame, cap)
            cv2.imshow('Face Recognition', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                name = input("Enter name to register: ").strip()
                if name and name not in face_db.get('faces', {}):
                    if register_face(name, cap):
                        print(f"Successfully registered {name}")
                    else:
                        print(f"Failed to register {name}")
                else:
                    print("Invalid name or name already exists")
            elif key == ord('d'):
                name = input("Enter name to delete: ").strip()
                if name in face_db.get('faces', {}):
                    if delete_face(name):
                        print(f"Deleted {name}")
                    else:
                        print(f"Failed to delete {name}")
                else:
                    print("Name not found")
            elif key == ord('l'):
                print("Registered faces:", ", ".join(list_registered_faces()) or "None")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("System shut down.")

if __name__ == '__main__':
    main()