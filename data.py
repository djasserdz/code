try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: faiss not available. Using slower numpy-based similarity search.")
    FAISS_AVAILABLE = False
import numpy as np
import os
import pickle
import json
import time
from datetime import datetime
import shutil
from ultralytics import YOLO
from datetime import datetime
import sqlite3
import cv2
import torch

# Constants
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_THRESHOLD = 0.6
DATABASE_FOLDER = "face_database"
DATABASE_FILE = os.path.join(DATABASE_FOLDER, "face_database.idx")
INDEX_FILE = os.path.join(DATABASE_FOLDER, "faiss_index.bin")
BACKUP_FOLDER = os.path.join(DATABASE_FOLDER, "backups")

# Global variables for database
face_database = {
    "embeddings": [],      # List of all embeddings
    "person_ids": [],      # ID of the person for each embedding
    "names": {},           # Dictionary mapping person_id -> name
    "rooms": {},           # Dictionary mapping person_id -> room
    "timestamps": {},      # Dictionary mapping person_id -> last seen timestamp
    "embedding_count": {}, # Dictionary mapping person_id -> number of embeddings
    "index": None,         # FAISS index
    "trackers": {}         # Dictionary to store face trackers
}

# Load the YOLOv8 model for face detection - use larger model for better distance detection
yolo_model = YOLO('yolov8m-face.pt')  # Using medium model for better accuracy and distance detection

# Track IDs for continuous face tracking
next_track_id = 0
# Dictionary to store tracked faces across frames
tracked_faces = {}
# Time window for tracking (seconds)
TRACKING_TIMEOUT = 2.0

# Ensure database folders exist
def ensure_db_folders():
    """Create necessary folders for the database if they don't exist"""
    os.makedirs(DATABASE_FOLDER, exist_ok=True)
    os.makedirs(BACKUP_FOLDER, exist_ok=True)

def initialize_index(dimension=DEFAULT_EMBEDDING_DIM):
    """Initialize a FAISS index for fast similarity search
    
    Parameters:
        dimension (int): Dimension of the face embeddings
        
    Returns:
        FAISS index object or None if FAISS is not available
    """
    if not FAISS_AVAILABLE:
        return None
        
    # Create IndexFlatIP for cosine similarity with normalized vectors
    # (Inner product is equivalent to cosine similarity when vectors are normalized)
    index = faiss.IndexFlatIP(dimension)
    
    # For larger databases, we could use more advanced indices
    # Uncomment the following if you have many faces (>1000)
    """
    # Build IndexIVFFlat for faster search with slight accuracy tradeoff
    nlist = 100  # number of clusters
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    # Need to train this type of index before use
    if face_database["embeddings"] and len(face_database["embeddings"]) > nlist:
        index.train(np.array(face_database["embeddings"]).astype('float32'))
    """
    
    return index

def save_index(index, filename=INDEX_FILE):
    """Save FAISS index to a file
    
    Parameters:
        index: FAISS index object
        filename (str): Path to save the index
    """
    if FAISS_AVAILABLE and index is not None:
        faiss.write_index(index, filename)

def load_index(filename=INDEX_FILE, dimension=DEFAULT_EMBEDDING_DIM):
    """Load FAISS index from a file
    
    Parameters:
        filename (str): Path to the index file
        dimension (int): Dimension of the embeddings
        
    Returns:
        FAISS index object or newly initialized index if file doesn't exist
    """
    if not FAISS_AVAILABLE:
        return None
        
    if os.path.exists(filename):
        try:
            return faiss.read_index(filename)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return initialize_index(dimension)
    else:
        return initialize_index(dimension)

def backup_database():
    """Create a backup of the database files"""
    if not os.path.exists(DATABASE_FILE):
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_FOLDER, f"face_database_{timestamp}.pkl")
    
    try:
        shutil.copy2(DATABASE_FILE, backup_file)
        if os.path.exists(INDEX_FILE):
            index_backup = os.path.join(BACKUP_FOLDER, f"faiss_index_{timestamp}.bin")
            shutil.copy2(INDEX_FILE, index_backup)
        print(f"Database backup created: {backup_file}")
    except Exception as e:
        print(f"Error creating backup: {e}")

def load_database():
    """Load the face database from disk"""
    global face_database
    
    ensure_db_folders()
    
    # Try to load the database
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'rb') as f:
                saved_data = pickle.load(f)
                
                # Load database structure (handle both old and new formats)
                if "embeddings" in saved_data and "names" in saved_data:
                    # Handle old format data
                    if isinstance(saved_data["names"], list):
                        print("Converting from old database format...")
                        
                        # Convert old format to new format
                        embeddings = saved_data.get("embeddings", [])
                        old_names = saved_data.get("names", [])
                        old_rooms = saved_data.get("rooms", [])
                        
                        # Reset database
                        face_database = {
                            "embeddings": [],
                            "person_ids": [],
                            "names": {},
                            "rooms": {},
                            "timestamps": {},
                            "embedding_count": {},
                            "index": None,
                            "trackers": {}
                        }
                        
                        # Convert old entries to new format
                        for i, (emb, name, room) in enumerate(zip(embeddings, old_names, old_rooms)):
                            person_id = f"person_{i}"
                            face_database["embeddings"].append(emb)
                            face_database["person_ids"].append(person_id)
                            face_database["names"][person_id] = name
                            face_database["rooms"][person_id] = room
                            face_database["timestamps"][person_id] = time.time()
                            face_database["embedding_count"][person_id] = 1
                    else:
                        # Load new format directly
                        face_database["embeddings"] = saved_data.get("embeddings", [])
                        face_database["person_ids"] = saved_data.get("person_ids", [])
                        face_database["names"] = saved_data.get("names", {})
                        face_database["rooms"] = saved_data.get("rooms", {})
                        face_database["timestamps"] = saved_data.get("timestamps", {})
                        face_database["embedding_count"] = saved_data.get("embedding_count", {})
                
                # Load or create the FAISS index
                if face_database["embeddings"]:
                    dimension = len(face_database["embeddings"][0])
                    
                    if FAISS_AVAILABLE:
                        # Try to load saved index first, fallback to rebuilding
                        face_database["index"] = load_index(dimension=dimension)
                        
                        # Check if index is empty or doesn't match embeddings
                        if face_database["index"].ntotal != len(face_database["embeddings"]):
                            print("Rebuilding FAISS index to match embeddings...")
                            face_database["index"] = initialize_index(dimension)
                            face_database["index"].add(np.array(face_database["embeddings"]).astype('float32'))
                            
                        unique_people = len(set(face_database["person_ids"]))
                        print(f"Loaded {unique_people} people with {len(face_database['embeddings'])} faces in database")
                    else:
                        face_database["index"] = None
                        unique_people = len(set(face_database["person_ids"]))
                        print(f"Loaded {unique_people} people with {len(face_database['embeddings'])} faces (without FAISS index)")
                else:
                    face_database["index"] = None
                    print("Database empty, starting fresh")
        except Exception as e:
            print(f"Error loading database: {e}")
            face_database["index"] = None
    else:
        print("No existing database found, starting fresh")
        face_database["index"] = None

def save_database():
    """Save the face database to disk"""
    ensure_db_folders()
    
    # Create a backup before saving
    if os.path.exists(DATABASE_FILE):
        backup_database()
    
    # Save database
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump({
            "embeddings": face_database["embeddings"],
            "person_ids": face_database["person_ids"],
            "names": face_database["names"],
            "rooms": face_database["rooms"],
            "timestamps": face_database["timestamps"],
            "embedding_count": face_database["embedding_count"]
        }, f)
    
    # Save FAISS index separately
    if FAISS_AVAILABLE and face_database["index"] is not None:
        save_index(face_database["index"])
    
    unique_people = len(set(face_database["person_ids"]))
    print(f"Database saved with {unique_people} people and {len(face_database['embeddings'])} face embeddings")

def get_new_person_id():
    """Generate a new unique person ID"""
    timestamp = int(time.time())
    return f"person_{timestamp}_{np.random.randint(1000, 9999)}"

def detect_faces(image, min_confidence=0.35, enhance_small_faces=True):
    """Detect faces in an image using YOLOv8 model with improved distance detection
    
    Parameters:
        image (np.ndarray): Image in which to detect faces
        min_confidence (float): Minimum confidence for face detection (lower values help detect distant faces)
        enhance_small_faces (bool): Whether to apply enhancement for small/distant faces
        
    Returns:
        list: List of bounding boxes for detected faces
    """
    # Apply preprocessing to enhance distant faces if enabled
    if enhance_small_faces:
        # Create a slightly sharpened version of the image to detect distant faces better
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 5.0
        enhanced_img = cv2.filter2D(image, -1, kernel)
        
        # Run detection on both original and enhanced images
        results_orig = yolo_model(image, conf=min_confidence)
        results_enhanced = yolo_model(enhanced_img, conf=min_confidence)
        
        # Get boxes from both results
        boxes_orig = results_orig.xyxy[0].numpy() if len(results_orig) > 0 else np.array([])
        boxes_enhanced = results_enhanced.xyxy[0].numpy() if len(results_enhanced) > 0 else np.array([])
        
        # Merge the results (removing duplicates)
        if len(boxes_orig) > 0 and len(boxes_enhanced) > 0:
            all_boxes = np.vstack((boxes_orig, boxes_enhanced))
            # Remove duplicates based on IoU
            final_boxes = []
            for box in all_boxes:
                is_duplicate = False
                for final_box in final_boxes:
                    iou = calculate_iou(box[:4], final_box[:4])
                    if iou > 0.5:  # If IoU > 0.5, consider as duplicate
                        # Keep the one with higher confidence
                        if box[4] > final_box[4]:
                            final_box[:] = box[:]
                        is_duplicate = True
                        break
                if not is_duplicate:
                    final_boxes.append(box)
            return np.array(final_boxes)
        elif len(boxes_enhanced) > 0:
            return boxes_enhanced
        else:
            return boxes_orig
    else:
        # Use original detection method
        results = yolo_model(image, conf=min_confidence)
        if len(results) > 0:
            return results.xyxy[0].numpy()
        return np.array([])

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two boxes
    
    Parameters:
        box1: First bounding box in format [x1, y1, x2, y2]
        box2: Second bounding box in format [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # If boxes don't intersect, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def recognize_face(embedding, threshold=DEFAULT_THRESHOLD, top_k=3, use_adaptive_threshold=True):
    """Recognize a face by finding the closest match in the database
    
    Parameters:
        embedding (np.ndarray): Face embedding to search for
        threshold (float): Similarity threshold (0-1)
        top_k (int): Number of top matches to consider
        use_adaptive_threshold (bool): Whether to adaptively adjust threshold based on face count
        
    Returns:
        (name, info) tuple or (None, None) if no match
    """
    if not face_database["embeddings"]:
        return None, None
    
    # Adjust threshold based on database size if adaptive thresholding is enabled
    adjusted_threshold = threshold
    if use_adaptive_threshold:
        person_count = len(set(face_database["person_ids"]))
        if person_count > 20:
            # For larger databases, increase the strictness to avoid false positives
            adjusted_threshold = min(threshold + 0.05, 0.75)
        elif person_count < 5:
            # For smaller databases, decrease the strictness to improve recognition
            adjusted_threshold = max(threshold - 0.05, 0.5)
    
    if FAISS_AVAILABLE and face_database["index"] is not None:
        # Convert to proper format for FAISS
        query_embedding = np.array([embedding]).astype('float32')
        
        # Search the index (get top_k matches)
        k = min(top_k * 2, len(face_database["embeddings"]))  # Get more candidates for better selection
        distances, indices = face_database["index"].search(query_embedding, k)
        
        if indices[0][0] == -1 or distances[0][0] < adjusted_threshold:
            return None, None
            
        # Process results - find most common person_id in top matches
        found_persons = {}
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and distances[0][i] >= adjusted_threshold:
                person_id = face_database["person_ids"][idx]
                
                if person_id not in found_persons:
                    found_persons[person_id] = {
                        "count": 0,
                        "total_similarity": 0,
                        "best_similarity": 0,
                        "best_index": -1,
                        "recent_seen": face_database["timestamps"].get(person_id, 0)
                    }
                    
                found_persons[person_id]["count"] += 1
                found_persons[person_id]["total_similarity"] += distances[0][i]
                
                if distances[0][i] > found_persons[person_id]["best_similarity"]:
                    found_persons[person_id]["best_similarity"] = distances[0][i]
                    found_persons[person_id]["best_index"] = idx
        
        if not found_persons:
            return None, None
            
        # Find the person with highest confidence, considering recency
        # This adds a small bonus for recently seen people to improve tracking
        current_time = time.time()
        best_person_id = None
        best_score = -1
        
        for person_id, data in found_persons.items():
            # Calculate a weighted score considering similarity and recency
            recency_bonus = max(0, min(0.05, (1.0 - min(TRACKING_TIMEOUT, current_time - data["recent_seen"]) / TRACKING_TIMEOUT) * 0.05))
            score = data["best_similarity"] + recency_bonus
            
            if score > best_score:
                best_score = score
                best_person_id = person_id
                
        best_match = found_persons[best_person_id]
        best_similarity = best_match["best_similarity"]
        
        # Update last seen timestamp
        face_database["timestamps"][best_person_id] = current_time
            
        name = face_database["names"][best_person_id]
        room = face_database["rooms"][best_person_id]
        
        # Add extra info
        avg_similarity = best_match["total_similarity"] / best_match["count"]
        match_info = {
            "person_id": best_person_id,
            "room": room,
            "confidence": float(best_similarity),
            "avg_confidence": float(avg_similarity),
            "embedding_count": face_database["embedding_count"][best_person_id],
            "matches": best_match["count"],
            "adjusted_threshold": adjusted_threshold
        }
        
        return name, match_info
    else:
        # Fallback to numpy-based similarity search with the same improvements
        embeddings = np.array(face_database["embeddings"])
        similarities = np.dot(embeddings, embedding)
        
        # Get top k matches
        k = min(top_k * 2, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        if similarities[top_indices[0]] < adjusted_threshold:
            return None, None
            
        # Count occurrences of each person_id in top matches
        found_persons = {}
        current_time = time.time()
        
        for idx in top_indices:
            if similarities[idx] >= adjusted_threshold:
                person_id = face_database["person_ids"][idx]
                
                if person_id not in found_persons:
                    found_persons[person_id] = {
                        "count": 0,
                        "total_similarity": 0,
                        "best_similarity": 0,
                        "best_index": -1,
                        "recent_seen": face_database["timestamps"].get(person_id, 0)
                    }
                    
                found_persons[person_id]["count"] += 1
                found_persons[person_id]["total_similarity"] += similarities[idx]
                
                if similarities[idx] > found_persons[person_id]["best_similarity"]:
                    found_persons[person_id]["best_similarity"] = similarities[idx]
                    found_persons[person_id]["best_index"] = idx
        
        if not found_persons:
            return None, None
            
        # Find the person with highest confidence, considering recency
        best_person_id = None
        best_score = -1
        
        for person_id, data in found_persons.items():
            # Calculate a weighted score considering similarity and recency
            recency_bonus = max(0, min(0.05, (1.0 - min(TRACKING_TIMEOUT, current_time - data["recent_seen"]) / TRACKING_TIMEOUT) * 0.05))
            score = data["best_similarity"] + recency_bonus
            
            if score > best_score:
                best_score = score
                best_person_id = person_id
                
        best_match = found_persons[best_person_id]
        best_similarity = best_match["best_similarity"]
        
        # Update last seen timestamp
        face_database["timestamps"][best_person_id] = current_time
            
        name = face_database["names"][best_person_id]
        room = face_database["rooms"][best_person_id]
        
        # Add extra info
        avg_similarity = best_match["total_similarity"] / best_match["count"]
        match_info = {
            "person_id": best_person_id,
            "room": room,
            "confidence": float(best_similarity),
            "avg_confidence": float(avg_similarity),
            "embedding_count": face_database["embedding_count"][best_person_id],
            "matches": best_match["count"],
            "adjusted_threshold": adjusted_threshold
        }
        
        return name, match_info

def track_faces(frame, detected_boxes, recognized_faces=None):
    """Track faces across frames for consistent recognition
    
    Parameters:
        frame (np.ndarray): Current video frame
        detected_boxes (list): List of detected face boxes
        recognized_faces (list, optional): List of recognized face data
        
    Returns:
        dict: Updated tracking information
    """
    global next_track_id, tracked_faces
    current_time = time.time()
    
    # Clean up old tracks
    to_remove = []
    for track_id, track_info in tracked_faces.items():
        if current_time - track_info["last_seen"] > TRACKING_TIMEOUT:
            to_remove.append(track_id)
    for track_id in to_remove:
        del tracked_faces[track_id]
        
    # Match current detections with existing tracks
    if detected_boxes is not None and len(detected_boxes) > 0:
        # For each detection, try to match with existing tracks
        unmatched_detections = []
        matched_track_ids = []
        
        for i, box in enumerate(detected_boxes):
            box_coords = box[:4]
            matched = False
            best_iou = 0.3  # Minimum IoU to consider a match
            matched_id = None
            
            # Try to match with existing tracks
            for track_id, track_info in tracked_faces.items():
                iou = calculate_iou(box_coords, track_info["box"])
                if iou > best_iou:
                    best_iou = iou
                    matched_id = track_id
                    matched = True
            
            if matched:
                # Update existing track
                matched_track_ids.append(matched_id)
                tracked_faces[matched_id]["box"] = box_coords
                tracked_faces[matched_id]["confidence"] = box[4] if len(box) > 4 else 1.0
                tracked_faces[matched_id]["last_seen"] = current_time
                
                # If we have recognition info, update it
                if recognized_faces and i < len(recognized_faces):
                    name, info = recognized_faces[i]
                    if name and name != "Unknown":
                        tracked_faces[matched_id]["name"] = name
                        tracked_faces[matched_id]["info"] = info
            else:
                unmatched_detections.append((i, box))
                
        # Create new tracks for unmatched detections
        for idx, box in unmatched_detections:
            track_id = next_track_id
            next_track_id += 1
            
            # Initial track info
            track_info = {
                "box": box[:4],
                "confidence": box[4] if len(box) > 4 else 1.0,
                "first_seen": current_time,
                "last_seen": current_time,
                "frames_tracked": 1,
                "name": "Unknown",
                "info": None
            }
            
            # If we have recognition info, use it
            if recognized_faces and idx < len(recognized_faces):
                name, info = recognized_faces[idx]
                if name and name != "Unknown":
                    track_info["name"] = name
                    track_info["info"] = info
                    
            tracked_faces[track_id] = track_info
    
    return tracked_faces

def enhance_face_for_recognition(face_img):
    """Enhance a face image for better recognition, especially for distant faces
    
    Parameters:
        face_img (np.ndarray): The face image to enhance
        
    Returns:
        np.ndarray: The enhanced face image
    """
    # Skip empty or invalid images
    if face_img is None or face_img.size == 0:
        return face_img
    
    try:
        # Convert to YUV color space for luminance enhancement
        face_yuv = cv2.cvtColor(face_img, cv2.COLOR_BGR2YUV)
        
        # Apply histogram equalization to the Y channel
        face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_yuv[:, :, 0] = clahe.apply(face_yuv[:, :, 0])
        
        # Convert back to BGR
        enhanced_face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
        
        # Apply slight sharpening for better feature definition
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 5.0
        enhanced_face = cv2.filter2D(enhanced_face, -1, kernel)
        
        return enhanced_face
    except Exception as e:
        print(f"Error enhancing face: {e}")
        return face_img

def add_person(name, embedding, room="Unknown", person_id=None):
    """Add a new person to the database
    
    Parameters:
        name (str): Person's name
        embedding (np.ndarray): Face embedding
        room (str): Room label
        person_id (str, optional): Existing person ID to add another face to
                                   If None, creates a new person
    
    Returns:
        str: person_id of the added or updated person
    """
    global face_database
    
    # Generate new ID if needed
    if person_id is None or person_id not in face_database["names"]:
        person_id = get_new_person_id()
        face_database["names"][person_id] = name
        face_database["rooms"][person_id] = room
        face_database["timestamps"][person_id] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        face_database["embedding_count"][person_id] = 0

        time=face_database["timestamps"][person_id]
        
        conn = sqlite3.connect('face_database/database.sqlite')
        cursor = conn.cursor()
        existing_person = cursor.execute("SELECT * FROM users WHERE  user_id  = ?", (person_id,)).fetchone()
        if existing_person:
            cursor.execute("UPDATE users SET  room = ?, timestamp = ? WHERE user_id = ?", (room,time , person_id))
        else:
         cursor.execute("INSERT INTO users (user_id, name, room, timestamp) VALUES (?, ?, ?, ?)", (person_id, name, room, time))
        conn.commit()
        conn.close()

        is_new_person = True
    else:
        is_new_person = False
    

        
    # Initialize index if it doesn't exist
    if FAISS_AVAILABLE and face_database["index"] is None and embedding is not None:
        dimension = len(embedding)
        face_database["index"] = initialize_index(dimension)
    
    # Add to the lists
    face_database["embeddings"].append(embedding)
    face_database["person_ids"].append(person_id)
    face_database["embedding_count"][person_id] = face_database["embedding_count"].get(person_id, 0) + 1
    
    # Add to the index if available
    if FAISS_AVAILABLE and face_database["index"] is not None and embedding is not None:
        face_database["index"].add(np.array([embedding]).astype('float32'))
    
    if is_new_person:
        print(f"Added new person {name} to database (Total people: {len(face_database['names'])})")
    else:
        print(f"Added new face for {name} (Total faces: {face_database['embedding_count'][person_id]})")
        
    return person_id

def update_person(person_id, name=None, room=None):
    """Update information for an existing person
    
    Parameters:
        person_id (str): Person ID to update
        name (str, optional): New name (if None, keep existing)
        room (str, optional): New room (if None, keep existing)
    
    Returns:
        bool: Success
    """
    if person_id not in face_database["names"]:
        return False
        
    if name is not None:
        face_database["names"][person_id] = name
    
    if room is not None:
        face_database["rooms"][person_id] = room
        
    return True

def capture_frames_for_embedding(capture_device, face_recognizer, num_images=50, delay=0.2):
    """Capture multiple frames to generate a robust face embedding
    
    Parameters:
        capture_device: Video capture device (e.g., cv2.VideoCapture object)
        face_recognizer: Face recognition model to generate embeddings
        num_images (int): Number of images to capture
        delay (float): Delay between captures in seconds
        
    Returns:
        tuple: (success, embedding, frames)
            - success (bool): True if enough valid faces were detected
            - embedding (np.ndarray): Average embedding across all valid frames
            - frames (list): List of captured frames that contained valid faces
    """
    embeddings = []
    valid_frames = []
    min_required = max(1, num_images // 2)  # At least half of the frames should have valid faces
    
    print(f"Capturing {num_images} frames for robust embedding...")
    
    # Create a dedicated window for capture status
    status_window_name = "Capture Status"
    status_height, status_width = 400, 600
    
    # Initialize counters
    valid_count = 0
    invalid_count = 0
    
    # Create windows
    cv2.namedWindow("Capture")
    cv2.namedWindow(status_window_name)
    cv2.moveWindow(status_window_name, 650, 100)  # Position it to the right of the main window
    
    # Initial status display
    status_window = np.zeros((status_height, status_width, 3), dtype=np.uint8)
    cv2.putText(status_window, "Starting Capture...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(status_window_name, status_window)
    cv2.waitKey(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        for i in range(num_images):
            # Capture frame
            ret, frame = capture_device.read()
            if not ret:
                print(f"Error capturing frame {i+1}/{num_images}")
                continue
                
            # Display countdown/progress on the main capture window
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing {i+1}/{num_images}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture", display_frame)
            
            # Create new status window for this frame
            status_window = np.zeros((status_height, status_width, 3), dtype=np.uint8)
            
            # Draw title and progress
            cv2.putText(status_window, "Capture Status", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(status_window, f"Progress: {i+1}/{num_images}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw progress bar
            progress_percent = (i + 1) / num_images
            bar_width = int(progress_percent * (status_width - 40))
            cv2.rectangle(status_window, (20, 100), (20 + bar_width, 130), (0, 255, 0), -1)
            cv2.rectangle(status_window, (20, 100), (status_width - 20, 130), (255, 255, 255), 1)
            
            # Detect faces in the frame
            try:
                bboxes = detect_faces(frame)
                
                # If face detected, get the largest one (assume it's the main subject)
                if len(bboxes) > 0:
                    # Find the face with the largest area
                    areas = [(box[2]-box[0])*(box[3]-box[1]) for box in bboxes]
                    largest_idx = np.argmax(areas)
                    box = bboxes[largest_idx]
                    
                    # Extract face and generate embedding
                    x1, y1, x2, y2, conf, cls = box
                    
                    # Safety check for dimensions
                    x1, y1, x2, y2 = [max(0, int(coord)) for coord in [x1, y1, x2, y2]]
                    # Ensure the box has dimensions and is within frame
                    if x1 < x2 and y1 < y2 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                        face_img = frame[y1:y2, x1:x2]
                        
                        if face_img.size == 0:
                            raise ValueError("Empty face image")
                        
                        # Draw rectangle on the display frame
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.imshow("Capture", display_frame)
                        
                        # Prepare face image for the recognizer
                        # Resize to 160x160 (standard for many face recognition models)
                        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_img_resized = cv2.resize(face_img_rgb, (160, 160))
                        
                        # Generate embedding
                        try:
                            # Convert to tensor and normalize
                            face_tensor = torch.from_numpy(face_img_resized.transpose((2, 0, 1))).float().to(device)
                            face_tensor = face_tensor / 255.0
                            face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
                            
                            # Generate embedding using the model
                            with torch.no_grad():
                                face_embedding = face_recognizer(face_tensor).cpu().numpy()[0]
                            
                            # Normalize the embedding to unit length
                            face_embedding = face_embedding / np.linalg.norm(face_embedding)
                            
                            embeddings.append(face_embedding)
                            valid_frames.append(frame)
                            valid_count += 1
                            
                            # Draw a preview of the detected face in the status window
                            face_preview = cv2.resize(face_img, (150, 150))
                            h, w = face_preview.shape[:2]
                            y_offset = 140
                            status_window[y_offset:y_offset+h, 20:20+w] = face_preview
                            
                            # Display success message in status window
                            cv2.putText(status_window, "Face detected!", (180, 200), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(status_window, f"Confidence: {conf:.2f}", (180, 230), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                                      
                        except Exception as e:
                            invalid_count += 1
                            print(f"Frame {i+1}: Error generating embedding: {str(e)}")
                            cv2.putText(status_window, "Error processing face", (20, 180), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(status_window, str(e)[:30], (20, 210), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        invalid_count += 1
                        print(f"Frame {i+1}: Invalid face dimensions")
                        cv2.putText(status_window, "Invalid face dimensions", (20, 180), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    invalid_count += 1
                    print(f"Frame {i+1}: No face detected")
                    cv2.putText(status_window, "No face detected", (20, 180), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                invalid_count += 1
                print(f"Frame {i+1}: Error in face detection: {str(e)}")
                cv2.putText(status_window, "Error in face detection", (20, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display counts
            cv2.putText(status_window, f"Valid captures: {valid_count}", (20, 310), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(status_window, f"Invalid captures: {invalid_count}", (300, 310), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show instruction
            cv2.putText(status_window, "Press ESC to cancel", (20, 350), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Update and show status window
            cv2.imshow(status_window_name, status_window)
            
            # Wait for keys
            key = cv2.waitKey(1)
            if key == 27:  # ESC to cancel
                print("Capture canceled by user")
                break
                
            # Wait between captures
            time.sleep(delay)
        
        # Final status display
        status_window = np.zeros((status_height, status_width, 3), dtype=np.uint8)
        cv2.putText(status_window, "Capture Completed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(status_window, f"Total attempts: {num_images}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(status_window, f"Valid captures: {valid_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(status_window, f"Required minimum: {min_required}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if valid_count >= min_required:
            cv2.putText(status_window, "SUCCESS: Enough valid faces captured", (20, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            success = True
        else:
            cv2.putText(status_window, f"FAILED: Not enough valid faces", (20, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(status_window, f"({valid_count}/{min_required} required)", (20, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            success = False
        
        cv2.putText(status_window, "Press any key to continue...", (20, 350), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.imshow(status_window_name, status_window)
        cv2.waitKey(0)  # Wait for key press before continuing

    finally:
        # Always clean up windows
        cv2.destroyAllWindows()
    
    # Check if we have enough valid frames
    if valid_count < min_required:
        print(f"Not enough valid faces detected ({valid_count}/{num_images})")
        return False, None, valid_frames
    
    if len(embeddings) == 0:
        print("No valid embeddings generated")
        return False, None, valid_frames
        
    # Calculate average embedding
    avg_embedding = np.mean(embeddings, axis=0)
    # Normalize again to ensure unit length
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    print(f"Successfully captured {valid_count}/{num_images} valid frames")
    return True, avg_embedding, valid_frames