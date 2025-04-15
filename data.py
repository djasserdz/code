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
    "index": None          # FAISS index
}

# Load the YOLOv8 model for face detection
yolo_model = YOLO('yolov8n-face.pt')

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
                            "index": None
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

def recognize_face(embedding, threshold=DEFAULT_THRESHOLD, top_k=1):
    """Recognize a face by finding the closest match in the database
    
    Parameters:
        embedding (np.ndarray): Face embedding to search for
        threshold (float): Similarity threshold (0-1)
        top_k (int): Number of top matches to consider
        
    Returns:
        (name, info) tuple or (None, None) if no match
    """
    if not face_database["embeddings"]:
        return None, None
    
    if FAISS_AVAILABLE and face_database["index"] is not None:
        # Convert to proper format for FAISS
        query_embedding = np.array([embedding]).astype('float32')
        
        # Search the index (get top_k matches)
        k = min(top_k, len(face_database["embeddings"]))
        distances, indices = face_database["index"].search(query_embedding, k)
        
        if indices[0][0] == -1 or distances[0][0] < threshold:
            return None, None
            
        # Process results - find most common person_id in top matches
        found_persons = {}
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and distances[0][i] >= threshold:
                person_id = face_database["person_ids"][idx]
                
                if person_id not in found_persons:
                    found_persons[person_id] = {
                        "count": 0,
                        "total_similarity": 0,
                        "best_similarity": 0,
                        "best_index": -1
                    }
                    
                found_persons[person_id]["count"] += 1
                found_persons[person_id]["total_similarity"] += distances[0][i]
                
                if distances[0][i] > found_persons[person_id]["best_similarity"]:
                    found_persons[person_id]["best_similarity"] = distances[0][i]
                    found_persons[person_id]["best_index"] = idx
        
        if not found_persons:
            return None, None
            
        # Find the person with highest confidence
        best_person_id = max(found_persons, key=lambda p: found_persons[p]["best_similarity"])
        best_match = found_persons[best_person_id]
        best_similarity = best_match["best_similarity"]
        
        # Update last seen timestamp
        face_database["timestamps"][best_person_id] = time.time()
            
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
            "matches": best_match["count"]
        }
        
        return name, match_info
    else:
        # Fallback to numpy-based similarity search
        embeddings = np.array(face_database["embeddings"])
        similarities = np.dot(embeddings, embedding)
        
        # Get top k matches
        k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        if similarities[top_indices[0]] < threshold:
            return None, None
            
        # Count occurrences of each person_id in top matches
        found_persons = {}
        
        for idx in top_indices:
            if similarities[idx] >= threshold:
                person_id = face_database["person_ids"][idx]
                
                if person_id not in found_persons:
                    found_persons[person_id] = {
                        "count": 0,
                        "total_similarity": 0,
                        "best_similarity": 0,
                        "best_index": -1
                    }
                    
                found_persons[person_id]["count"] += 1
                found_persons[person_id]["total_similarity"] += similarities[idx]
                
                if similarities[idx] > found_persons[person_id]["best_similarity"]:
                    found_persons[person_id]["best_similarity"] = similarities[idx]
                    found_persons[person_id]["best_index"] = idx
        
        if not found_persons:
            return None, None
            
        # Find the person with highest average similarity
        best_person_id = max(found_persons, key=lambda p: found_persons[p]["best_similarity"])
        best_match = found_persons[best_person_id]
        best_similarity = best_match["best_similarity"]
        
        # Update last seen timestamp
        face_database["timestamps"][best_person_id] = time.time()
            
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
            "matches": best_match["count"]
        }
        
        return name, match_info

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


def detect_faces(image):
    """Detect faces in an image using YOLOv8 model
    
    Parameters:
        image (np.ndarray): Image in which to detect faces
        
    Returns:
        list: List of bounding boxes for detected faces
    """
    results = yolo_model(image)
    # Extract bounding boxes from results
    bboxes = results.xyxy[0].numpy()  # Assuming results are in xyxy format
    return bboxes
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