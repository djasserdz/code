from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
import sqlite3
import os
import time
import cv2
import numpy as np
import torch

# Local imports
from data import add_person, load_database, save_database, recognize_face, DEFAULT_THRESHOLD
from dat import facenet, device, model as yolo_model

# --- FastAPI App Setup ---
app = FastAPI()

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Initialization ---
def initialize_database():
    conn = sqlite3.connect('face_database/database.sqlite')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            room TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

initialize_database()
load_database()


# --- Routes ---
@app.get("/users")
async def get_users(date: str = Query(default=None)):
    if date is None:
        date = datetime.now().strftime('%d-%m-%Y')
    
    with sqlite3.connect("face_database/database.sqlite") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE timestamp LIKE ?", (f"{date}%",))
        users = cursor.fetchall()
    
    return [
        {"user_id": u[0], "name": u[1], "room": u[2], "timestamp": u[3]}
        for u in users
    ]


@app.post('/use/create')
async def create_user(photo: UploadFile = File(...), room_name: str = Form(...), personal_name: str = Form(...)):
    temp_file_path = f"temp_{int(time.time())}.jpg"

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await photo.read())

        img = cv2.imread(temp_file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = yolo_model(img)
        if len(results) == 0 or len(results[0].boxes) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
        yolo_confs = results[0].boxes.conf.cpu().numpy()
        best_idx = np.argmax(yolo_confs)
        box = yolo_boxes[best_idx]

        x1, y1, x2, y2 = [max(0, int(c)) for c in box]
        if not (x1 < x2 and y1 < y2 and x2 - x1 >= 20 and y2 - y1 >= 20):
            raise HTTPException(status_code=400, detail="Invalid face dimensions")

        h_margin, w_margin = int((y2 - y1) * 0.15), int((x2 - x1) * 0.15)
        y1_m, y2_m = max(0, y1 - h_margin), min(img.shape[0], y2 + h_margin)
        x1_m, x2_m = max(0, x1 - w_margin), min(img.shape[1], x2 + w_margin)

        face = img[y1_m:y2_m, x1_m:x2_m]
        if face.shape[0] <= 0 or face.shape[1] <= 0:
            raise HTTPException(status_code=400, detail="Invalid face crop")

        face_yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
        face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
        face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)

        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face_tensor = torch.from_numpy(face.transpose((2, 0, 1))).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = facenet(face_tensor).detach().cpu().numpy()[0]

        embedding = embedding / np.linalg.norm(embedding)
        threshold = DEFAULT_THRESHOLD + 0.05
        existing_name, info = recognize_face(embedding, threshold)

        if existing_name and info:
            return {
                "success": False,
                "message": f"User already exists as '{existing_name}'",
                "details": {
                    "user_id": info.get("person_id"),
                    "name": existing_name,
                    "room": info.get("room", "Unknown"),
                    "confidence": info.get("confidence", 0.0),
                    "matched": True
                }
            }

        person_id = add_person(personal_name, embedding, room_name)
        save_database()

        face_folder = os.path.join("face_database", "face_images")
        os.makedirs(face_folder, exist_ok=True)
        face_path = os.path.join(face_folder, f"{person_id}.jpg")
        cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        return {
            "success": True,
            "message": f"User {personal_name} successfully created",
            "details": {
                "user_id": person_id,
                "name": personal_name,
                "room": room_name,
                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "matched": False
            }
        }

    except Exception as e:
        error_message = str(e)
        if "No face detected" in error_message:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        raise HTTPException(status_code=500, detail=f"Error processing face: {error_message}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post('/user/info')
async def get_user_info(
    photo: UploadFile = File(...)
):
    """
    Identify a user from a face image and return their information.
    
    Parameters:
        photo: Upload file containing the user's face image
    
    Returns:
        dict: User information if found, or notification that user was not recognized
    """
    temp_file_path = f"temp_info_{int(time.time())}.jpg"
    
    try:
        # 1. Save the uploaded image temporarily
        with open(temp_file_path, "wb") as buffer:
            content = await photo.read()
            buffer.write(content)
        
        # 2. Read and process the image
        img = cv2.imread(temp_file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 3. Detect face in the image using YOLOv8
        from dat import model as yolo_model
        results = yolo_model(img)
        
        # Extract bounding boxes
        if len(results) == 0 or len(results[0].boxes) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Get the face with highest confidence
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
        yolo_confs = results[0].boxes.conf.cpu().numpy()
        
        if len(yolo_boxes) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Find the face with highest confidence
        best_idx = np.argmax(yolo_confs)
        box = yolo_boxes[best_idx]
        
        # 4. Extract and process the face
        x1, y1, x2, y2 = [max(0, int(coord)) for coord in box]
        if not (x1 < x2 and y1 < y2 and x2 - x1 >= 20 and y2 - y1 >= 20):
            raise HTTPException(status_code=400, detail="Invalid face dimensions")
        
        # Add margins around face for better recognition
        h_margin = int((y2 - y1) * 0.15)
        w_margin = int((x2 - x1) * 0.15)
        y1_m = max(0, y1 - h_margin)
        y2_m = min(img.shape[0], y2 + h_margin)
        x1_m = max(0, x1 - w_margin)
        x2_m = min(img.shape[1], x2 + w_margin)
        
        face = img[y1_m:y2_m, x1_m:x2_m]
        if face.shape[0] <= 0 or face.shape[1] <= 0:
            raise HTTPException(status_code=400, detail="Invalid face crop")
        
        # Enhance face image using histogram equalization
        face_yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
        face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
        face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
        
        # Resize to FaceNet input size
        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Convert face to tensor for FaceNet
        face_tensor = torch.from_numpy(face.transpose((2, 0, 1))).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)
        
        # 5. Generate embedding using FaceNet
        with torch.no_grad():
            embedding = facenet(face_tensor).detach().cpu().numpy()[0]
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # 6. Recognize face in database
        name, info = recognize_face(embedding, threshold=DEFAULT_THRESHOLD)
        
        # 7. Get additional details from SQLite database
        if name is not None and info is not None:
            person_id = info.get("person_id")
            
            # Get additional user information from SQLite database
            conn = sqlite3.connect('face_database/database.sqlite')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (person_id,))
            user_record = cursor.fetchone()
            conn.close()
            
            # Get access history (optional)
            access_history = get_user_access_history(person_id)
            
            # Combine information from all sources
            user_info = {
                "user_id": person_id,
                "name": name,
                "room": info.get("room", "Unknown"),
                "recognized": True,
                "sqlite_record": {
                    "timestamp": user_record[3] if user_record else "Not found"
                },
            }
            
            return {
                "success": True,
                "message": f"User recognized as {name}",
                "user_info": user_info
            }
        else:
            # No matching user found
            return {
                "success": False,
                "message": "No matching user found in database",
                "user_info": {
                    "recognized": False
                }
            }
        
    except Exception as e:
        # Handle specific errors
        error_message = str(e)
        if "No face detected" in error_message:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        elif "Invalid face" in error_message:
            raise HTTPException(status_code=400, detail="Invalid face in the image")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing face: {error_message}")
    
    finally:
        # Clean up temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_user_access_history(person_id: str, limit: int = 5) -> list:
    """
    Get the user's recent access history (optional implementation)
    
    Parameters:
        person_id: The user's ID
        limit: Maximum number of records to return
        
    Returns:
        list: List of access records (timestamps)
    """
    try:
        # This is a placeholder - you can implement actual access logging
        # For now, we'll return an empty list or mock data
        
        # Example implementation if you had an access_logs table:
        """
        conn = sqlite3.connect('face_database/database.sqlite')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, location FROM access_logs 
            WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
        ''', (person_id, limit))
        history = [{"timestamp": row[0], "location": row[1]} for row in cursor.fetchall()]
        conn.close()
        return history
        """
        
        # For now, return empty list as placeholder
        return []
    except Exception as e:
        print(f"Error getting access history: {e}")
        return []