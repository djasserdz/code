#!/usr/bin/env python
import os
import cv2
import time
import torch
import argparse
import numpy as np
from facenet_pytorch import InceptionResnetV1
from data import (
    load_database, save_database, detect_faces, enhance_face_for_recognition, 
    add_person, recognize_face, update_person
)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load FaceNet model for face recognition/embedding
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def generate_embedding(face_img):
    """Generate a face embedding from a face image
    
    Parameters:
        face_img (np.ndarray): Face image (RGB)
        
    Returns:
        np.ndarray: Face embedding
    """
    if face_img.shape[:2] != (160, 160):
        face_img = cv2.resize(face_img, (160, 160))
    
    # Convert to RGB if needed
    if face_img.shape[2] == 3 and face_img.dtype == np.uint8:
        if cv2.COLOR_BGR2RGB != -1:  # If it's a BGR image
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    face_tensor = torch.from_numpy(face_img.transpose((2, 0, 1))).float() / 255.0
    face_tensor = face_tensor.unsqueeze(0).to(device)
    
    # Generate embedding
    with torch.no_grad():
        embedding = facenet(face_tensor).cpu().numpy()[0]
    
    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding


def train_from_photos(name, room="Unknown", min_faces=5, max_faces=20, confidence_threshold=0.8):
    """Train the model from multiple photos taken from the webcam
    
    Parameters:
        name (str): Person's name
        room (str): Room label
        min_faces (int): Minimum number of face images to capture
        max_faces (int): Maximum number of face images to capture
        confidence_threshold (float): Minimum confidence for face detection
    
    Returns:
        bool: True if training successful, False otherwise
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create output directory for captured face images (for reference)
    output_dir = os.path.join("training_faces", name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Start capture process
    captured_embeddings = []
    captured_face_images = []
    
    print(f"Starting face capture for {name}. Press SPACE to capture, ESC to cancel.")
    print(f"Goal: Capture {min_faces}-{max_faces} different face images.")
    
    while len(captured_embeddings) < max_faces:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Make a copy for display
        display_frame = frame.copy()
        
        # Detect faces in frame
        faces = detect_faces(frame, min_confidence=confidence_threshold)
        
        # If faces detected, get the largest one
        largest_face_img = None
        largest_face_rect = None
        largest_area = 0
        
        for face_rect in faces:
            x1, y1, x2, y2 = [int(coord) for coord in face_rect[:4]]
            area = (x2 - x1) * (y2 - y1)
            
            if area > largest_area:
                largest_area = area
                largest_face_rect = (x1, y1, x2, y2)
                
                # Add margin to face
                h_margin = int((y2 - y1) * 0.15)
                w_margin = int((x2 - x1) * 0.15)
                y1_m = max(0, y1 - h_margin)
                y2_m = min(frame.shape[0], y2 + h_margin)
                x1_m = max(0, x1 - w_margin)
                x2_m = min(frame.shape[1], x2 + w_margin)
                
                face_img = frame[y1_m:y2_m, x1_m:x2_m]
                if face_img.size > 0:
                    # Enhance face image
                    largest_face_img = enhance_face_for_recognition(face_img)
        
        # Draw guidance on display frame
        cv2.putText(display_frame, f"Captured: {len(captured_embeddings)}/{max_faces}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(display_frame, "SPACE: Capture | ESC: Finish | Q: Quit", (20, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # If face detected, highlight it
        if largest_face_rect:
            x1, y1, x2, y2 = largest_face_rect
            # Draw rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw quality indicator
            if largest_face_img is not None:
                # Show a quality indicator based on face size
                quality = min(1.0, largest_area / 10000)  # Area of 100x100 gives 1.0
                quality_text = f"Quality: {'*' * int(quality * 5)}"
                cv2.putText(display_frame, quality_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "No face detected", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show preview of captured faces
        if captured_face_images:
            # Create a small thumbnail of the last captured face
            thumbnail_size = 100
            last_face = cv2.resize(captured_face_images[-1], (thumbnail_size, thumbnail_size))
            
            # Place it in the corner
            display_frame[20:20+thumbnail_size, display_frame.shape[1]-20-thumbnail_size:display_frame.shape[1]-20] = last_face
            
            # Label it
            cv2.putText(display_frame, "Last Capture", (display_frame.shape[1]-20-thumbnail_size, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow("Face Training", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key to capture
            if largest_face_img is not None:
                try:
                    # Generate embedding
                    embedding = generate_embedding(largest_face_img)
                    
                    # Check if this face is too similar to already captured ones
                    is_duplicate = False
                    for existing_embedding in captured_embeddings:
                        similarity = np.dot(embedding, existing_embedding)
                        if similarity > 0.95:  # Very high similarity threshold for duplicates
                            is_duplicate = True
                            print("Similar face already captured. Try a different pose or expression.")
                            break
                    
                    if not is_duplicate:
                        # Save the face image and embedding
                        timestamp = int(time.time())
                        cv2.imwrite(os.path.join(output_dir, f"{name}_{timestamp}.jpg"), largest_face_img)
                        captured_embeddings.append(embedding)
                        captured_face_images.append(largest_face_img)
                        print(f"Captured face {len(captured_embeddings)}/{max_faces}")
                    
                except Exception as e:
                    print(f"Error generating embedding: {e}")
            else:
                print("No face detected to capture")
        
        elif key == 27:  # ESC key to finish
            break
        
        elif key == ord('q'):  # Q key to quit
            if len(captured_embeddings) < min_faces:
                print(f"Not enough faces captured. Minimum required: {min_faces}")
                cap.release()
                cv2.destroyAllWindows()
                return False
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Check if we have enough faces
    if len(captured_embeddings) < min_faces:
        print(f"Not enough faces captured. Minimum required: {min_faces}")
        return False
    
    # Average the embeddings (optional, can also add them individually)
    avg_embedding = np.mean(captured_embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    # First check if person already exists in database
    person_id = None
    for embedding in captured_embeddings:
        # Try to recognize this face
        recog_name, info = recognize_face(embedding, threshold=0.7, top_k=1)
        if recog_name is not None and recog_name.lower() == name.lower():
            person_id = info.get("person_id")
            print(f"Found existing entry for {name}, will update with new faces")
            break
    
    # Add embeddings to database
    if person_id:
        # Add additional embeddings to existing person
        for embedding in captured_embeddings:
            add_person(name, embedding, room, person_id)
    else:
        # Add first embedding which will create the person
        first_embedding = captured_embeddings[0]
        person_id = add_person(name, first_embedding, room)
        
        # Add the rest of the embeddings
        for embedding in captured_embeddings[1:]:
            add_person(name, embedding, room, person_id)
    
    print(f"Successfully added {len(captured_embeddings)} face embeddings for {name}")
    # Save the database
    save_database()
    return True


def train_from_image_folder(name, folder_path, room="Unknown"):
    """Train the model from a folder of images
    
    Parameters:
        name (str): Person's name
        folder_path (str): Path to folder containing face images
        room (str): Room label
    
    Returns:
        bool: True if training successful, False otherwise
    """
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return False
    
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return False
    
    successful_images = 0
    captured_embeddings = []
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                print(f"Could not read image: {img_path}")
                continue
            
            # Detect faces in the image
            faces = detect_faces(img, min_confidence=0.5)
            
            # If no faces detected, skip
            if len(faces) == 0:
                print(f"No faces detected in {img_path}")
                continue
            
            # Get the largest face
            largest_area = 0
            largest_face = None
            
            for face_rect in faces:
                x1, y1, x2, y2 = [int(coord) for coord in face_rect[:4]]
                area = (x2 - x1) * (y2 - y1)
                
                if area > largest_area:
                    largest_area = area
                    
                    # Add margin to face
                    h_margin = int((y2 - y1) * 0.15)
                    w_margin = int((x2 - x1) * 0.15)
                    y1_m = max(0, y1 - h_margin)
                    y2_m = min(img.shape[0], y2 + h_margin)
                    x1_m = max(0, x1 - w_margin)
                    x2_m = min(img.shape[1], x2 + w_margin)
                    
                    face_img = img[y1_m:y2_m, x1_m:x2_m]
                    if face_img.size > 0:
                        # Enhance face image
                        largest_face = enhance_face_for_recognition(face_img)
            
            if largest_face is None:
                print(f"Could not extract valid face from {img_path}")
                continue
            
            # Generate embedding
            embedding = generate_embedding(largest_face)
            
            # Check if this face is too similar to already captured ones
            is_duplicate = False
            for existing_embedding in captured_embeddings:
                similarity = np.dot(embedding, existing_embedding)
                if similarity > 0.95:  # Very high similarity threshold for duplicates
                    is_duplicate = True
                    print(f"Image {img_file} too similar to existing training image, skipping")
                    break
            
            if not is_duplicate:
                captured_embeddings.append(embedding)
                successful_images += 1
                print(f"Processed image {successful_images}: {img_file}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    if successful_images == 0:
        print("Could not process any images successfully")
        return False
    
    print(f"Successfully processed {successful_images} images")
    
    # Check if person already exists in database
    person_id = None
    for embedding in captured_embeddings:
        # Try to recognize this face
        recog_name, info = recognize_face(embedding, threshold=0.7, top_k=1)
        if recog_name is not None and recog_name.lower() == name.lower():
            person_id = info.get("person_id")
            print(f"Found existing entry for {name}, will update with new faces")
            break
    
    # Add embeddings to database
    if person_id:
        # Add additional embeddings to existing person
        for embedding in captured_embeddings:
            add_person(name, embedding, room, person_id)
    else:
        # Add first embedding which will create the person
        first_embedding = captured_embeddings[0]
        person_id = add_person(name, first_embedding, room)
        
        # Add the rest of the embeddings
        for embedding in captured_embeddings[1:]:
            add_person(name, embedding, room, person_id)
    
    print(f"Successfully added {len(captured_embeddings)} face embeddings for {name}")
    # Save the database
    save_database()
    return True


def main():
    parser = argparse.ArgumentParser(description='Train face recognition model')
    parser.add_argument('--name', type=str, required=True, help='Name of the person')
    parser.add_argument('--room', type=str, default='Unknown', help='Room label')
    parser.add_argument('--folder', type=str, help='Path to folder with face images')
    parser.add_argument('--min-faces', type=int, default=5, help='Minimum number of faces to capture')
    parser.add_argument('--max-faces', type=int, default=20, help='Maximum number of faces to capture')
    
    args = parser.parse_args()
    
    # Load database
    print("Loading face database...")
    load_database()
    
    if args.folder:
        print(f"Training from image folder: {args.folder}")
        success = train_from_image_folder(args.name, args.folder, args.room)
    else:
        print(f"Training from webcam for {args.name}")
        success = train_from_photos(args.name, args.room, args.min_faces, args.max_faces)
    
    if success:
        print(f"Training successful for {args.name}")
    else:
        print(f"Training failed for {args.name}")


if __name__ == "__main__":
    main() 