import pickle
import numpy as np
import time
import json
import os
import sqlite3




def get_data():
# Path to your .idx file
  IDX_FILE = "face_database/face_database.idx"
  OUTPUT_JSON = "face_database_extracted.json"
  
  if not os.path.exists(IDX_FILE):
      print(f"File '{IDX_FILE}' not found!")
      exit(1)
  
  # Load the .idx file
  with open(IDX_FILE, "rb") as f:
      face_database = pickle.load(f)
  
  # Prepare a dict for JSON export
  export_data = {
      "people": [],
  }
  
  print("People in the database:")
  for person_id, name in face_database["names"].items():
      room = face_database["rooms"].get(person_id, "Unknown")
      timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(face_database["timestamps"][person_id]))
      count = face_database["embedding_count"].get(person_id, 0)
  
      print(f"- person name :{name}")
      print(f"  ID: {person_id}")
      print(f"  Room: {room}")
      print(f"  Added: {timestamp}")
  
      export_data["people"].append({
          "id": person_id,
          "name": name,
          "room": room,
          "timestamp": timestamp,
      })
  
  # Save to JSON
  with open(OUTPUT_JSON, "w") as json_file:
      json.dump(export_data, json_file, indent=4)
  
  print(f"Exported all data to {OUTPUT_JSON}")

get_data()





