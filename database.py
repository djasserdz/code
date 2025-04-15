import sqlite3

conn = sqlite3.connect("face_database/database.sqlite")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    room TEXT NOT NULL,
    timestamp  timestamp  NULL default CURRENT_TIMESTAMP
)
""")

print("Database created successfully")

conn.close()
