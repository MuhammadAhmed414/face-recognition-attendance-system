import face_recognition # <-- This must be the first line
import os
import pickle

# --- Configuration ---
# The path must be relative to the script location (src/)
DATASET_DIR = "../dataset" 
# This file will be created in the root project folder
ENCODINGS_FILE = "../encodings.pkl" 

known_encodings = []
known_names = []

print("Starting encoding process...")

# Loop through every person's folder in the dataset
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    
    # Check if it's a directory (i.e., a student folder)
    if os.path.isdir(person_path):
        print(f"Processing images for: {person_name}")

        # Loop through every image in the person's folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            
            # Load the image
            image = face_recognition.load_image_file(image_path)
            
            # Get the 128-dimension face encoding (features)
            encodings = face_recognition.face_encodings(image) 

            if encodings:
                # Append the first found encoding and the person's name
                known_encodings.append(encodings[0])
                known_names.append(person_name)
                print(f"   -> Encoded {image_name}")
            else:
                print(f"   -> ❌ Warning: No face found in {image_name}. Skipping.")

# Save the final data (encodings and names) into a pickle file
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("\n✅ Face encodings saved successfully!")
print(f"File created: {ENCODINGS_FILE}")