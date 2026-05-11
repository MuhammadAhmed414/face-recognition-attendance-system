import cv2
import face_recognition
import numpy as np
import pandas as pd
import datetime
import pickle
import os

# --- Configuration ---
ENCODINGS_FILE = "../encodings.pkl"
ATTENDANCE_FILE = "../attendance/Attendance.csv"
DATASET_DIR = "../dataset" 

# --- Global Variables for Recognition (will be updated during enrollment) ---
known_face_encodings = []
known_face_names = []
face_locations = []
face_names = [] # Global lists to hold the last recognized faces (used for display in skipped frames)

# Load the saved encodings and names
try:
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"✅ Loaded {len(known_face_names)} known faces.")
except FileNotFoundError:
    print("❌ ERROR: encodings.pkl not found. Please run 'encode_faces.py' first.")
    exit()

# List to hold the names of students who have already been marked present
marked_attendance_today = []

# --- Helper Function: Log Attendance ---
def mark_attendance(name):
    """Logs the attendance to the CSV file if the student hasn't been marked today."""
    
    if name in marked_attendance_today:
        return

    now = datetime.datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    attendance_data = {
        'Name': [name],
        'Date': [date_str],
        'Time': [time_str]
    }
    df_new = pd.DataFrame(attendance_data)
    
    # Ensure the attendance directory exists
    os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
    
    write_header = not os.path.exists(ATTENDANCE_FILE)
    
    df_new.to_csv(ATTENDANCE_FILE, mode='a', header=write_header, index=False)
    
    marked_attendance_today.append(name)
    print(f"✅ ATTENDANCE LOGGED: {name} at {time_str}")


# --- Helper Function: Re-encode Faces for Enrollment ---
def re_encode_faces():
    """
    Reruns the encoding process for all images in the dataset and updates encodings.pkl.
    """
    global known_face_encodings, known_face_names
    
    new_encodings = []
    new_names = []
    
    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image) 
                        
                        if encodings:
                            new_encodings.append(encodings[0])
                            new_names.append(person_name)
                    except Exception as e:
                        print(f"Error processing image {image_name}: {e}")

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((new_encodings, new_names), f)

    known_face_encodings = new_encodings
    known_face_names = new_names
    print(f"\n✅ ENROLLMENT SUCCESS: {len(new_names)} faces loaded (including the new one).")


# --- Initialize Video Capture ---
video_capture = cv2.VideoCapture(0)

print("\nStarting video stream. Press 'q' to quit.")

# --- Frame Skipping Configuration ---
frame_skip_counter = 0 
frame_skip_rate = 5  # Check for a face every 5th frame
# ------------------------------------
print("If an Unknown face appears, press 'e' to enroll.")


# --- Main Recognition Loop ---
while True:
    ret, frame = video_capture.read()
    # Create a clean copy of the frame for saving/enrollment purposes later
    clean_frame_copy = frame.copy()
    
    if not ret:
        print("Failed to capture video. Exiting.")
        break

    # ----------------------------------------------------
    # START: Face Recognition Frame Skipping Logic
    # ----------------------------------------------------
    
    # Only run the expensive recognition logic every N frames
    if frame_skip_counter % frame_skip_rate == 0:
        
        # 1. Scale down the frame for faster processing (1/4 size)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Update the global lists that will be used for display
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = [] 
        
        # Loop through encodings and find matches
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                mark_attendance(name)

            face_names.append(name)

    # Increment the counter every loop cycle
    frame_skip_counter += 1
    
    # Reset the counter to 0 when it reaches the skip rate
    if frame_skip_counter >= frame_skip_rate:
        frame_skip_counter = 0

    # ----------------------------------------------------
    # END: Face Recognition Frame Skipping Logic
    # ----------------------------------------------------
    
    
    # --- Display and Enrollment Logic ---
    unknown_face_detected = False
    
    # The display logic uses the last calculated face_locations and face_names (even if skipped)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up the coordinates
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) 
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # New Enrollment Check: If 'Unknown', prompt user to enroll
        if name == "Unknown":
            unknown_face_detected = True
            cv2.putText(frame, "Press 'E' to Enroll", (left, top - 10), font, 0.7, (0, 0, 255), 2)


    cv2.imshow('Face Recognition Attendance System', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    # New Enrollment Trigger (ONLY ONE BLOCK)
    if unknown_face_detected and key == ord('e'):
        print("\n--- Enrollment Mode Activated ---")
        
        # 1. Get the new person's name from the user via terminal
        print("Please enter the name for this new person (e.g., 'Sara').")
        
        # Input() automatically blocks and pauses the script execution
        new_name = input("Enter Name: ").strip()
        
        if new_name:
            # 2. Create the new folder path
            new_person_path = os.path.join(DATASET_DIR, new_name)
            os.makedirs(new_person_path, exist_ok=True)
            
            # 3. Save the clean frame copy (no text/boxes) to the new folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{new_name}_{timestamp}.jpg"
            save_path = os.path.join(new_person_path, image_filename)
            
            cv2.imwrite(save_path, clean_frame_copy) # USING THE CLEAN COPY
            print(f"Image saved to: {save_path}")
            
            # 4. Process all images and update encodings.pkl
            re_encode_faces()
            
            # Optional: Log attendance immediately for the new user
            mark_attendance(new_name)
            
        else:
            print("Name not entered. Enrollment cancelled.")

video_capture.release()
cv2.destroyAllWindows()
print("\nSystem Shutdown.")