import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Load transparent PNG skull overlay
overlay = cv2.imread("skull.png", cv2.IMREAD_UNCHANGED)  # Must have alpha channel

# Open webcam
capture = cv2.VideoCapture(0)

# Define facial landmarks
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

while capture.isOpened():
    success, frame = capture.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get key facial coordinates
            x_forehead = int(face_landmarks.landmark[FOREHEAD].x * w)
            y_forehead = int(face_landmarks.landmark[FOREHEAD].y * h)
            x_chin = int(face_landmarks.landmark[CHIN].x * w)
            y_chin = int(face_landmarks.landmark[CHIN].y * h)
            x_left = int(face_landmarks.landmark[LEFT_CHEEK].x * w)
            x_right = int(face_landmarks.landmark[RIGHT_CHEEK].x * w)

            # Compute bounding box center and size
            center_x, center_y = (x_left + x_right)//2, (y_forehead + y_chin)//2
            face_width = int(1.3 * abs(x_right - x_left))
            face_height = int(1.5 * abs(y_chin - y_forehead))

            # Resize the overlay to fit the face
            resized = cv2.resize(overlay, (face_width, face_height))

            # Optional rotation adjustment based on head tilt
            angle = np.degrees(np.arctan2(y_chin - y_forehead, x_right - x_left))
            M = cv2.getRotationMatrix2D((face_width//2, face_height//2), angle, 1.0)
            rotated = cv2.warpAffine(resized, M, (face_width, face_height),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # Placement coordinates
            top_left = (center_x - face_width//2, center_y - face_height//2)
            y1, y2 = max(0, top_left[1]), min(h, top_left[1] + face_height)
            x1, x2 = max(0, top_left[0]), min(w, top_left[0] + face_width)

            # Adjust overlay crop to fit frame bounds
            overlay_cropped = rotated[0:(y2 - y1), 0:(x2 - x1)]

            if overlay_cropped.shape[2] == 4:  # Ensure has alpha channel
                alpha = overlay_cropped[:, :, 3:] / 255.0
                frame[y1:y2, x1:x2] = (1 - alpha) * frame[y1:y2, x1:x2] + alpha * overlay_cropped[:, :, :3]

    cv2.imshow('Skull Filter', frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

capture.release()
cv2.destroyAllWindows()
