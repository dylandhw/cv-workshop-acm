import cv2 
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils 
mp_face_mesh = mp.solutions.face_mesh 

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

capture = cv2.VideoCapture(0)

while capture.isOpened():
    success, frame = capture.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    frame.flags.writeable = True
    
    if results.multi_face_landmakrs:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION, 
                landmark_drawing_spec=None, 
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )