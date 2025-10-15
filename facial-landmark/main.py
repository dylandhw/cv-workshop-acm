import cv2 
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils 
mp_face_wash = mp.solutions.face_mesh 

face_mesh = mp_face_wash.FaceWash(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

