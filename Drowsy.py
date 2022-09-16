import mediapipe as mp
import cv2
import time
from sqlalchemy import false
from scipy.spatial import distance as dis
from playsound import playsound


def draw_landmarks(image, outputs, land_mark, color):
    height, width =image.shape[:2]
             
    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        
        cv2.circle(image, point_scale, 2, color, 1)
        
def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
            
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    
    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]
            
    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    
    top_bottom_dis = euclidean_distance(image, top, bottom)
    
    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    
    left_right_dis = euclidean_distance(image, left, right)
    
    aspect_ratio = left_right_dis/ top_bottom_dis
    
    return aspect_ratio


COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)

RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]


LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

FACE=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
       377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

landmark_style = mp_drawing.DrawingSpec((0,255,0), thickness=1, circle_radius=1)
connection_style = mp_drawing.DrawingSpec((0,0,255), thickness=1, circle_radius=1)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

frame_count = 0
min_tolerance = 7.0
min_frame = 6

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    
  while cap.isOpened():
    prev_frame_time = 0
    new_frame_time = 0
    success, frame = cap.read()
    new_frame_time = time.time()
 
    # If video finished or no Video Input
    if not success:
        continue

    font = cv2.FONT_HERSHEY_SIMPLEX

    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        continue
    draw_landmarks(image, results, FACE, COLOR_GREEN)
                    
                        
    draw_landmarks(image, results, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
    draw_landmarks(image, results, LEFT_EYE_LEFT_RIGHT, COLOR_RED)
                
    ratio_left =  get_aspect_ratio(image, results, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
                
        
    draw_landmarks(image, results, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
    draw_landmarks(image, results, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)
                
    ratio_right =  get_aspect_ratio(image, results, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
                
    ratio = (ratio_left + ratio_right)/2.0

    # Change status to Awake
    drowsyStatus = "Awake"
    
    # If ratio is greater than min increase frame_count
    if ratio > min_tolerance:
        frame_count +=1
    else:
        frame_count = 0
                    
    # If frame_count > min_frame, play sound and change status to Drowsy
    if frame_count > min_frame:
        playsound('Audio/sound.mp3')
        print("Drowsiness Detected")
        drowsyStatus = "Drowsy"
    else:
        drowsyStatus = "Awake"

    # Status text
    status = cv2.putText(image, f"Status = {drowsyStatus}", (7, 700), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
    print("Ratio: ", ratio)

    # Draw the face mesh on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None)

    prev_frame_time = new_frame_time
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    fps = int(fps)
    fps = str(fps)

    # FPS counter text
    fpsImage = cv2.putText(image, f"fps = {fps}", (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    print(fps)
    cv2.imshow('MediaPipe Face Mesh', fpsImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()