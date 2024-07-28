import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

image_path = r'C:\Users\musta\OneDrive\Desktop\Virtual Zoom Gesture\p1.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

default_size = (200, 200)
image = cv2.resize(image, default_size)
img_height, img_width, _ = image.shape

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def overlay_image(frame, image, position, zoom_factor):
    frame_height, frame_width, _ = frame.shape
    x, y = position

    new_width = int(img_width * zoom_factor)
    new_height = int(img_height * zoom_factor)

    x1 = max(x - new_width // 2, 0)
    y1 = max(y - new_height // 2, 0)
    x2 = min(x + new_width // 2, frame_width)
    y2 = min(y + new_height // 2, frame_height)

    resized_image = cv2.resize(image, (x2 - x1, y2 - y1))

    frame[y1:y2, x1:x2] = resized_image
    return frame

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_distance = None
zoom_factor = 1.0
position = (img_width // 2, img_height // 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks

        if len(hand_landmarks) == 2:
            hand1 = hand_landmarks[0]
            hand2 = hand_landmarks[1]

            index_finger1 = hand1.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger2 = hand2.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            index_finger_point1 = (int(index_finger1.x * frame.shape[1]), int(index_finger1.y * frame.shape[0]))
            index_finger_point2 = (int(index_finger2.x * frame.shape[1]), int(index_finger2.y * frame.shape[0]))

            center_point = ((index_finger_point1[0] + index_finger_point2[0]) // 2, 
                            (index_finger_point1[1] + index_finger_point2[1]) // 2)

            distance = calculate_distance(index_finger_point1, index_finger_point2)

            if prev_distance:
                if distance > prev_distance:
                    zoom_factor += 0.02  
                elif distance < prev_distance:
                    zoom_factor -= 0.02 
                zoom_factor = max(0.5, min(zoom_factor, 3.0))  

            prev_distance = distance
            position = center_point

    frame = overlay_image(frame, image, position, zoom_factor)

    cv2.imshow('Hand Tracking with Image Overlay', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
