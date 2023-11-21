import cv2
import numpy as np
from efficientnet_pytorch import EfficientNetB0
from efficientnet_pytorch import preprocess_input
labels = {
    0: 'person', 
    1: 'bicycle', 
    2: 'car', 
    3: 'motorcycle',
    4: 'bus',
    5: 'truck',
    6: 'van',
    7: 'traffic light',
    8: 'fire hydrant',
    9: 'stop sign',
    10: 'parking meter',
    11: 'bench',
    12: 'bird',
    13: 'cat',
    14: 'dog',
    15: 'horse',
    16: 'cow',
    17: 'sheep',
    18: 'elephant',
    19: 'bear',
    20: 'zebra',
    21: 'giraffe',
    22: 'backpack',
    23: 'umbrella',
    24: 'handbag',
    25: 'tie',
    26: 'suitcase',
    27: 'frisbee',
    28: 'skis',
    29: 'snowboard',
    30: 'sports ball',
    31: 'kite',
    32: 'baseball bat',
    33: 'baseball glove',
    34: 'skateboard',
    35: 'surfboard',
    36: 'tennis racket',
    37: 'bottle',
    38: 'wine glass',
    39: 'cup',
    40: 'fork',
    41: 'knife',
    42: 'spoon',
    43: 'bowl',
    44: 'banana',
    45: 'apple',
    46: 'sandwich',
    47: 'orange',
    48: 'broccoli',
    49: 'carrot',
    50: 'hot dog',
    51: 'pizza',
    52: 'donut',
    53: 'cake',
    54: 'chair',
    55: 'couch',
    56: 'potted plant',
    57: 'bed',
    58: 'dining table',
    59: 'toilet',
    60: 'TV',
    61: 'laptop',
    62: 'mouse',
    63: 'remote',
    64: 'keyboard',
    65: 'cell phone',
    66: 'microwave',
    67: 'oven',
    68: 'toaster',
    69: 'sink',
    70: 'refrigerator',
    71: 'book',
    72: 'clock',
    73: 'vase',
    74: 'scissors',
    75: 'teddy bear',
    76: 'hair drier',
    77: 'toothbrush',
    78: 'washing machine',
    79: 'dishwasher',
    80: 'sponge',
    81: 'hair brush',
    82: 'basketball hoop',
    83: 'bicycle helmet',
    84: 'toaster',
    85: 'scarf',
    86: 'glasses',
    87: 'helmet',
    88: 'face mask',
    89: 'shoe',
    90: 'glove',
    91: 'backpack',
    92: 'tie',
    93: 'suitcase',
    94: 'frisbee',
    95: 'skis',
    96: 'snowboard',
    97: 'sports ball',
    98: 'kite',
    99: 'baseball bat',
    100: 'baseball glove',
    
}

# Load EfficientNet model
model = EfficientNetB0(weights='imagenet')

def detect_objects(frame):
    # Preprocess the frame for input to EfficientNet
    resized_frame = cv2.resize(frame, (224, 224))
    expanded_frame = np.expand_dims(resized_frame, axis=0)
    preprocessed_frame = preprocess_input(expanded_frame)
    
    # Perform object detection using EfficientNet
    predictions = model.predict(preprocessed_frame)
    class_indices = np.argmax(predictions, axis=1)
    class_labels = [labels[idx] for idx in class_indices]
    
    return class_labels

def detect_movement(frame1, frame2):
    # Perform movement detection on two consecutive frames
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small movements and compute area-based movement percentage
    movement_area = 0
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            movement_area += cv2.contourArea(contour)
    
    frame_area = frame1.shape[0] * frame1.shape[1]
    movement_percentage = (movement_area / frame_area) * 100
    
    return frame1, movement_percentage

def summarize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    
    while cap.isOpened():
        objects = detect_objects(frame1)
        frame, movement_percentage = detect_movement(frame1, frame2)
        
        # Add object labels and movement percentage to the frame
        cv2.putText(frame, f"Objects: {', '.join(objects)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Movement: {movement_percentage:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        out.write(frame)
        
        cv2.imshow("Summarized Video", frame)
        
        frame1 = frame2
        ret, frame2 = cap.read()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage: summarize_video('path/to/video.mp4')
