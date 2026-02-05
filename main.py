import cv2
import cvzone
from ultralytics import YOLO
import math
from sort import *

"""
Vehicle Counting System using YOLOv8 and SORT
---------------------------------------------
This script detects vehicles (cars, trucks, buses) in a video feed, 
tracks them across frames using the SORT algorithm, and counts them 
as they cross a designated line.

Author: Sharuna
Date: 2026-02-05
Dependencies: ultralytics, cvzone, opencv-python, sort, numpy
"""

# --- Configuration & Initialization ---

# Initialize video capture from file
cap=cv2.VideoCapture('./cars.mp4')

# Load the YOLOv8 Nano model (optimized for speed)
# ensure 'yolov8n.pt' is in the root directory or provide absolute path
model=YOLO("./yolov8n.pt")

# Load class names from the model (COCO dataset classes)
class_names = model.names

# Load the masking image
# The mask is used to filter out non-road areas to improve detection accuracy
mask=cv2.imread('./mask.png')

# Initialize the SORT (Simple Online and Realtime Tracking) tracker
# max_age: Maximum frames to keep a track alive without new detections
# min_hits: Minimum detections required to start tracking an object
# iou_threshold: Intersection over Union threshold for matching
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

# Define the crossing line coordinates [x1, y1, x2, y2]
# Vehicles crossing this line will be counted
line_limits=[300,370,673,370]

# Set to store unique IDs of counted vehicles to prevent double counting
total_counts=set()

# --- Main Processing Loop ---
while True:
    success,frame=cap.read()

    # Break the loop if the video ends or cannot be read
    if not success:
        break

    # Apply the mask to the current frame using bitwise AND
    # This isolates the region of interest (ROI)
    img_region = cv2.bitwise_and(frame, mask)

    # Run YOLOv8 object detection on the masked region
    # stream=True uses a generator for better memory efficiency
    results=model(img_region,stream=True,classes=[2, 5, 7])

    # List to hold detections for the current frame before passing to tracker
    current_detections = []

    # Process each detection result
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            bbox=(x1,y1,w,h)

            # Extract confidence score and class ID
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = class_names[cls]

            # Filter logic: Only track vehicles with high confidence (>0.5)
            if current_class in ["car", "truck", "bus"] and conf > 0.5:
                # Append valid detection to list: [x1, y1, x2, y2, confidence]
                current_detections.append([x1, y1, x2, y2, conf])

    # Update the SORT tracker with the new detections
    # If no detections pass an empty array
    if current_detections:
        results_tracker = tracker.update(np.array(current_detections))
    else:
        results_tracker = tracker.update(np.empty((0, 5)))

    cvzone.putTextRect(frame, f'{len(total_counts)}', (50, 50))

    # Draw the counting line on the frame
    cv2.line(frame, (line_limits[0], line_limits[1]),(line_limits[2], line_limits[3]), (0, 0, 255), 5)

    # Process tracked objects
    for result in results_tracker:
        x1,y1,x2,y2,id=result
        x1,y1,w,h=int(x1),int(y1),int(x2-x1),int(y2-y1)
        #print(x1,y1,x2,y2,id)

        # Visuals: Draw bounding box and ID
        cvzone.cornerRect(frame, (x1,y1,w,h), l=10, t=2,colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35, y1 - 10)), scale=2,thickness=3, offset=10)

        # Calculate the center point of the vehicle
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        # Counting Logic: Check if center point crosses the line area
        # We use a buffer of +/- 20 pixels around the line's Y-coordinate
        if line_limits[0] < cx < line_limits[2] and line_limits[1] - 20 < cy < line_limits[3] + 20:
            total_counts.add(id)
            # Visual feedback: Flash line green when count occurs
            cv2.line(frame, (line_limits[0], line_limits[1]), (line_limits[2], line_limits[3]), (0, 255, 0), 5)

    # Display the total count on the screen
    cvzone.putTextRect(frame, f'Count: {len(total_counts)}', (50, 50))

    # Show the output frame
    cv2.imshow('Vehicle Counter', frame)

    # Exit condition: Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()