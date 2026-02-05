import cv2
import cvzone
from ultralytics import YOLO
import math
from sort import *

resource = cv2.VideoCapture('./cars.mp4')
model = YOLO("./yolov8n.pt")
classes = model.names
mask = cv2.imread('./mask.png')

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [300, 370, 673, 370]
totalCount = []

while True:
    # read each frame
    success, frame = resource.read()

    if not success:
        break

    # combine the mask and frame
    imgRegion = cv2.bitwise_and(frame, mask)
    detection = np.empty((0, 5))

    # object detection
    results = model(imgRegion, stream=True)

    # create the limit line
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in results:
        for box in result.boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)

            # Class Name
            class_name = box.cls[0]
            class_name = classes[int(class_name)]

            # Confidence
            conf = box.conf[0]
            conf = math.ceil((conf * 100)) / 100

            if (class_name == 'car' or class_name == 'truck' or class_name == 'bus') and conf >= 0.5:
                # cvzone.putTextRect(frame, f'{class_name} {conf}', (max(0, x1), max(35, y1 - 10)), scale=1.2,thickness=1, offset=3)
                # cvzone.cornerRect(frame, bbox, l=10, t=2)
                currentArray = np.array((x1, y1, x2, y2, conf))
                detection = np.vstack((detection, currentArray))
                # print(detection)

    resultsTracker = tracker.update(detection)
    cvzone.putTextRect(frame, f'{len(totalCount)}', (50, 50))

    for trackresult in resultsTracker:
        x1, y1, x2, y2, id = trackresult
        x1, y1, w, h, id = int(x1), int(y1), int(x2 - x1), int(y2 - y1), int(id)
        # print(x1,y1,x2,y2,id)

        cvzone.cornerRect(frame, (x1, y1, w, h), l=10, t=2)
        cvzone.putTextRect(frame, f'{id}', (max(0, x1), max(35, y1 - 10)), scale=2, thickness=2, offset=10,colorR=(255, 0, 0))

        xCenter = int(x1 + w / 2)
        yCenter = int(y1 + h / 2)
        cv2.circle(frame, (xCenter, yCenter), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < xCenter < limits[2] and limits[1] - 20 < yCenter < limits[3] + 20:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

resource.release()  # closeing the video/ Release the hardware/file lock
cv2.destroyAllWindows()
