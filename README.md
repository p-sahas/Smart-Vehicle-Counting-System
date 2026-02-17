# Vehicle Counting System using YOLOv8 & SORT

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=flat&logo=opencv)

## Project Overview

This project implements a real-time **Vehicle Counting System** capable of detecting and tracking vehicles (Cars, Trucks, and Buses) in video feeds.

By integrating **YOLOv8 (You Only Look Once)** for object detection and **SORT (Simple Online and Realtime Tracking)** for object tracking, the system accurately identifies unique vehicles and counts them as they cross a designated line on the road.

This tool demonstrates the practical application of Computer Vision and Deep Learning in traffic analysis and smart city solutions.

## Key Features

* **Object Detection:** Utilizes the `YOLOv8n` (Nano) model for fast and accurate detection of vehicles.
* **Object Tracking:** Implements the **SORT Algorithm** to maintain vehicle IDs across frames, preventing double-counting.
* **Class Filtering:** Specifically filters for `Car`, `Truck`, and `Bus` classes to ignore pedestrians or other objects.
* **Region of Interest (ROI):** Uses a masking image to focus detection only on the road area, reducing false positives.
* **Counting Logic:** Counts vehicles only when they cross a specific virtual line with a defined threshold.

## Technologies Used

* **Language:** Python
* **Computer Vision:** OpenCV, cvzone
* **Deep Learning:** Ultralytics YOLOv8
* **Tracking:** SORT (Simple Online and Realtime Tracking)
* **Math/Data:** NumPy, Math

## Project Structure

```text
├── cars.mp4                    # Input Video
├── mask.png                    # Mask image for ROI
├── yolov8n.pt                  # YOLOv8 Nano Weights
├── sort.py                     # SORT Tracking Module
├── main.py                     # Main execution script
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Installation & Setup

1.**Clone the Repository**

```bash
    git clone [https://github.com/Chinthaka-Sharuna/Smart-Vehicle-Counting-System](https://github.com/Chinthaka-Sharuna/Smart-Vehicle-Counting-System.git)
    cd Vehicle-Counting-System
```

2.**Install Dependencies** Note: This project requires specific versions of libraries (e.g., NumPy < 2.0) to ensure compatibility.

- If you're running on CPU

```bash
    pip install -r requirements.txt 
```

- If you're running on GPU

```bash
    pip install -r requirements-cuda.txt 
```

3.**Download YOLO Weights** The system will automatically download yolov8n.pt on the first run, or you can place your own model file in the root directory.

## How to Run

Ensure your video file path is correct in main.py, then run:

Ensure your video file path is correct in main.py, then run:
```bash
    python main.py
```

+ 'q': Press to exit the video window.

## Methodology

1. **Frame Capture**: Reads video frames one by one.
2. **Masking**: Applies a bitwise AND operation to isolate the road.
3. **Detection**: YOLOv8 detects objects and returns bounding boxes.
4. **Tracking**: SORT assigns a unique ID to each detection box based on IoU (Intersection over Union).

2. **Masking**: Applies a bitwise AND operation to isolate the road.

3. **Detection**: YOLOv8 detects objects and returns bounding boxes.

4. **Tracking**: SORT assigns a unique ID to each detection box based on IoU (Intersection over Union).

5. **Counting**: The system checks if the center point of a tracked object crosses the defined line coordinates.

## Credits & Acknowledgements

+ **SORT Algorithm**: The tracking logic is based on the original implementation by [Alex Bewley](https://github.com/abewley/sort).

+ **YOLOv8**: Object detection models provided by [Ultralytics](https://github.com/ultralytics/ultralytics).
