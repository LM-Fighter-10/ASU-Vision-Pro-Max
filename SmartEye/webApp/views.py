from django.shortcuts import render
from django.conf import settings
from django.http import *
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path  # Import Path from pathlib
from collections import Counter


def home(request):
    context = {
        'page': "home",
        'nav': True,
        'footer': True,
    }
    return render(request, "home.html", context)


# load model
model = YOLO('webApp/models/best (Large).pt')

var = 0
# print(torch.cuda.is_available())

classesStr = ""


def stream(isstream):
    global var
    var = isstream
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    classNames = ['Cup', 'Female', 'Glasses', 'Headphone', 'Keyboard', 'Laptop', 'Male', 'Pen', 'Phone', 'Shoe']
    while True and var == 1:
        try:
            success, img = cap.read()
            results = model(img, stream=True)

            # Initialize an empty list to store classes with confidence >= 50%
            valid_classes = []

            # Process detection results
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # Extract confidence and class name
                    confidence = round(float(box.conf[0]) * 100, 2)
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    # Filter out detections with confidence < 50%
                    if confidence >= 50:
                        # Append class name to valid_classes list
                        valid_classes.append(class_name)

                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # Prepare label text (class name + confidence)
                        label_text = f"{class_name}: {confidence}%"

                        # Draw label on the image
                        org = (x1, y1 - 10)  # Adjust position of label
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.9
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(img, label_text, org, font, fontScale, color, thickness)

            # Concatenate valid_classes to form log_string
            log_string = ', '.join(valid_classes)

            # Display the log_string on the image
            # cv2.putText(img, log_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Call write_results to get additional information
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get the frame number as an integer
            p = Path(f'frame_{frame_number}.jpg')  # Create a Path object with the frame number
            global classesStr
            classesStr = log_string

            try:
                # Convert the processed frame to JPEG format
                ret, jpeg = cv2.imencode('.jpg', img)
                frame_bytes = jpeg.tobytes()
            except Exception as e:
                continue

            # Yield the frame as a stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except FileNotFoundError as f:
            continue


def fetchClasses(request):
    global classesStr
    temp = classesStr
    context = {'Cup': 0, 'Female': 0, 'Male': 0,
               'Phone': 0, 'Glasses': 0, 'Headphone': 0,
               'Keyboard': 0, 'Laptop': 0, 'Pen': 0, 'Shoe': 0}

    if temp and not temp.__contains__('('):
        # Split the string by ', ' and count occurrences of each class
        counts = Counter(temp.split(', '))

        for class_name, count in counts.items():
            context[class_name] = count

    return JsonResponse(context)


def video_feed(request, isstream):
    return StreamingHttpResponse(stream(isstream), content_type='multipart/x-mixed-replace; boundary=frame')


def stopStream(request):
    global var
    var = 0
    return render(request, 'home.html')
