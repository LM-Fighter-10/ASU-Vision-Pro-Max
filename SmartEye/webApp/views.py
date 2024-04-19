from django.shortcuts import render
from django.conf import settings
from django.http import *
import torch
import cv2
import numpy as np
from ultralytics import YOLO

def home(request):
    context = {
        'page': "home",
        'nav': True,
        'footer': True,
    }
    return render(request, "home.html", context)

# load model
model = YOLO('webApp/models/best.pt')

var = 0
print(torch.cuda.is_available())
def stream(isstream):
    global var
    var = isstream
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    classNames = ['Cup', 'Female', 'Glasses', 'Headphone', 'Keyboard', 'Laptop', 'Male', 'Pen', 'Phone', 'Shoe']
    while True and var == 1:
        success, img = cap.read()
        results = model(img, stream=True)

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

        # Convert the processed frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', img)
        frame_bytes = jpeg.tobytes()

        # Yield the frame as a stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def video_feed(request, isstream):
    return StreamingHttpResponse(stream(isstream), content_type='multipart/x-mixed-replace; boundary=frame')

def stopStream(request):
    global var
    var = 0
    return render(request, 'home.html')
