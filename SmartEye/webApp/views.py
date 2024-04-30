from django.shortcuts import render
from django.conf import settings
from django.http import *
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path  # Import Path from pathlib


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
print(torch.cuda.is_available())

classesStr = ""

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

        # Call write_results to get additional information
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get the frame number as an integer
        p = Path(f'frame_{frame_number}.jpg')  # Create a Path object with the frame number
        log_string = model.predictor.getClasses(0)
        global classesStr
        classesStr = log_string

        # Display the log_string on the image
        cv2.putText(img, log_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert the processed frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', img)
        frame_bytes = jpeg.tobytes()

        # Yield the frame as a stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def fetchClasses(request):
    global classesStr
    temp = classesStr
    cupNum = 0
    femaleNum = 0
    maleNum = 0
    phoneNum = 0
    glassesNum = 0
    headphoneNum = 0
    keyboardNum = 0
    laptopNum = 0
    penNum = 0
    shoeNum = 0
    if not temp.__contains__('('):
        if temp.__contains__('Cup'):
            if temp.split(' Cup')[0].__contains__(', '):
                cupNum = int(temp.split(' Cup')[0].split(', ')[1])
            else:
                cupNum = int(temp.split(' Cup')[0])
        if temp.__contains__('Female'):
            if temp.split(' Female')[0].__contains__(', '):
                femaleNum = int(temp.split(' Female')[0].split(', ')[1])
            else:
                femaleNum = int(temp.split(' Female')[0])
        if temp.__contains__('Male'):
            if temp.split(' Male')[0].__contains__(', '):
                maleNum = int(temp.split(' Male')[0].split(', ')[1])
            else:
                maleNum = int(temp.split(' Male')[0])
        if temp.__contains__('Phone'):
            if temp.split(' Phone')[0].__contains__(', '):
                phoneNum = int(temp.split(' Phone')[0].split(', ')[1])
            else:
                phoneNum = int(temp.split(' Phone')[0])
        if temp.__contains__('Glasses'):
            if temp.split(' Glasses')[0].__contains__(', '):
                glassesNum = int(temp.split(' Glasses')[0].split(', ')[1])
            else:
                glassesNum = int(temp.split(' Glasses')[0])
        if temp.__contains__('Headphone'):
            if temp.split(' Headphone')[0].__contains__(', '):
                headphoneNum = int(temp.split(' Headphone')[0].split(', ')[1])
            else:
                headphoneNum = int(temp.split(' Headphone')[0])
        if temp.__contains__('Keyboard'):
            if temp.split(' Keyboard')[0].__contains__(', '):
                keyboardNum = int(temp.split(' Keyboard')[0].split(', ')[1])
            else:
                keyboardNum = int(temp.split(' Keyboard')[0])
        if temp.__contains__('Laptop'):
            if temp.split(' Laptop')[0].__contains__(', '):
                laptopNum = int(temp.split(' Laptop')[0].split(', ')[1])
            else:
                laptopNum = int(temp.split(' Laptop')[0])
        if temp.__contains__('Pen'):
            if temp.split(' Pen')[0].__contains__(', '):
                penNum = int(temp.split(' Pen')[0].split(', ')[1])
            else:
                penNum = int(temp.split(' Pen')[0])
        if temp.__contains__('Shoe'):
            if temp.split(' Female')[0].__contains__(', '):
                shoeNum = int(temp.split(' Shoe')[0].split(', ')[1])
            else:
                shoeNum = int(temp.split(' Shoe')[0])
    list = []
    list.append(cupNum)
    list.append(femaleNum)
    list.append(maleNum)
    list.append(phoneNum)
    list.append(glassesNum)
    list.append(headphoneNum)
    list.append(keyboardNum)
    list.append(laptopNum)
    list.append(penNum)
    list.append(shoeNum)
    context = {'Cup': cupNum, 'Female': femaleNum, 'Male': maleNum,
               'Phone': phoneNum, 'Glasses': glassesNum, 'Headphone': headphoneNum,
               'Keyboard': keyboardNum, 'Laptop': laptopNum, 'Pen': penNum, 'Shoe': shoeNum}
    return JsonResponse(context)

def video_feed(request, isstream):
    return StreamingHttpResponse(stream(isstream), content_type='multipart/x-mixed-replace; boundary=frame')

def stopStream(request):
    global var
    var = 0
    return render(request, 'home.html')
