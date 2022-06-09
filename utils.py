import cv2
import numpy as np
import serial
from SendEmail import *

import os
import time
texts=[]
import json

accessKeyr=''
secretAccessKeyr=''

ser=serial.Serial('COM3',9600,timeout=0.5)
ser.close()
ser.open()
pFlag1=0
pFlag2=0

# Draw a prediction box with confidence and title
def draw_prediction(frame, classes, classId, conf, left, top, right, bottom):
    global texts, pFlag1,pFlag2

    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    # Assign confidence to label
    label = '%.2f' % conf

    # Print a label of class.
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    texts.append(classes[classId])
    cv2.imwrite('test.jpg',frame)

    if(classes[classId]=='person'):
        pFlag1+=1
        pFlag2=0
    else:
        pFlag2+=1
        pFlag1=0
    print(pFlag1,pFlag2)
    if pFlag1==1:
        send_email('otp.service@makeskilled.com','kolisettysai26@gmail.com','Person Identified', 'Hi Hello, Your Bot Here','test.jpg')
        print('Sending Person')
        ser.write('person'.encode('utf-8'))
    if pFlag2==1:
        print('Sending noperson')
        ser.write('noperson'.encode('utf-8'))

# Process frame, eliminating boxes with low confidence scores and applying non-max suppression
def process_frame(frame, outs, classes, confThreshold, nmsThreshold):
    global texts
    # Get the width and height of the image
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Network produces output blob with a shape NxC where N is a number of
    # detected objects and C is a number of classes + 4 where the first 4
    # numbers are [center_x, center_y, width, height]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                # Scale the detected coordinates back to the frame's original width and height
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # Save the classId, confidence and bounding box for later use
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        texts.append(classes[classId])
        draw_prediction(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)
    else:
        print('Sending noperson')
        ser.write('noperson'.encode('utf-8'))

def process_frame1(frame):
    cv2.imwrite('test.jpg',frame)
    imageSource=open("test.jpg",'rb')
    client=boto3.client('rekognition',aws_access_key_id=accessKeyr,aws_secret_access_key=secretAccessKeyr,region_name=region)
    response=client.detect_labels(Image={'Bytes':imageSource.read()},MaxLabels=1)
    # response=json.loads(response)
    label=(response['Labels'][0]['Name'])
    print(label)
    if(label=='Person'):
        send_email('otp.service@makeskilled.com','kolisettysai26@gmail.com','Person Identified', 'Hi Hello, Your Bot Here','test.jpg')
        ser.write('person'.encode('utf-8'))
    else:
        ser.write('noperson'.encode('utf-8'))