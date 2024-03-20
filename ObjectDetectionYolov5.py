pip install torch torchvision torchaudio

# Cloning yolov5 github
!git clone https://github.com/ultralytics/yolov5

# Installing required dependencies for yolov5
%cd yolov5
%pip install -qr requirements.txt

# Installing dependencies
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Creating a stock model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Checking the model summary
model

# Performing object detection on a image
img = 'https://images.seattletimes.com/wp-content/uploads/2017/07/1a04f45a-6689-11e7-8665-356bf84600f6.jpg?d=960x534'
result = model(img)
result.print()

%matplotlib inline
plt.imshow(np.squeeze(result.render()))
plt.show()

# Performing object detection using yolo model using our web-cam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()