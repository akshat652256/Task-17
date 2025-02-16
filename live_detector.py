import torch
import torch.nn.functional as F
import cv2
import numpy as np
from model import LeNet5  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
model.load_state_dict(torch.load('lenet5.pth'))
model.eval() 


def predict_digit(image):
    image = np.uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28)) 
    image = image.astype('float32') / 255.0  
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)  
    
    with torch.no_grad():
        output = model(image)
        pred = F.softmax(output, dim=1)
        predicted_class = pred.argmax(dim=1).item()  
    
    return predicted_class


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


lower_range = np.array([170, 101, 0])  
upper_range = np.array([179, 255, 255])  


kernel = np.ones((5, 5), np.uint8)


canvas = np.zeros((720, 1280, 3))  
x1, y1 = 0, 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 800:
            x2, y2, w, h = cv2.boundingRect(c)
            if x1 == 0 and y1 == 0:
                x1, y1 = x2, y2
            else:
                canvas = cv2.line(canvas, (x1, y1), (x2, y2), [0, 255, 0], 4)
            x1, y1 = x2, y2

    
    digit = predict_digit(canvas)
    cv2.putText(frame, f'Predicted Digit: {digit}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

   
    stacked = np.hstack((canvas, frame))
    cv2.imshow('Handwritten Digit Recognition', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    key = cv2.waitKey(1)
    if key == 10:  
        break
    if key & 0xFF == ord('c'):  
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
