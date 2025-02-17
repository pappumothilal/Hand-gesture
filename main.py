import cv2
import numpy as np


camera_index = 0

min_area = 10000
max_area = 100000

lower_skin = np.array([0, 20, 70])
upper_skin = np.array([20, 255, 255])

kernel = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture(camera_index)

while True:

    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.erode(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            aspect_ratio = float(w) / h

            if aspect_ratio > 2.5:

                print("Five")
            elif aspect_ratio > 1.5:

                print("Three")
            elif aspect_ratio > 1.0:

                print("Two")
            else:

                print("One")

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
