
import cv2
import numpy as np

# Start the webcam
cap = cv2.VideoCapture(0)

# Define color ranges in HSV
colors_hsv = {
    "Red": [(0, 120, 70), (10, 255, 255)],
    "Green": [(36, 50, 70), (89, 255, 255)],
    "Blue": [(94, 80, 2), (126, 255, 255)]
}

def detect_and_draw(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower, upper) in colors_hsv.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        
        mask = cv2.inRange(hsv, lower_np, upper_np)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_and_draw(frame)
    cv2.imshow("Color Detection & Sorting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
