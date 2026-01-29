"""
OpenCV Hands-On Exercises
Author: Your Name
Date: 2025-09-06

This script is designed to teach the fundamentals of OpenCV:
- Image loading and display
- Basic image operations
- Drawing shapes and text
- Accessing pixels
- Video capture and processing
- Edge detection and saving snapshots
"""

# ------------------------------
# 1. Packages
# ------------------------------
import cv2
import numpy as np

# ------------------------------
# 2. Load and Display an Image
# ------------------------------
# Exercise 1: Load an image and display it in a window
img = cv2.imread("example.jpg")  # replace with your image path
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------
# 3. Basic Image Operations
# ------------------------------
# Exercise 2: Resize and Crop the image
resized_img = cv2.resize(img, (200, 200))
cropped_img = img[50:250, 50:250]

cv2.imshow("Resized Image", resized_img)
cv2.imshow("Cropped Image", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exercise 3: Convert to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------
# 4. Drawing on Images
# ------------------------------
# Exercise 4: Draw shapes and text
img_copy = img.copy()
cv2.rectangle(img_copy, (50, 50), (200, 200), (0, 255, 0), 2)
cv2.circle(img_copy, (300, 300), 50, (255, 0, 0), -1)
cv2.putText(img_copy, "OpenCV", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

cv2.imshow("Shapes and Text", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------
# 5. Accessing Pixels
# ------------------------------
# Exercise 5: Modify pixels
# Make top-left 50x50 square completely red
img_copy[0:50, 0:50] = [0, 0, 255]
cv2.imshow("Modified Pixels", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------
# 6. Video Capture & Processing
# ------------------------------
# Exercise 6: Capture video from webcam and show live feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw a rectangle in the center of the frame
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (w//4, h//4), (w*3//4, h*3//4), (0, 255, 0), 2)

    # Stack color and grayscale side by side
    combined = np.hstack((frame, cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)))

    cv2.imshow("Webcam Feed (Color | Grayscale)", combined)

    # Press 's' to save a snapshot
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("snapshot.jpg", frame)
        print("Snapshot saved!")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ------------------------------
# 7. Edge Detection Challenge
# ------------------------------
# Exercise 7: Apply Canny edge detection on the webcam feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    cv2.imshow("Canny Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("All exercises completed!")
