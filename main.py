import cv2
face_cascade = cv2.CascadeClassifier('myhaar.xml')


img = cv2.imread("new_img.jpg")
new_width = 1080
new_height = 720

scale_factor = 0.5

resized_image = cv2.resize(img, (new_width, new_height))

cv2.imshow("Scientist image",resized_image )

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Scientist image",gray_image )

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    # Draw a rectangle around each detected face on the original image
    cv2.rectangle(gray_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with rectangles around the detected faces
cv2.imshow('Detected Faces',gray_image )
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.waitKey(0)
