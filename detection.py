import cv2
import numpy as np
import tensorflow as tf


# Function to load and preprocess an image


def preprocess_image(image):
    target_size = (224, 224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image/255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Loading the saved TensorFlow model for face mask detection
model = tf.keras.models.load_model('saved_model.h5')

# Loading the pre-trained face detection model from OpenCV
# This is being used to detect the face in the video  or through a webcam, not the mask
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video capture stream (0 for webcam, or file path for video file)
cap = cv2.VideoCapture(0) # We can change this to the appropriate index if using a different camera





while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Converting the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    text = "Press 'Q' to exit"
    position = (50, 50)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Performing mask detection and social distancing monitoring on each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        processed_img = preprocess_image(face_img)

        # Predicting using the loaded model for mask detection
        result = model.predict(processed_img)

        # Determining label for mask detection
        if result > 0.5:
            label = 'Mask'
            color = (0, 255, 0)  # Green for mask
        else:
            label = 'No Mask'
            color = (0, 0, 255)  # Red for no mask

        # Draw label and bounding box for mask detection
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow('Face Mask & Social Distancing Monitoring', frame)

    # We can exit the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
