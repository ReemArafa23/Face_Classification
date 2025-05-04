import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Load your trained model ===
model = load_model("C:/Users/reema/OneDrive/Desktop/face_classification/face_recognition_mobilenetv2.h5")

# === Label mapping (update with your real class names) ===
idx_to_label = {
    0: "trump",
    1: "musk",
    2: "modi",
    3: "jack",
    4: "gates"
}

# === Face detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Start webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            face_img = cv2.resize(face_img, (100, 100))
        except:
            continue  # skip if face too small

        face_input = np.expand_dims(face_img / 255.0, axis=0)

        pred = model.predict(face_input)
        confidence = np.max(pred)
        class_index = np.argmax(pred)

        # Decide label
        if confidence < 0.7:
            label = "Unknown"
        else:
            label = f"{idx_to_label[class_index]} ({confidence:.2f})"

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
