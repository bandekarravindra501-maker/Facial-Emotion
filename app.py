
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model once when the server starts
MODEL_PATH = "models/emotion_cnn_best.keras"
IMG_SIZE = 48

EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Sad", "Surprise", "Neutral"
]

model = load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"].split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid camera frame"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improve contrast for face detection
        gray = cv2.equalizeHist(gray)


        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return jsonify({"face_detected": False})

        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

        roi = gray[y:y+h, x:x+w]

        face_resized = cv2.resize(
            roi, (IMG_SIZE, IMG_SIZE)
        ).astype("float32") / 255.0

        face_input = face_resized.reshape(
            1, IMG_SIZE, IMG_SIZE, 1
        )

        probabilities = model.predict(
            face_input, verbose=0
        )[0]

        predicted_index = int(np.argmax(probabilities))

        return jsonify({
            "face_detected": True,
            "emotion": EMOTION_LABELS[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                label: float(probabilities[i])
                for i, label in enumerate(EMOTION_LABELS)
            },
            "box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            },
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0])
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(debug=False)
