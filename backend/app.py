from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
import pickle
from pydub import AudioSegment
import os

app = Flask(__name__)

# ✅ FFMPEG PATH (CRITICAL FIX)
FFMPEG_DIR = r"D:\ffmpeg-8.1.1-essentials_build\bin"

AudioSegment.converter = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_DIR, "ffprobe.exe")

# 🔥 IMPORTANT: add to system PATH
os.environ["PATH"] += os.pathsep + FFMPEG_DIR

# 🔥 CORS
CORS(app, origins=["http://localhost:5173"])

# 🔥 BASE DIR
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "training", "model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "training", "label_encoder.pkl")

WEBM_PATH = os.path.join(BASE_DIR, "backend", "temp.webm")
WAV_PATH = os.path.join(BASE_DIR, "backend", "temp.wav")

# 🔥 LOAD MODEL
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "rb") as f:
    le = pickle.load(f)


# 🎧 FEATURE EXTRACTION
def extract_features(file_path):
    print("LOADING AUDIO...")

    y, sr = librosa.load(file_path, sr=22050, duration=3)

    print("EXTRACTING MFCC...")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    return np.mean(mfcc.T, axis=0)


# 🔥 CONVERT WEBM → WAV (FIXED)
def convert_to_wav(input_path):
    print("CONVERTING TO WAV...")

    audio = AudioSegment.from_file(input_path, format="webm")

    audio.export(WAV_PATH, format="wav")

    return WAV_PATH


# 🚀 HOME
@app.route("/")
def home():
    return "Backend works"


# 🚀 PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("REQUEST RECEIVED")

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        file.save(WEBM_PATH)
        print("WEBM SAVED")

        wav_path = convert_to_wav(WEBM_PATH)
        print("WAV CREATED")

        features = extract_features(wav_path)
        features = np.expand_dims(features, axis=0)

        print("FEATURES READY")

        prediction = model.predict(features, verbose=0)

        predicted_class = np.argmax(prediction)

        emotion = le.inverse_transform([predicted_class])[0]

        print("EMOTION:", emotion)

        # cleanup
        os.remove(WEBM_PATH)
        os.remove(WAV_PATH)

        return jsonify({"emotion": emotion})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)