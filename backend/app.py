from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
import pickle
from pydub import AudioSegment
import os

app = Flask(__name__)

# 🔥 FFmpeg setup
FFMPEG_DIR = r"D:\ffmpeg-8.1.1-essentials_build\bin"

AudioSegment.converter = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_DIR, "ffprobe.exe")

os.environ["PATH"] += os.pathsep + FFMPEG_DIR

# 🌐 CORS
CORS(app, origins=["http://localhost:5173"])

# 📁 paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "training", "model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "training", "label_encoder.pkl")

WEBM_PATH = os.path.join(BASE_DIR, "backend", "temp.webm")
WAV_PATH = os.path.join(BASE_DIR, "backend", "temp.wav")

# 🧠 load model
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "rb") as f:
    le = pickle.load(f)

# 🎧 MFCC extraction (SAME AS TRAINING)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < 130:
        pad_width = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)))
    else:
        mfcc = mfcc[:, :130]

    return mfcc

# 🔄 convert webm → wav
def convert_to_wav(input_path):
    audio = AudioSegment.from_file(input_path, format="webm")
    audio.export(WAV_PATH, format="wav")
    return WAV_PATH


@app.route("/")
def home():
    return "Emotion API running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]
        file.save(WEBM_PATH)

        wav_path = convert_to_wav(WEBM_PATH)

        features = extract_features(wav_path)

        # 🔥 CNN input shape fix
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)

        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction)

        emotion = le.inverse_transform([predicted_class])[0]

        os.remove(WEBM_PATH)
        os.remove(WAV_PATH)

        return jsonify({"emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)