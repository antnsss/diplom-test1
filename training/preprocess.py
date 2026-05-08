import librosa
import numpy as np
import os

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def load_data(dataset_path):
    X, y = [], []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]
                emotion = EMOTIONS.get(emotion_code)

                if emotion:
                    path = os.path.join(root, file)
                    X.append(extract_mfcc(path))
                    y.append(emotion)

    return np.array(X), np.array(y)