import os
import numpy as np
import librosa

# 🎧 FIXED SHAPE: (40, 130)
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # padding / truncation
    if mfcc.shape[1] < 130:
        pad_width = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :130]

    return mfcc


def load_data(dataset_path):
    X = []
    y = []

    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprised"
    }

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):

                parts = file.split("-")
                emotion_code = parts[2]

                label = emotion_map.get(emotion_code)
                if label is None:
                    continue

                path = os.path.join(root, file)

                mfcc = extract_mfcc(path)

                X.append(mfcc)
                y.append(label)

    X = np.array(X)
    X = X[..., np.newaxis]  # (samples, 40, 130, 1)

    return X, np.array(y)