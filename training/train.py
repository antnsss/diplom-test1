import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from preprocess import load_data

# 📦 load dataset
X, y = load_data("../dataset/RAVDESS")

# 🔤 encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# ✂️ split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 🧠 CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(40,130,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(len(np.unique(y)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 🚀 train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 💾 save model + labels
model.save("model.h5")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)