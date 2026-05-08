import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from preprocess import load_data

# 1. завантаження даних
X, y = load_data("../dataset/RAVDESS")

# 2. encode емоцій
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 4. модель
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(40,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(8, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. тренування
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 6. збереження моделі
model.save("model.h5")

# 7. збереження labels
pickle.dump(le, open("label_encoder.pkl", "wb"))