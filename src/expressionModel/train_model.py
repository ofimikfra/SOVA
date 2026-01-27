import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, "data")

LABELS = {
    "neutral": 0,
    "smile": 1,
    "frown": 2,
    "left_wink": 3,
    "right_wink": 4
}

X, y = [], []

for label_name, label_id in LABELS.items():
    folder = os.path.join(SAVE_PATH, label_name)
    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        X.append(data)
        y.append(label_id)

X = np.array(X)
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    max_iter=500
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "expression_model.pkl")
print("Model saved as expression_model.pkl")
