import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, "data")

# 1. Define Labels (Updated to include eyebrows_raised)
LABELS = {
    "neutral": 0,
    "smile": 1,
    "frown": 2,
    "left_wink": 3,
    "right_wink": 4,
    "eyebrows_raised": 5
}

X, y = [], []

print("Loading data...")
for label_name, label_id in LABELS.items():
    folder = os.path.join(SAVE_PATH, label_name)

    # Check if folder exists
    if not os.path.exists(folder):
        print(f"Warning: Folder '{label_name}' not found. Skipping...")
        continue

    files = os.listdir(folder)
    print(f" - {label_name}: {len(files)} samples found.")

    for file in files:
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))
            # Flatten data if it's stored as a multi-dimensional array
            X.append(data.flatten())
            y.append(label_id)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("Error: No data found. Check your 'data' folder path.")
    exit()

print(f"\nTotal Dataset Size: {X.shape}")

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Model Configuration
# Increased max_iter and added early_stopping for better convergence
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # Added a third layer for complexity
    activation='relu',
    solver='adam',
    max_iter=1000,
    early_stopping=True,
    random_state=42,
    verbose=False
)

print("Training model (this may take a moment)...")
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
target_names = [name for name in LABELS.keys() if LABELS[name] in np.unique(y)]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))

# 5. Save Model
joblib.dump(model, "expression_model.pkl")
print("\nModel saved successfully as 'expression_model.pkl'")