import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define paths
dataset_path = "Face-Mask-Detection/dataset/"
categories = ["with_mask", "without_mask"]

data = []
labels = []

# Load images and assign labels
for label, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))  # Resize images for consistency
        data.append(image)
        labels.append(label)  # 0 = with_mask, 1 = without_mask

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(f"Dataset Loaded: {len(X_train)} training samples, {len(X_test)} testing samples")

# Feature Extraction Functions
def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

def extract_lbp_features(images, P=8, R=1):
    lbp_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), range=(0, P+2))
        hist = hist.astype("float")
        hist /= hist.sum()  # Normalize histogram
        lbp_features.append(hist)
    return np.array(lbp_features)

# Extract features
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

X_train_lbp = extract_lbp_features(X_train)
X_test_lbp = extract_lbp_features(X_test)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_hog = scaler.fit_transform(X_train_hog)
X_test_hog = scaler.transform(X_test_hog)

X_train_lbp = scaler.fit_transform(X_train_lbp)
X_test_lbp = scaler.transform(X_test_lbp)

# Combine HOG + LBP (optional but can improve results)
X_train_combined = np.hstack((X_train_hog, X_train_lbp))
X_test_combined = np.hstack((X_test_hog, X_test_lbp))

print("Feature Extraction Completed!")
print(f"Shape of HOG features: {X_train_hog.shape}")
print(f"Shape of LBP features: {X_train_lbp.shape}")
print(f"Shape of Combined Features: {X_train_combined.shape}")

# Train SVM with Combined Features
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_combined, y_train)

# Predictions & Accuracy
y_pred_svm = svm.predict(X_test_combined)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy (HOG + LBP): {accuracy_svm:.4f}")

# Train MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train_combined, y_train)

# Predictions & Accuracy
y_pred_mlp = mlp.predict(X_test_combined)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Accuracy (HOG + LBP): {accuracy_mlp:.4f}")

# Model Evaluation - Confusion Matrix & Classification Report
print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nMLP Classification Report:")
print(classification_report(y_test, y_pred_mlp))

# Visualization - Model Comparison
accuracies = {
    "SVM (HOG+LBP)": accuracy_svm,
    "MLP (HOG+LBP)": accuracy_mlp
}

plt.figure(figsize=(6, 4))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()
