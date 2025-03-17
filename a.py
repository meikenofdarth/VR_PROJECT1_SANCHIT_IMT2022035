import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

from skimage.feature import hog

def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

print("HOG Feature Extraction Completed")
print(f"Shape of HOG feature vectors: {X_train_hog.shape}")


from skimage.feature import local_binary_pattern

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

X_train_lbp = extract_lbp_features(X_train)
X_test_lbp = extract_lbp_features(X_test)

print("LBP Feature Extraction Completed")
print(f"Shape of LBP feature vectors: {X_train_lbp.shape}")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Train SVM with HOG features
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_hog, y_train)

# Predictions
y_pred_svm = svm.predict(X_test_hog)

# Accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy with HOG Features: {accuracy_svm:.4f}")


from sklearn.neural_network import MLPClassifier

# Train MLP with HOG features
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp.fit(X_train_hog, y_train)

# Predictions
y_pred_mlp = mlp.predict(X_test_hog)

# Accuracy
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Accuracy with HOG Features: {accuracy_mlp:.4f}")
import seaborn as sns

# Compare accuracy of models
accuracies = {
    "SVM (HOG)": accuracy_svm,
    "MLP (HOG)": accuracy_mlp
}

# Plot results
plt.figure(figsize=(6, 4))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()
