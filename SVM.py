import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset paths
def load_images_from_folder(folder, label, image_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = img.flatten()  # Flatten the image
            images.append(img)
            labels.append(label)
    return images, labels

# Save predictions to CSV
def save_predictions_to_csv(predictions, output_file='submission.csv'):
    df = pd.DataFrame(predictions, columns=['ImageId', 'Label'])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Paths to the dataset folders
bird_folder = 'path_to_your_bird_images'  # Update with the path to your bird images
fish_folder = 'path_to_your_fish_images'  # Update with the path to your fish images

# Load bird images
bird_images, bird_labels = load_images_from_folder(bird_folder, label=0)  # label 0 for birds
# Load fish images
fish_images, fish_labels = load_images_from_folder(fish_folder, label=1)  # label 1 for fish

# Combine bird and fish images
X = np.array(bird_images + fish_images)
y = np.array(bird_labels + fish_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM model
svm = SVC(kernel='rbf')  # You can try other kernels like 'linear', 'poly', etc.

# Train the model
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save predictions to CSV
# Assuming the test images are named in a way that corresponds to their index
test_image_ids = range(len(y_pred))  # Create an image ID list
predictions = list(zip(test_image_ids, y_pred))  # Combine IDs with predictions
save_predictions_to_csv(predictions)
