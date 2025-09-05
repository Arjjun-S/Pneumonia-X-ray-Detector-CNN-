# Pneumonia Dataset Visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

# CONFIGURATION
labels = ['pneumonia', 'normal']  
img_size = 150                    
train_path = 'dataset/train'

# LOAD TRAINING DATA
def load_images_for_viz(data_dir):
    data = []
    for label in labels:
        folder = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(folder):
            try:
                img_arr = cv2.imread(os.path.join(folder, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print("Error loading image:", e)
    return data

print("Loading training images...")
train = load_images_for_viz(train_path)
print(f"Total training images loaded: {len(train)}")

# CLASS DISTRIBUTION PLOT
classes = ['PNEUMONIA' if i[1] == 0 else 'NORMAL' for i in train]
sns.set_style('darkgrid')
plt.figure(figsize=(6,4))
sns.countplot(classes, palette='viridis')
plt.title("Class Distribution (Training Set)")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.savefig("class_distribution.png")  # saved plot
plt.show()
print("Saved class_distribution.png")

# SAMPLE IMAGES
plt.figure(figsize=(5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])
plt.savefig("sample_pneumonia.png")
plt.show()
print("Saved sample_pneumonia.png")

plt.figure(figsize=(5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])
plt.savefig("sample_normal.png")
plt.show()
print("Saved sample_normal.png")
