# ==============================
# Pneumonia Detection Training Script
# ==============================

import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ==============================
# CONFIGURATION
# ==============================
labels = ['pneumonia', 'normal']   # Folder names inside dataset
img_size = 150                      # Image resize dimensions

# ==============================
# LOAD DATASET
# ==============================
def load_data(data_dir):
    data = []
    for label in labels:
        folder = os.path.join(data_dir, label)
        class_num = labels.index(label)  # 0 = pneumonia, 1 = normal
        for img in os.listdir(folder):
            try:
                img_arr = cv2.imread(os.path.join(folder, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return data  # <-- return as list, not np.array


print("Loading datasets...")
train = load_data('dataset/train')
val = load_data('dataset/val')
test = load_data('dataset/test')
print("Datasets loaded successfully!")

# SPLIT FEATURES AND LABELS
def split_features_labels(data):
    x, y = [], []
    for feature, label in data:
        x.append(feature)
        y.append(label)
    return np.array(x), np.array(y)
x_train, y_train = split_features_labels(train)
x_val, y_val = split_features_labels(val)
x_test, y_test = split_features_labels(test)

# NORMALIZE & RESHAPE
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Reshape for CNN (batch_size, height, width, channels)
x_train = x_train.reshape(-1, img_size, img_size, 1)
x_val = x_val.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)

print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")

# DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)


# BUILD CNN MODEL
model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(150,150,1)),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Conv2D(64, (3,3), padding='same', activation='relu'),
    Dropout(0.1),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Conv2D(128, (3,3), padding='same', activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Conv2D(256, (3,3), padding='same', activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPool2D((2,2), padding='same'),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# TRAIN MODEL
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=2,
    factor=0.3,
    min_lr=1e-6,
    verbose=1
)

print("Starting training...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=12,
    validation_data=datagen.flow(x_val, y_val),
    callbacks=[learning_rate_reduction]
)

# EVALUATE MODEL
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# SAVE TRAINED MODEL
model.save("pneumonia_model.h5")
print("Model saved as pneumonia_model.h5")

# PLOT TRAINING HISTORY

import matplotlib.pyplot as plt

epochs = range(len(history.history['accuracy']))

plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(epochs, history.history['accuracy'], 'g-', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(epochs, history.history['loss'], 'g-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print("Training complete! Graph saved as training_history.png")
