import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import os

# ✅ Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
DATASET_PATH = "dataset/Comprehensive Disaster Dataset(CDD)"
MODEL_PATH = "cnn_model/disaster_cnn.h5"

# Data preparation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

print("\n✅ Classes found:", train_data.class_indices)

# Build model using MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
print("\n🧠 Training started...\n")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save
os.makedirs("cnn_model", exist_ok=True)
model.save(MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")