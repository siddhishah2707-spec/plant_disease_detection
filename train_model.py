import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# === Dataset Paths ===
base_dir = os.path.join("data", "Plant_Disease_Dataset")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")




# === Image Parameters ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# === Data Generators ===
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === Class Info ===
class_labels = list(train_data.class_indices.keys())
print("\nDetected Classes:")
for idx, label in enumerate(class_labels):
    print(f"{idx}: {label}")
print(f"\nTotal Classes: {len(class_labels)}\n")

# === Model Setup (Transfer Learning) ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(class_labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks ===
checkpoint_path = "models/best_model.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# === Train ===
EPOCHS = 8  # you can increase to 10 if time allows
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print("\n✅ Training complete. Best model saved to models/best_model.h5")

# === Evaluate ===
loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

# Save class names for use in Streamlit app
import pickle
with open("models/class_labels.pkl", "wb") as f:
    pickle.dump(class_labels, f)

print("✅ Class labels saved to models/class_labels.pkl")
