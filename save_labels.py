import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ✅ Corrected dataset path
base_dir = os.path.join("data", "Plant_Disease_Dataset")
train_dir = os.path.join(base_dir, "train")

datagen = ImageDataGenerator(rescale=1.0/255)
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

class_labels = list(train_data.class_indices.keys())
print("Detected classes:", class_labels)

# Save to models/
os.makedirs("models", exist_ok=True)
with open("models/class_labels.pkl", "wb") as f:
    pickle.dump(class_labels, f)

print("✅ class_labels.pkl created successfully!")
