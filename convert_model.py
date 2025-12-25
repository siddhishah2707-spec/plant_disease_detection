import tensorflow as tf

# Load the existing .h5 model
model = tf.keras.models.load_model("models/best_model.h5")

# Save it again in the new .keras format (for TensorFlow 2.20+ compatibility)
model.save("models/best_model.keras", save_format="keras")

print("✅ Model successfully converted and saved as best_model.keras")
