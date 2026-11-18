import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ✅ Enable GPU Usage
if not tf.config.list_physical_devices('GPU'):
    print("❌ GPU not detected. Please ensure GPU runtime is enabled in Colab.")
else:
    print("✅ GPU detected and enabled.")

# ✅ Step 1: Load the Trained Model
model_path = "/content/drive/MyDrive/efficientnet_best_model.keras"
model = load_model(model_path)
print(f"✅ Loaded trained model from: {model_path}")

# ✅ Step 2: Perform Individual Image Testing
dataset_path = "/content/UNSEEN3/UNSEEN3"
correct_predictions = 0
incorrect_predictions = 0
total_images = 0

for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            true_class = 0 if class_name == 'WithPollution' else 1
            confidence = np.max(prediction) * 100

            print(f"🟩 Image: {img_name} | Predicted: {'WithPollution' if predicted_class == 0 else 'WithoutPollution'} | Original: {class_name} | Confidence: {confidence:.2f}%")

            if predicted_class == true_class:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

            total_images += 1

# ✅ Step 3: Display Results
accuracy = correct_predictions / total_images
print(f"✅ Correctly Predicted Images: {correct_predictions}")
print(f"❌ Incorrectly Predicted Images: {incorrect_predictions}")
print(f"✅ Final Testing Accuracy: {accuracy:.4f}")