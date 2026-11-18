import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ✅ Step 1: Load the Trained Model
model_path = "/content/drive/MyDrive/efficientnet_best_model.keras"
model = load_model(model_path)
print(f"✅ Loaded trained model from: {model_path}")

# ✅ Step 2: Define Class Labels (Make sure these match your training data)
class_labels = {0: "WithPollution", 1: "WithoutPollution"}

# ✅ Step 3: Load and Preprocess the Image
def preprocess_image(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)  # Load image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize (same as training)
    return img_array

# ✅ Step 4: Predict Class of a Given Image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)  # Get model predictions
    predicted_class = np.argmax(predictions)  # Get class index
    confidence = np.max(predictions)  # Get confidence score

    print(f"\n✅ Prediction: {class_labels[predicted_class]}")
    print(f"✅ Confidence Score: {confidence:.4f}")

# ✅ Step 5: Test with an Example Image
image_path = "/content/with.jpg"  # Change this to your image path
predict_image(image_path)
