from google.colab import files
uploaded = files.upload()
import zipfile
import os

with zipfile.ZipFile("DATASET1.zip", "r") as zip_ref:
    zip_ref.extractall("/content/")

print("Extracted folders:", os.listdir("/content/DATASET1"))
import os

data_dir = "/content/DATASET1"

# List class folders inside the dataset directory
classes = os.listdir(data_dir)

# Print detected classes
print(f"Number of classes found: {len(classes)}")
print("Classes:", classes)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import numpy as np
import os
from google.colab import drive

# ✅ Ensure GPU is being used
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU Available:", gpus)
else:
    print("⚠️ No GPU detected, training will be slow.")

# ✅ Set float32 precision to avoid NaN loss issues
tf.keras.mixed_precision.set_global_policy('float32')

# Mount Google Drive
drive.mount('/content/drive')

# Define dataset path
data_dir = "/content/DATASET1"
img_size = (224, 224)
batch_size = 32  # ✅ Optimized for GPU
epochs = 10

# Load Dataset with Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_data = datagen.flow_from_directory(directory=data_dir, target_size=img_size,
                                         batch_size=batch_size, class_mode='categorical',
                                         subset='training', shuffle=True)
val_data = datagen.flow_from_directory(directory=data_dir, target_size=img_size,
                                       batch_size=batch_size, class_mode='categorical',
                                       subset='validation', shuffle=False)
print("Class indices:", train_data.class_indices)

# Load EfficientNetV2S Model
def get_efficientnet_model(input_shape=(224, 224, 3)):
    input_tensor = Input(shape=input_shape)
    base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_tensor=input_tensor)

    for layer in base_model.layers:
        layer.trainable = True

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model

# ✅ No TPU code, directly initialize model
model = get_efficientnet_model()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
checkpoint_path = "/content/drive/MyDrive/efficientnet_best_model.keras"
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Custom Callback to Print All Metrics After Each Epoch
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # ✅ Get Predictions Correctly on GPU
        val_preds = model.predict(val_data, verbose=1)

        # ✅ Extract True Labels
        val_true_labels = np.array(val_data.classes)
        val_preds_labels = np.argmax(val_preds, axis=1)  # Convert softmax outputs to class labels

        # Compute Classification Metrics
        report = classification_report(val_true_labels, val_preds_labels, target_names=list(train_data.class_indices.keys()), output_dict=True)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1_score = report["weighted avg"]["f1-score"]

        # Compute ROC-AUC
        roc_auc = roc_auc_score(tf.keras.utils.to_categorical(val_true_labels, num_classes=2), val_preds, multi_class='ovr')

        # Compute PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(val_true_labels.ravel(), val_preds[:, 1].ravel())
        pr_auc = auc(recall_curve, precision_curve)

        # ✅ Get Training & Validation Accuracy & Loss from logs
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)

        print(f"\nEpoch {epoch+1}:")
        print(f"🔹 Training Accuracy: {train_acc:.4f}")
        print(f"🔹 Validation Accuracy: {val_acc:.4f}")
        print(f"🔹 Training Loss: {train_loss:.4f}")
        print(f"🔹 Validation Loss: {val_loss:.4f}")
        print(f"🔹 Precision: {precision:.4f}")
        print(f"🔹 Recall: {recall:.4f}")
        print(f"🔹 F1-Score: {f1_score:.4f}")
        print(f"🔹 ROC-AUC: {roc_auc:.4f}")
        print(f"🔹 PR-AUC: {pr_auc:.4f}\n")

metrics_callback = MetricsCallback()

history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[checkpoint, early_stopping, metrics_callback])
print("Training Complete! Best model saved at:", checkpoint_path)


