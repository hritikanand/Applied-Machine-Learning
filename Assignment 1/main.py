import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Use GPU acceleration with mixed precision
mixed_precision.set_global_policy('mixed_float16')

#  Paths and config
train_data_dir = "/Users/hritikanand/Library/CloudStorage/OneDrive-SwinburneUniversity/Applied Machine Learning/Assignment 1/Train"
img_height, img_width = 224, 224
batch_size = 64

# Load data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)

# ---------------------------------------------
# CNN BASELINE MODEL (Light Version)
# ---------------------------------------------
def build_cnn_model(input_shape=(224, 224, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model

cnn_model = build_cnn_model(num_classes=num_classes)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining CNN Model...")
cnn_history = cnn_model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("cnn_best_model.h5", save_best_only=True)
    ]
)

# ---------------------------------------------
# Transfer Learning: MobileNetV2 Fine-Tuned
# ---------------------------------------------
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

transfer_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax', dtype='float32')
])

transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining MobileNetV2 Transfer Model...")
transfer_history = transfer_model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("mobilenet_best_model.h5", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
)

# ---------------------------------------------
# Evaluation Function
# ---------------------------------------------
def evaluate_model(name, model, val_gen):
    val_gen.reset()
    pred = np.argmax(model.predict(val_gen), axis=1)
    true = val_gen.classes
    cm = confusion_matrix(true, pred)
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    avg_acc = np.mean(acc_per_class)
    top1 = model.evaluate(val_gen, verbose=0)[1]
    print(f"\n {name} - Top-1 Accuracy: {top1:.4f}, Avg Accuracy per Class: {avg_acc:.4f}")
    return top1, avg_acc

cnn_top1, cnn_avg = evaluate_model("CNN Model", cnn_model, val_generator)
transfer_top1, transfer_avg = evaluate_model("MobileNetV2 (Fine-Tuned)", transfer_model, val_generator)

# ---------------------------------------------
# Accuracy Plot Function
# ---------------------------------------------
def plot_accuracy(history, title):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_accuracy(cnn_history, "CNN Model Accuracy")
plot_accuracy(transfer_history, "Transfer Learning (MobileNetV2 Fine-Tuned) Accuracy")

# Show if GPU was used
print("\n Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
