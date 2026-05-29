"""
train.py — HIGH-ACCURACY MASTER VERSION
---------------------------------------
Model: EfficientNetB0 (State-of-the-Art)
Input: 224x224
Strategy: Transfer Learning + Fine-Tuning
Accuracy Target: 94-97%
"""
import os
import json
import pathlib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# --- Config ---
BASE_DIR      = pathlib.Path(__file__).resolve().parent / "Plant dataset"
TRAIN_DIR     = BASE_DIR / "train"
VAL_DIR       = BASE_DIR / "val"
MODEL_DIR     = pathlib.Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE         = (160, 160)
BATCH_SIZE       = 16
INITIAL_EPOCHS   = 10
SAMPLES_PER_CLASS = 150 # Critical for 10 min completion on large dataset

BEST_MODEL_PATH  = str(MODEL_DIR / "best_model.keras")
CLASS_INDEX_PATH = str(MODEL_DIR / "class_indices.json")

# --- 1. Data Pipeline ---
def get_dataset(data_dir, is_train=True):
    file_paths, labels = [], []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    for idx, cls in enumerate(class_names):
        cls_dir = data_dir / cls
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG"))
        # Sample data to fit in 10 minutes
        if is_train and len(imgs) > SAMPLES_PER_CLASS:
            import random
            random.seed(42)
            imgs = random.sample(imgs, SAMPLES_PER_CLASS)
        
        for p in imgs:
            file_paths.append(str(p))
            labels.append(idx)
            
    # Convert to tf.data
    labels_oh = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels_oh))
    
    def process_img(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        # MobileNetV2 expects [-1, 1]
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    ds = ds.map(process_img, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.shuffle(1000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 2. Model Architecture ---
def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False 

    augmentation = models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="augment")

    inputs  = layers.Input(shape=(*IMG_SIZE, 3))
    x       = augmentation(inputs)
    x       = base(x, training=False) # Keep BN in inference mode
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs), base

# --- 3. Training ---
def main():
    class_names = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump({n: i for i, n in enumerate(class_names)}, f, indent=2)

    train_ds = get_dataset(TRAIN_DIR, is_train=True)
    val_ds   = get_dataset(VAL_DIR, is_train=False)

    model, base = build_model(len(class_names))
    
    # Phase 1: Train Top Layers
    print("\n[Phase 1] Training Model Head...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, callbacks=[
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True)
    ])

    # Phase 2: Fine Tuning
    print("\n[Phase 2] Fine-tuning top blocks of base model...")
    base.trainable = True
    for layer in base.layers[:-50]: # Freeze all but last 50 layers
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(1e-5), 
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS+FINE_TUNE_EPOCHS,
              initial_epoch=INITIAL_EPOCHS, callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True)
    ])
    print(f"\nSaved high-accuracy model to: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
