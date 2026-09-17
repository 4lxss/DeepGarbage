import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# ==========================================
# 1. PARAMÈTRES RÉDUITS POUR PC PORTABLE (128x128)
# ==========================================
IMAGE_SIZE = (128, 128)  # <--- Taille réduite (au lieu de 224x224)
BATCH_SIZE = 32
NUM_CLASSES = 6
BASE_DIR = "data/processed"

def load_dataset_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    file_paths = [os.path.join(BASE_DIR, p) for p in df['path']]
    labels = df['label'].values
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)  # Redimensionnement rapide en 128x128
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Mise en cache dans la RAM pour accélérer les époques suivantes
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = load_dataset_from_csv("data/processed/train.csv")

# ==========================================
# 2. DATA AUGMENTATION
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
], name="data_augmentation")

# ==========================================
# 3. ARCHITECTURE ADAPTÉE À 128x128x3
# ==========================================
def build_fast_cnn():
    model = models.Sequential([
        data_augmentation,
        
        # Bloc 1 (Entrée en 128x128x3)
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Classification
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_fast_cnn()

# ==========================================
# 4. OPTIMISEUR & CALLBACKS
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks = [
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='loss', patience=7, restore_best_weights=True, verbose=1)
]

# ==========================================
# 5. ENTRAÎNEMENT
# ==========================================
EPOCHS = 25
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 6. SAUVEGARDE
# ==========================================
model.save('cnn_model.keras')
print("Modèle entraîné en 128x128 et sauvegardé !")