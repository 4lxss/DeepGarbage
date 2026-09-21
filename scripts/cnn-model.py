import os
import glob
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# ==========================================
# 1. PARAMÈTRES
# ==========================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
SORTED_DIR = "data/sorted"  # Dossier contenant cardboard, glass, metal, etc.
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Correspondance des dossier aux indices (0 à 5)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# ==========================================
# 2. PRÉPARATION DU SPLIT (70% TRAIN / 30% VAL)
# ==========================================
all_paths = []
all_labels = []

# Scan des dossiers
for class_name in CLASS_NAMES:
    class_dir = os.path.join(SORTED_DIR, class_name)
    if not os.path.exists(class_dir):
        continue
    
    # Récupérer toutes les images jpg/png/jpeg
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(glob.glob(os.path.join(class_dir, ext)))
        
    for img_path in images:
        all_paths.append(img_path)
        all_labels.append(LABEL_MAP[class_name])

df = pd.DataFrame({'path': all_paths, 'label': all_labels})

# Séparation 70% Train - 30% Val (avec stratification pour garder l'équilibre des classes)
train_df, val_df = train_test_split(
    df, 
    test_size=0.30, 
    random_state=42, 
    stratify=df['label']
)

# Sauvegarde des fichiers CSV
train_csv_path = os.path.join(OUTPUT_DIR, "train.csv")
val_csv_path = os.path.join(OUTPUT_DIR, "val.csv")

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)

print(f"Dataset divisé : {len(train_df)} images d'entraînement (70%), {len(val_df)} images d'évaluation (30%).")

# ==========================================
# 3. CHARGEMENT DU DATASET TENSORFLOW
# ==========================================
def load_dataset_from_df(dataframe):
    file_paths = dataframe['path'].values
    labels = dataframe['label'].values
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)  # 224x224
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    # Retrait de .cache() pour économiser la RAM avec 224x224
    ds = ds.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = load_dataset_from_df(train_df)

# ==========================================
# 4. ARCHITECTURE ET ENTRAÎNEMENT
# ==========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
], name="data_augmentation")

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),  # Forme Keras 3 propre en 224x224
        data_augmentation,
        
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks = [
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='loss', patience=7, restore_best_weights=True, verbose=1)
]

# Entraînement
history = model.fit(
    train_ds,
    epochs=25,
    callbacks=callbacks
)

model.save('cnn_model.keras')
print("Modèle entraîné et sauvegardé avec succès !")