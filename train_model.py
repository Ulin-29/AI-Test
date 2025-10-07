import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
import keras

@keras.saving.register_keras_serializable()
def grayscale_to_rgb(x):
    """Konversi gambar grayscale (1 channel) ke RGB (3 channel)."""
    return tf.image.grayscale_to_rgb(x)

# --- Konfigurasi ---
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
FINE_TUNE_EPOCHS = 10
MODEL_SAVE_PATH = 'document_classifier_model.keras'
CLASS_NAMES_SAVE_PATH = 'class_names.json'

def train():
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"Error: Folder '{DATASET_PATH}' kosong atau tidak ditemukan.")
        return

    # --- Memuat Dataset ---
    print("Memuat dataset gambar...")
    train_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    class_names = train_dataset.class_names
    if not class_names:
        print("Error: Tidak ada sub-folder kategori yang ditemukan di dalam 'dataset'.")
        return
        
    print(f"Kelas yang ditemukan secara otomatis: {class_names}")
    with open(CLASS_NAMES_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Nama kelas disimpan di '{CLASS_NAMES_SAVE_PATH}'")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # --- Augmentasi Data ---
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
    ], name="data_augmentation")

    # --- Membangun Arsitektur Model ---
    print("Membangun arsitektur model...")
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 1))
    
    # Menggunakan fungsi yang sudah didaftarkan
    x = tf.keras.layers.Lambda(grayscale_to_rgb)(inputs)
    
    x = data_augmentation(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    # --- Compile & Latih Model (Tahap Awal) ---
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    print(f"\nMemulai pelatihan awal selama {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    # --- Fine-Tuning ---
    print("\nMemulai tahap Fine-Tuning...")
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"Melanjutkan pelatihan (fine-tuning) selama {FINE_TUNE_EPOCHS} epochs...")
    history_fine = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history.epoch[-1]
    )
    
    # --- Simpan Model ---
    print(f"\nMenyimpan model terlatih ke '{MODEL_SAVE_PATH}'...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model berhasil disimpan!")
    
    # --- Visualisasi Hasil ---
    try:
        acc = history.history['accuracy'] + history_fine.history['accuracy']
        val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
        loss = history.history['loss'] + history_fine.history['loss']
        val_loss = history.history['val_loss'] + history_fine.history['val_loss']

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Akurasi Training')
        plt.plot(val_acc, label='Akurasi Validasi')
        plt.axvline(x=EPOCHS-1, color='r', linestyle='--', label='Mulai Fine-Tuning')
        plt.legend()
        plt.title('Akurasi Model')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Loss Training')
        plt.plot(val_loss, label='Loss Validasi')
        plt.axvline(x=EPOCHS-1, color='r', linestyle='--', label='Mulai Fine-Tuning')
        plt.legend()
        plt.title('Loss Model')
        
        plt.savefig('hasil_pelatihan.png')
        print("Grafik hasil pelatihan disimpan sebagai 'hasil_pelatihan.png'")
    except Exception as e:
        print(f"Tidak bisa membuat grafik: {e}")

if __name__ == '__main__':
    train()