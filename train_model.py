import os
import json
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetV2B0

@keras.saving.register_keras_serializable()
def grayscale_to_rgb(x):
    """Konversi gambar grayscale (1 channel) ke RGB (3 channel)."""
    return tf.image.grayscale_to_rgb(x)


# --- Konfigurasi ---
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224) # Ukuran gambar yang optimal untuk EfficientNetB0
BATCH_SIZE = 16
EPOCHS = 30 # Jumlah epoch disesuaikan untuk model baru
FINE_TUNE_EPOCHS = 15
MODEL_SAVE_PATH = 'document_classifier_model.keras'
CLASS_NAMES_SAVE_PATH = 'class_names.json'


def train():
    """Fungsi utama untuk melatih model klasifikasi dokumen."""

    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"❌ Error: Folder '{DATASET_PATH}' kosong atau tidak ditemukan.")
        return

    print("📦 Memuat dataset gambar...")
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
        print("❌ Error: Tidak ada sub-folder kategori yang ditemukan di dalam 'dataset'.")
        return

    print(f"✅ Kelas yang ditemukan: {class_names}")

    with open(CLASS_NAMES_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"📁 Nama kelas disimpan di '{CLASS_NAMES_SAVE_PATH}'")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Augmentasi data yang sedikit lebih bervariasi ---
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

    print("🧠 Membangun arsitektur model dengan EfficientNetV2B0...")
    # Menggunakan model dasar yang lebih cerdas ---
    base_model = EfficientNetV2B0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 1))
    x = tf.keras.layers.Lambda(grayscale_to_rgb)(inputs)
    x = data_augmentation(x)
    
    # Ini adalah baris yang memperbaiki error ukuran gambar ---
    x = tf.keras.layers.Resizing(IMG_SIZE[0], IMG_SIZE[1])(x)
    
    # Gunakan preprocessor yang sesuai untuk EfficientNetV2
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x) # Sedikit menaikkan dropout
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    print(f"\n🚀 Memulai pelatihan awal selama {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    print("\n🔧 Memulai tahap Fine-Tuning...")
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # learning rate lebih rendah untuk fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    total_epochs = EPOCHS + FINE_TUNE_EPOCHS
    print(f"⚙️ Melanjutkan pelatihan (fine-tuning) hingga epoch ke-{total_epochs}...")
    history_fine = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1]
    )

    print(f"\n💾 Menyimpan model terlatih ke '{MODEL_SAVE_PATH}'...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model berhasil disimpan!")

    try:
        acc = history.history['accuracy'] + history_fine.history['accuracy']
        val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
        loss = history.history['loss'] + history_fine.history['loss']
        val_loss = history.history['val_loss'] + history_fine.history['val_loss']

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Akurasi Training')
        plt.plot(val_acc, label='Akurasi Validasi')
        plt.axvline(x=EPOCHS - 1, color='r', linestyle='--', label='Mulai Fine-Tuning')
        plt.legend()
        plt.title('Akurasi Model')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Loss Training')
        plt.plot(val_loss, label='Loss Validasi')
        plt.axvline(x=EPOCHS - 1, color='r', linestyle='--', label='Mulai Fine-Tuning')
        plt.legend()
        plt.title('Loss Model')

        plt.savefig('hasil_pelatihan.png')
        print("📊 Grafik hasil pelatihan disimpan sebagai 'hasil_pelatihan.png'")
    except Exception as e:
        print(f"⚠️ Tidak bisa membuat grafik: {e}")

if __name__ == '__main__':
    train()