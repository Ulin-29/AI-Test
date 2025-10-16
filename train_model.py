import os
import json
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 🧩 Mixed precision untuk RTX 3050 (lebih cepat & efisien)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

@keras.saving.register_keras_serializable()
def grayscale_to_rgb(x):
    """Konversi gambar grayscale (1 channel) ke RGB (3 channel)."""
    return tf.image.grayscale_to_rgb(x)

# --- Konfigurasi Utama ---
DATASET_PATH = 'dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 35
FINE_TUNE_EPOCHS = 15
MODEL_SAVE_PATH = 'document_classifier_model.keras'
CLASS_NAMES_SAVE_PATH = 'class_names.json'

def train():
    """Melatih model klasifikasi dokumen dengan optimasi GPU RTX 3050."""

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
        print("❌ Tidak ada subfolder kelas di dalam folder dataset.")
        return

    print(f"✅ Kelas terdeteksi: {class_names}")

    with open(CLASS_NAMES_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"📁 Nama kelas disimpan di '{CLASS_NAMES_SAVE_PATH}'")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    print("🧠 Membangun model EfficientNetV2S (GPU Optimized)...")
    base_model = EfficientNetV2S(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 1))
    x = tf.keras.layers.Lambda(grayscale_to_rgb)(inputs)
    x = data_augmentation(x)
    x = tf.keras.layers.Resizing(IMG_SIZE[0], IMG_SIZE[1])(x)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)

    # --- [PERBAIKAN] Hapus lr_schedule ---
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.001,
    #     decay_steps=8000,
    #     decay_rate=0.9
    # )

    # --- [PERBAIKAN] Gunakan learning rate awal berupa angka (float) ---
    # Ini memungkinkan ReduceLROnPlateau untuk mengubahnya nanti.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    model.summary()

    # Callback sudah benar, biarkan seperti ini
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
    ]

    print(f"\n🚀 Mulai training awal ({EPOCHS} epochs)...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\n🔧 Fine-tuning model...")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) // 2
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    total_epochs = EPOCHS + FINE_TUNE_EPOCHS
    print(f"⚙️ Fine-tuning hingga epoch ke-{total_epochs}...")
    history_fine = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] if history.epoch else 0, # Tambah pengaman jika training awal berhenti cepat
        callbacks=callbacks
    )

    print(f"\n💾 Menyimpan model final ke '{MODEL_SAVE_PATH}'...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model berhasil disimpan!")

    try:
        acc = history.history.get('accuracy', []) + history_fine.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', []) + history_fine.history.get('val_accuracy', [])
        loss = history.history.get('loss', []) + history_fine.history.get('loss', [])
        val_loss = history.history.get('val_loss', []) + history_fine.history.get('val_loss', [])

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Akurasi Training')
        plt.plot(val_acc, label='Akurasi Validasi')
        if history.epoch:
             plt.axvline(x=history.epoch[-1], color='r', linestyle='--', label='Mulai Fine-Tuning')
        plt.legend()
        plt.title('Akurasi Model')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Loss Training')
        plt.plot(val_loss, label='Loss Validasi')
        if history.epoch:
            plt.axvline(x=history.epoch[-1], color='r', linestyle='--', label='Mulai Fine-Tuning')
        plt.legend()
        plt.title('Loss Model')

        plt.savefig('hasil_pelatihan.png')
        print("📊 Grafik hasil pelatihan disimpan sebagai 'hasil_pelatihan.png'")
    except Exception as e:
        print(f"⚠️ Tidak bisa membuat grafik: {e}")

if __name__ == '__main__':
    train()