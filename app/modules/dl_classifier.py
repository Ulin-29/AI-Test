import tensorflow as tf
import numpy as np
import json
import os
import keras

# --- Daftarkan fungsi kustom agar bisa dibaca saat memuat model ---
@keras.saving.register_keras_serializable()
def grayscale_to_rgb(x):
    """Konversi gambar grayscale (1 channel) ke RGB (3 channel)."""
    return tf.image.grayscale_to_rgb(x)

# --- Konfigurasi Path ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'document_classifier_model.keras')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')

# --- Variabel Global untuk Model & Class ---
_model = None
_class_names = None

def _load_model_and_classes():
    """Memuat model dan nama kelas ke memori, hanya jika belum ada."""
    global _model, _class_names
    if _model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"File model tidak ditemukan di: {MODEL_PATH}")
            if not os.path.exists(CLASS_NAMES_PATH):
                raise FileNotFoundError(f"File class_names.json tidak ditemukan di: {CLASS_NAMES_PATH}")

            # Memuat model
            _model = tf.keras.models.load_model(MODEL_PATH) 
            
            with open(CLASS_NAMES_PATH, 'r') as f:
                _class_names = json.load(f)
            print(f"✅ Model dan {len(_class_names)} kelas berhasil dimuat.")
        except Exception as e:
            print(f"❌ Gagal memuat model atau kelas: {e}")
            _model = None
            _class_names = None

def predict_page_class(image_path: str, img_size=(224, 224)):
    """
    Memprediksi kelas halaman dari path gambar.
    """
    global _model, _class_names
    if _model is None:
        _load_model_and_classes()

    if _model is None or _class_names is None:
        return "UNKNOWN", 0.0

    try:
        img = tf.keras.utils.load_img(image_path, target_size=img_size, color_mode='grayscale')
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = _model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])

        predicted_class_index = np.argmax(score)
        confidence = np.max(score)
        predicted_class_name = _class_names[predicted_class_index]

        return predicted_class_name, float(confidence)

    except Exception as e:
        print(f"Error saat memprediksi gambar {image_path}: {e}")
        return "UNKNOWN", 0.0

# Panggil fungsi load saat modul ini diimpor pertama kali
_load_model_and_classes()