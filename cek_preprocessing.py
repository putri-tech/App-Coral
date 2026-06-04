import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# 1. JALUR FOLDER (Sudah disesuaikan dengan folder laptop kamu)
IMAGE_FOLDER = r"C:\Users\ASUS\Downloads\KULIAH_PENGOLAHAN_CITRA\dataset_coral_terpisah\Healthy"

# 2. KONFIGURASI PREPROCESSING & AUGMENTASI
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

print("" * 50)
print("🔍 SYSTEM CHECK:")
print(f"Memeriksa folder: {IMAGE_FOLDER}")

# 3. PROSES EKSEKUSI
if not os.path.exists(IMAGE_FOLDER):
    print("❌ ERROR: Folder tidak ditemukan! Periksa kembali ejaan nama foldermu.")
else:
    file_list = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:4]
    print(f"📄 Menemukan {len(file_list)} sampel gambar untuk diproses.")
    
    if len(file_list) == 0:
        print("❌ ERROR: Folder Healthy kosong atau tidak ada gambar di dalamnya.")
    else:
        print("🔄 Sedang memproses augmentasi...")
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        for i, file_name in enumerate(file_list):
            img_path = os.path.join(IMAGE_FOLDER, file_name)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            for batch in datagen.flow(img_array, batch_size=1):
                processed_img = batch[0]
                axes[i].imshow(processed_img)
                axes[i].set_title(f"Sampel {i+1}\n(224x224)", fontsize=10)
                axes[i].axis('off')
                break
        
        plt.tight_layout()
        
        # --- MODIFIKASI DISINI: LANGSUNG SIMPAN JADI FILE FOTO ---
        NAMA_FILE_HASIL = "hasil_preprocessing_kelompok5.png"
        plt.savefig(NAMA_FILE_HASIL)
        print(f"🎉 SUKSES! Gambar hasil preprocessing berhasil disimpan dengan nama: '{NAMA_FILE_HASIL}'")
        print("Silakan cek folder 'proyek_coral_app' kamu, file gambarnya sudah ada di sana!")