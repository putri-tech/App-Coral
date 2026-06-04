import matplotlib.pyplot as plt

# Data diambil persis dari hasil log terminal Epoch 1 - 10 milik Kelompok 5
epochs = range(1, 11)
accuracy = [0.5167, 0.5333, 0.4237, 0.5000, 0.4333, 0.6333, 0.5333, 0.5000, 0.4333, 0.5167]
val_accuracy = [0.4500, 0.4500, 0.5500, 0.5500, 0.7500, 0.3500, 0.3500, 0.3500, 0.4500, 0.6000]
loss = [0.7071, 0.7025, 0.7116, 0.6904, 0.7299, 0.6726, 0.7213, 0.7087, 0.7304, 0.7213]
val_loss = [0.7528, 0.6966, 0.6901, 0.6921, 0.6746, 0.8229, 0.7411, 0.7709, 0.7686, 0.6764]

# 1. MEMBUAT GRAFIK AKURASI
plt.figure(figsize=(7, 4))
plt.plot(epochs, accuracy, marker='o', color='blue', label='Training Accuracy')
plt.plot(epochs, val_accuracy, marker='o', color='orange', label='Validation Accuracy')
plt.title('Grafik Tren Akurasi Model Terumbu Karang (EfficientNetB0)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig('screenshot_accuracy.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. MEMBUAT GRAFIK LOSS
plt.figure(figsize=(7, 4))
plt.plot(epochs, loss, marker='o', color='blue', label='Training Loss')
plt.plot(epochs, val_loss, marker='o', color='orange', label='Validation Loss')
plt.title('Grafik Tren Loss Model Terumbu Karang (EfficientNetB0)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig('screenshot_loss.png', bbox_inches='tight', dpi=300)
plt.close()

print("Sukses! Gambar 'screenshot_accuracy.png' dan 'screenshot_loss.png' telah dibuat di folder proyekmu.")