import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_fft_shift(img1, img2):
    # Étape 1 : Conversion en float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Étape 2 : Calcul FFT
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)
    
    # Étape 3 : Phase Correlation
    cross_power = (fft1 * np.conj(fft2)) / (np.abs(fft1) * np.abs(fft2) + 1e-10)
    correlation = np.fft.ifft2(cross_power).real
    print(correlation)
    # Étape 4 : Détection du pic
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Ajustement des coordonnées
    h, w = img1.shape
    if x > w // 2: x -= w
    if y > h // 2: y -= h
    
    return (x, y)

# Exemple d'utilisation
img1 = np.array([[1,2,1], [0,1,0], [1,2,1]])
img2 = np.array([[0,1,2], [0,0,1], [0,1,2]])
shift = compute_fft_shift(img1, img2)
print(f"Décalage détecté : {shift}") 

# Affichage des images et du spectre
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img1, cmap='gray'), plt.title('Image 1')
plt.subplot(132), plt.imshow(img2, cmap='gray'), plt.title('Image 2')
plt.subplot(133), plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img1))) + 1e-10), cmap='viridis')
plt.title('Spectre de Fourier (Image 1)')
plt.show() # Sortie : (1, 0)