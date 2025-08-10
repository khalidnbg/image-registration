import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_fourier_spectrum(image_path):
    """
    Calcule et affiche le spectre de magnitude et de phase d'une image.
    
    Args:
        image_path (str): Chemin vers l'image (niveaux de gris ou couleur)
    """
    # 1. Chargement de l'image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Conversion en niveaux de gris
    if img is None:
        raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin.")
    
    # 2. Pré-traitement (optionnel)
    img_float = img.astype(np.float32) / 255.0  # Normalisation [0,1]
    
    # 3. Calcul de la FFT 2D
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)  # Centre les basses fréquences
    
    # 4. Extraction du spectre de magnitude et de phase
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    # 5. Affichage des résultats
    plt.figure(figsize=(10, 8))
    
    # Image originale
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image Originale')
    plt.axis('off')
    
    # Spectre de magnitude (logarithmique)
    plt.subplot(2, 2, 2)
    magnitude_log = 20 * np.log(magnitude + 1e-10)  # Échelle logarithmique
    plt.imshow(magnitude_log, cmap='viridis')
    plt.title('Spectre de Magnitude (log)')
    plt.colorbar()
    plt.axis('off')
    
    # Spectre de phase
    plt.subplot(2, 1, 2)
    plt.imshow(phase, cmap='twilight')
    plt.title('Spectre de Phase')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    return magnitude, phase

# Exemple d'utilisation
if __name__ == "__main__":
    image_path = "images/tisdrin.png"  # Remplacez par votre image
    magnitude, phase = compute_fourier_spectrum(image_path)