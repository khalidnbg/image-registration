import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def load_images(image_path1, image_path2):
    # Charge l'image depuis le chemin donné.
    # cv2.IMREAD_GRAYSCALE : Convertit l'image en niveaux de gris.
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    print(img1.shape, img2.shape)
    # Renvoie les deux images sous forme de tableaux numpy.
    return img1, img2

def fourier_transform(img):
    return fft2(img)

def phase_correlation(f1, f2):
    # f1 * np.conj(f2) : Multiplie f1 par le conjugué de f2. Cela donne une mesure de la similarité entre les deux images dans le domaine fréquentiel.
    # np.abs(f1 * np.conj(f2)) : Calcule la magnitude du résultat pour normaliser le spectre.
    # cross_power_spectrum : Le spectre de puissance croisé normalisé, qui contient la différence de phase entre les deux images.
    cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
    print(cross_power_spectrum)
    print("*********")
    # ifft2 : Applique la Transformée de Fourier Inverse 2D pour revenir au domaine spatial.
    shift = ifft2(cross_power_spectrum)
    print(shift)
    # np.abs(shift) : Prend la magnitude du résultat pour obtenir une image réelle (la corrélation de phase).
    shift = np.abs(shift)

    # np.argmax(shift) : Trouve l'indice du maximum dans la matrice shift. 
    # Ce maximum correspond au pic de la corrélation de phase. 
    # np.unravel_index : Convertit l'indice linéaire en coordonnées 2D (ligne, colonne).
    max_idx = np.unravel_index(np.argmax(shift), shift.shape)
    print("***")
    print(shift)
    print("***")
    print(np.argmax(shift))
    print("***")
    print(max_idx)

    # shifts : Les coordonnées du pic. 
    shifts = np.array(max_idx)
    print("***")
    print(shifts)
    # Boucle for : Ajuste les coordonnées pour tenir compte du décalage circulaire 
    # de la Transformée de Fourier. Si le pic est dans la moitié supérieure ou droite
    #  de l'image, on soustrait la taille de l'image pour obtenir un décalage négatif.
    for i in range(2):
        if shifts[i] > shift.shape[i] // 2:
            shifts[i] -= shift.shape[i]

    return shifts # Le vecteur de décalage estimé (en pixels).

def register_images(img1, img2):
    f1, f2 = fourier_transform(img1), fourier_transform(img2)
    translation = phase_correlation(f1, f2)

    print(f"Décalage estimé (y, x) : {translation}")

    return translation

def visualize_results(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image Originale')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Image Déformée')


    plt.show()

if __name__ == '__main__':
    # Load images
    img1, img2 = load_images('images/reference_image.jpeg', 'images/reference_image_rotate_90.jpeg')
    
    # Visualize the images
    visualize_results(img1, img2)

    # Register images
    translation = register_images(img1, img2)

# Interprétation : 
# L'image déformée est décalée de 79 pixels vers le bas 
# et de 19 pixels vers la gauche par rapport à l'image originale.
