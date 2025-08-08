import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image_and_histogram(image_path, mode='color'):
    """
    Affiche une image et son histogramme (niveaux de gris ou couleurs).

    Paramètres :
    ------------
    image_path : str
        Chemin de l'image
    mode : str
        'gray' pour niveaux de gris, 'color' pour histogramme couleur
    """
    if mode == 'gray':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Erreur de chargement de l'image.")
            return

        # Calcul histogramme
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        # Affichage
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Image en niveaux de gris')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.plot(hist, color='black')
        plt.title('Histogramme des niveaux de gris')
        plt.xlabel('Niveau de gris')
        plt.ylabel('Nombre de pixels')
        plt.grid()
        plt.tight_layout()
        plt.show()

    elif mode == 'color':
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print("Erreur de chargement de l'image.")
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        colors = ('b', 'g', 'r')
        plt.figure(figsize=(12, 5))

        # Affichage image
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title('Image couleur')
        plt.axis('off')

        # Affichage histogrammes B, G, R
        plt.subplot(1, 2, 2)
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image_bgr], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title('Histogrammes des canaux de couleur')
        plt.xlabel('Intensité (0–255)')
        plt.ylabel('Nombre de pixels')
        plt.grid()
        plt.tight_layout()
        plt.show()

    else:
        print("Mode non supporté. Utilisez 'gray' ou 'color'.")

# Exemple d'utilisation :
# En niveaux de gris
show_image_and_histogram('images/tisdrin.jpg', mode='gray')

# En couleur
show_image_and_histogram('images/tisdrin.jpg', mode='color')
