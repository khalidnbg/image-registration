import cv2
import numpy as np
import matplotlib.pyplot as plt

def increase_brightness(image, value=30):
    """
    Augmente la luminosité d'une image.

    Parameters:
    -----------
    image : numpy.ndarray
        Image d'entrée (BGR ou grayscale)
    value : int
        Valeur à ajouter à chaque pixel (positive pour éclaircir)

    Returns:
    --------
    bright_img : numpy.ndarray
        Image avec luminosité augmentée
    """
    # Convertir en uint16 pour éviter le dépassement lors de l'addition
    bright_img = cv2.add(image, np.array([value], dtype=np.uint8))
    return bright_img

# Exemple d'utilisation
image = cv2.imread('images/tisdrin.png')
bright_image = increase_brightness(image, value=50)

cv2.imwrite('images/tisdrin_bright.png', bright_image)

# Affichage avec matplotlib (convertir BGR en RGB)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))
plt.title('Luminosité augmentée')
plt.axis('off')

plt.show()
