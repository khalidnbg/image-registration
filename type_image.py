import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_image_types(image, threshold=128):
    """
    Create different types of images from input image.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image (color or grayscale)
    threshold : int
        Threshold for binary conversion (0-255)

    Returns:
    --------
    color_img : numpy.ndarray
        Color image
    binary_img : numpy.ndarray
        Binary image
    grayscale_img : numpy.ndarray
        Grayscale image
    """
    # (a) Image en couleur
    if len(image.shape) == 3:
        color_img = image.copy()
    else:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # (c) Image en niveaux de gris
    if len(image.shape) == 3:
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_img = image.copy()

    # (b) Image binaire
    _, binary_img = cv2.threshold(grayscale_img, threshold, 255, cv2.THRESH_BINARY)

    return color_img, binary_img, grayscale_img

def demonstrate_image_types(image_path, save_prefix='img_types'):
    # Read the image in color
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # Set threshold
    threshold = 128

    # Create image types
    color_img, binary_img, grayscale_img = create_image_types(img_color, threshold)

    # Save images
    cv2.imwrite(f'{save_prefix}_color.jpg', color_img)
    cv2.imwrite(f'{save_prefix}_binary.jpg', binary_img)
    cv2.imwrite(f'{save_prefix}_grayscale.jpg', grayscale_img)
    print(f"Images saved with prefix: {save_prefix}")

    # Affichage des images
    plt.figure(figsize=(10, 8))

    # Première ligne
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title('Image polychrome (135146 couleurs)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title("Image binaire (2 couleurs)")
    plt.axis('off')

    # Deuxième ligne, image centrée (colonne 2)
    plt.subplot(2, 1, 2)
    plt.imshow(grayscale_img, cmap='gray')
    plt.title('Image Monochrome (256 couleurs)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return color_img, binary_img, grayscale_img

# Exemple d'utilisation :
color, binary, grayscale = demonstrate_image_types('images/tisdrin.jpg', 'images/tisdrin_img_types')
