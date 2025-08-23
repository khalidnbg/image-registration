import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_translation(image, Tx, Ty=0):
    """
    Apply translation (Tx, Ty) to an image.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    Tx : float
        Translation in x-direction
    Ty : float
        Translation in y-direction

    Returns:
    --------
    translated_img : numpy.ndarray
        Translated image
    """
    rows, cols = image.shape[:2]

    # Create translation matrix
    M_translation = np.float32([[1, 0, Tx],
                                [0, 1, Ty]])

    # Apply translation
    translated_img = cv2.warpAffine(image, M_translation, (cols, rows))

    return translated_img

def demonstrate_translation(image_path, save_path='translated_image.jpg'):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Define translation
    Tx, Ty = 30, 12   # Décalage de 50 px à droite

    # Apply translation
    translated = apply_translation(img, Tx, Ty)

    # Save the translated image
    cv2.imwrite(save_path, translated)
    print(f"Translated image saved to {save_path}")

    # Display original and translated images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(translated, cmap='gray'), plt.title(f'Translated (Tx={Tx}, Ty={Ty})')
    plt.tight_layout()
    plt.show()

    return img, translated

# Exemple d’utilisation
original, translated = demonstrate_translation(
    'images/tisdrin_test.jpg',
    'images/tisdrin_test_30_12.jpg'
)
