import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_translation(image, tx=0, ty=0):
    """
    Apply translation to an image.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    tx, ty : int
        Translation in x and y directions

    Returns:
    --------
    translated_img : numpy.ndarray
        Translated image
    """
    rows, cols = image.shape[:2]

    # Create translation matrix
    M_translation = np.float32([[1, 0, tx], [0, 1, ty]])

    # Apply translation
    translated_img = cv2.warpAffine(image, M_translation, (cols, rows))

    return translated_img

def demonstrate_translation(image_path, save_path='translated_image.jpg'):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Set translation values
    tx, ty = 15.15, 7.88  # pixels

    # Apply translation
    translated = apply_translation(img, tx, ty)

    # Save the translated image
    cv2.imwrite(save_path, translated)
    print(f"Translated image saved to {save_path}")

    # Display original and translated images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(translated, cmap='gray'), plt.title('Translated Image' + f' (tx={tx}, ty={ty})')
    plt.tight_layout()
    plt.show()

    return img, translated

# Example usage:
original, translated = demonstrate_translation('images/tisdrin.jpg', 'images/tisdrin_translated_15.15_7.88.png')
