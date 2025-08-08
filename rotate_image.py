import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_rotation(image, angle):
    """
    Apply only rotation to an image.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    angle : float
        Rotation angle in degrees

    Returns:
    --------
    rotated_img : numpy.ndarray
        Rotated image
    """
    rows, cols = image.shape[:2]
    
    # Create rotation matrix (centered on the image)
    M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    
    # Apply rotation
    rotated_img = cv2.warpAffine(image, M_rotation, (cols, rows))
    
    return rotated_img

def demonstrate_rotation(image_path, save_path='rotated_image.jpg'):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Define rotation angle
    angle = 18  # in degrees

    # Apply rotation
    rotated = apply_rotation(img, angle)

    # Save the rotated image
    cv2.imwrite(save_path, rotated)
    print(f"Rotated image saved to {save_path}")

    # Display original and rotated images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(rotated, cmap='gray'), plt.title('Rotated Image')
    plt.tight_layout()
    plt.show()

    return img, rotated

# Example usage:
original, rotated = demonstrate_rotation('images', 'images/tisdrin_rotated_18.jpg')
