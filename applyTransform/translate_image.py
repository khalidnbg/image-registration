import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_affine_transform(image, pts1, pts2):
    """
    Apply affine transformation to an image.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    pts1 : numpy.ndarray
        Source points (3 points)
    pts2 : numpy.ndarray
        Destination points (3 points)

    Returns:
    --------
    transformed_img : numpy.ndarray
        Affine transformed image
    """
    rows, cols = image.shape[:2]

    # Calculate affine transformation matrix
    M_affine = cv2.getAffineTransform(pts1, pts2)

    # Apply affine transformation
    transformed_img = cv2.warpAffine(image, M_affine, (cols, rows))

    return transformed_img


def demonstrate_affine(image_path, save_path='affine_image.jpg'):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    rows, cols = img.shape[:2]

    # Source points (top-left, top-right, bottom-left)
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])

    # Destination points with stronger transformation:
    # - Décalage important
    # - Rotation + cisaillement accentué
    # - Échelle changée
    pts2 = np.float32([
        [80, 150],   # plus bas et décalé
        [250, 20],   # plus haut et plus à droite
        [130, 280]   # beaucoup plus bas et à droite
    ])

    # Apply affine transformation
    transformed = apply_affine_transform(img, pts1, pts2)

    # Save the transformed image
    cv2.imwrite(save_path, transformed)
    print(f"Affine transformed image saved to {save_path}")

    # Display original and transformed images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(transformed, cmap='gray'), plt.title('Transformation affine pour l\'image')
    plt.tight_layout()
    plt.show()

    return img, transformed


# Example usage
original, transformed = demonstrate_affine('images/tisdrin.png', 'images/tisdrin_affine_strong.jpg')
