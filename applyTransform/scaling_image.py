import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_scaling(image, scale_x, scale_y):
    """
    Apply scaling to an image but keep the original dimensions.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    scale_x : float
        Scaling factor along the x-axis
    scale_y : float
        Scaling factor along the y-axis

    Returns:
    --------
    scaled_img : numpy.ndarray
        Scaled image with original dimensions
    """
    rows, cols = image.shape[:2]

    # Scaling matrix centered at the image center
    center_x, center_y = cols / 2, rows / 2
    M_scale = cv2.getRotationMatrix2D((center_x, center_y), 0, scale_x)  # No rotation, only scaling
    M_scale[1,1] = scale_y  # Manually update y scaling

    # Apply the affine transformation while keeping the original size
    scaled_img = cv2.warpAffine(image, M_scale, (cols, rows))

    return scaled_img

def demonstrate_scaling(image_path, scale_x=1.2, scale_y=1.2, save_path='scaled_image.jpg'):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Apply scaling
    scaled = apply_scaling(img, scale_x, scale_y)

    # Save the scaled image
    cv2.imwrite(save_path, scaled)
    print(f"Scaled image saved to {save_path}")

    # Display original and scaled images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(scaled, cmap='gray'), plt.title(f'Scaled Image ({scale_x}, {scale_y})')
    plt.tight_layout()
    plt.show()

    return img, scaled

# Example usage:
original, scaled = demonstrate_scaling('images/tisdrin_tra_15_7_rot_18.jpg', scale_x=1.2, scale_y=1.2, save_path='images/tisdrin_tra_15_7_rot_18_scale_1.2.jpg')
