import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_contrast(image, alpha=1.0):
    """
    Apply contrast adjustment to an image.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    alpha : float
        Contrast control. Values > 1.0 increase contrast, < 1.0 decrease contrast

    Returns:
    --------
    contrast_img : numpy.ndarray
        Image with adjusted contrast
    """
    # Convert to float for calculation
    img_float = image.astype(np.float32)
    
    # Apply true contrast adjustment
    # Formula: new_pixel = (pixel - 128) * alpha + 128
    contrast_img = (img_float - 128) * alpha + 128
    
    # Clip values to valid range and convert back to uint8
    contrast_img = np.clip(contrast_img, 0, 255).astype(np.uint8)
    
    return contrast_img

def demonstrate_contrast(image_path, save_path='contrast_image.jpg'):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Set contrast value
    alpha = 4  # Contrast control (>1.0 increases, <1.0 decreases)

    # Apply contrast adjustment
    contrast_adjusted = apply_contrast(img, alpha)

    # Save the contrast-adjusted image
    cv2.imwrite(save_path, contrast_adjusted)
    print(f"Contrast-adjusted image saved to {save_path}")

    # Display original and contrast-adjusted images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(contrast_adjusted, cmap='gray'), plt.title(f'Contrast Adjusted (α={alpha}, sans unité - un facteur de multiplication)')
    plt.tight_layout()
    plt.show()

    return img, contrast_adjusted

# Example usage:
original, contrast_adjusted = demonstrate_contrast('images/elephent.png', 'images/elephant_contrast_2.png')