import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_transformation(image, tx=0, ty=0, angle=0, scale=1.0):
    """
    Apply transformation (translation, rotation, scaling) to an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    tx, ty : int
        Translation in x and y directions
    angle : float
        Rotation angle in degrees
    scale : float
        Scaling factor
        
    Returns:
    --------
    transformed_img : numpy.ndarray
        Transformed image
    """
    rows, cols = image.shape[:2]
    
    # Create transformation matrix for rotation and scaling
    M_rot_scale = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    
    # Apply rotation and scaling
    img_rotated_scaled = cv2.warpAffine(image, M_rot_scale, (cols, rows))
    
    # Create translation matrix
    M_translation = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Apply translation
    transformed_img = cv2.warpAffine(img_rotated_scaled, M_translation, (cols, rows))
    
    return transformed_img

def demonstrate_transformation(image_path, save_path='transformed_image.jpg'):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    # Apply transformation
    tx, ty = 50, 30  # Translation
    angle = 15       # Rotation angle in degrees
    scale = 0.9      # Scaling factor
    
    transformed = apply_transformation(img, tx, ty, angle, scale)
    
    # Save the transformed image
    cv2.imwrite(save_path, transformed)
    print(f"Transformed image saved to {save_path}")
    
    # Display original and transformed images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(transformed, cmap='gray'), plt.title('Transformed Image')
    plt.tight_layout()
    plt.show()
    
    return img, transformed

# Example usage:
original, transformed = demonstrate_transformation('images/img_org_translated_img.PNG', 'images/test_image.png')