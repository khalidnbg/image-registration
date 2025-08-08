import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_noise(image, noise_type='gaussian', intensity=25):
    """
    Apply noise to an image.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    noise_type : str
        Type of noise ('gaussian', 'salt_pepper', 'uniform')
    intensity : float
        Noise intensity (écart-type pour gaussian, probabilité pour salt_pepper, amplitude pour uniform)

    Returns:
    --------
    noisy_img : numpy.ndarray
        Image with added noise
    """
    # Convert to float for calculation
    img_float = image.astype(np.float32)
    
    if noise_type == 'gaussian':
        # Gaussian noise
        noise = np.random.normal(0, intensity, image.shape)
        noisy_img = img_float + noise
        
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        noisy_img = img_float.copy()
        # Salt noise (white pixels)
        salt = np.random.random(image.shape) < intensity/2
        noisy_img[salt] = 255
        # Pepper noise (black pixels)
        pepper = np.random.random(image.shape) < intensity/2
        noisy_img[pepper] = 0
        
    elif noise_type == 'uniform':
        # Uniform noise
        noise = np.random.uniform(-intensity, intensity, image.shape)
        noisy_img = img_float + noise
    
    # Clip values to valid range and convert back to uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img

def demonstrate_noise(image_path, save_path='noisy_image.jpg'):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Set noise parameters
    noise_type = 'gaussian'  # 'gaussian', 'salt_pepper', 'uniform'
    intensity = 25           # σ pour gaussian, probabilité pour salt_pepper, amplitude pour uniform

    # Apply noise
    noisy = apply_noise(img, noise_type, intensity)

    # Save the noisy image
    cv2.imwrite(save_path, noisy)
    print(f"Noisy image saved to {save_path}")

    # Display original and noisy images
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    if noise_type == 'gaussian':
        plt.subplot(122), plt.imshow(noisy, cmap='gray'), plt.title(f'Gaussian Noise (σ={intensity}, sans unité)')
    elif noise_type == 'salt_pepper':
        plt.subplot(122), plt.imshow(noisy, cmap='gray'), plt.title(f'Salt & Pepper Noise (p={intensity}, probabilité)')
    elif noise_type == 'uniform':
        plt.subplot(122), plt.imshow(noisy, cmap='gray'), plt.title(f'Uniform Noise (A={intensity}, amplitude)')
    plt.tight_layout()
    plt.show()

    return img, noisy

original, contrast_adjusted = demonstrate_noise('images/elephent.png', 'images/elephant_contrast_2.png')