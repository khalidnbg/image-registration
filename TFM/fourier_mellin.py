import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import warp_polar, rotate, AffineTransform, warp
from scipy.optimize import minimize


def high_pass_filter(image, size):
    """
    Applique un filtre passe-haut à l'image
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Créer un masque avec un cercle central à zéro (filtre passe-haut)
    mask = np.ones((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= size**2
    mask[mask_area] = 0
    
    return mask


def fourier_mellin_registration(image1, image2, high_pass_size=5, plot_steps=False):
    """
    Effectue le recalage d'image en utilisant la transformation de Fourier-Mellin.
    
    Parameters:
    -----------
    image1 : array_like
        Image de référence
    image2 : array_like
        Image à recaler
    high_pass_size : int
        Taille du filtre passe-haut
    plot_steps : bool
        Si True, affiche les étapes intermédiaires
        
    Returns:
    --------
    registered_img : array_like
        Image recalée
    (scale, angle, tx, ty) : tuple
        Paramètres de transformation estimés
    """
    # Conversion en niveaux de gris si nécessaire
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
    # Normalisation des images
    image1 = image1.astype(np.float32) / 255.0
    image2 = image2.astype(np.float32) / 255.0
    
    # Taille des images
    height, width = image1.shape
    
    # 1. Calculer la FFT des deux images
    f1 = np.fft.fft2(image1)
    f2 = np.fft.fft2(image2)
    
    # 2. Déplacer les composantes de basse fréquence au centre
    f1_shift = np.fft.fftshift(f1)
    f2_shift = np.fft.fftshift(f2)
    
    # 3. Appliquer un filtre passe-haut pour réduire la sensibilité aux variations d'illumination
    mask = high_pass_filter(f1_shift, high_pass_size)
    f1_shift_hp = f1_shift * mask
    f2_shift_hp = f2_shift * mask
    
    # 4. Calculer le spectre de magnitude
    mag1 = np.abs(f1_shift_hp)
    mag2 = np.abs(f2_shift_hp)
    
    if plot_steps:
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(np.log1p(mag1), cmap='viridis'), plt.title('Spectre magnitude 1')
        plt.subplot(122), plt.imshow(np.log1p(mag2), cmap='viridis'), plt.title('Spectre magnitude 2')
        plt.tight_layout()
        plt.show()
    
    # 5. Convertir en coordonnées log-polaires pour gérer la rotation et le redimensionnement
    lp_mag1 = warp_polar(mag1, radius=min(height, width)//2, output_shape=(height, width), scaling='log')
    lp_mag2 = warp_polar(mag2, radius=min(height, width)//2, output_shape=(height, width), scaling='log')
    
    if plot_steps:
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(lp_mag1, cmap='viridis'), plt.title('Log-Polar Spectre 1')
        plt.subplot(122), plt.imshow(lp_mag2, cmap='viridis'), plt.title('Log-Polar Spectre 2')
        plt.tight_layout()
        plt.show()
    
    # 6. Calculer la corrélation de phase pour trouver la rotation et l'échelle
    f_lp1 = np.fft.fft2(lp_mag1)
    f_lp2 = np.fft.fft2(lp_mag2)
    
    # Corrélation de phase
    phase_correlation = np.fft.ifft2(f_lp1 * np.conj(f_lp2) / (np.abs(f_lp1 * np.conj(f_lp2)) + 1e-10))
    phase_correlation = np.abs(phase_correlation)
    
    # Trouver le pic de corrélation
    row, col = np.unravel_index(np.argmax(phase_correlation), phase_correlation.shape)
    
    # 7. Calculer l'angle et l'échelle
    center_row, center_col = height // 2, width // 2
    
    # L'angle est déterminé par la position horizontale du pic
    angle = 360 * (col - center_col) / width
    
    # L'échelle est déterminée par la position verticale du pic
    # Le facteur d'échelle est exponentiel à cause de l'échelle logarithmique
    log_base = np.exp(np.log(height) / height)
    scale = log_base ** (row - center_row)
    
    if plot_steps:
        print(f"Angle estimé: {angle} degrés")
        print(f"Échelle estimée: {scale}")
    
    # 8. Appliquer la correction de l'échelle et de la rotation à l'image2
    corrected_img = image2.copy()
    
    # Correction de l'échelle
    if scale != 1.0:
        corrected_img = ndimage.zoom(corrected_img, 1.0/scale)
        
        # Ajuster la taille si nécessaire
        if corrected_img.shape[0] != height or corrected_img.shape[1] != width:
            temp = np.zeros((height, width), dtype=np.float32)
            h, w = corrected_img.shape
            h_offset = max(0, (height - h) // 2)
            w_offset = max(0, (width - w) // 2)
            temp[h_offset:min(h_offset+h, height), w_offset:min(w_offset+w, width)] = corrected_img[:min(h, height-h_offset), :min(w, width-w_offset)]
            corrected_img = temp
    
    # Correction de la rotation
    if angle != 0:
        corrected_img = rotate(corrected_img, angle, resize=False, preserve_range=True)
    
    if plot_steps:
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.imshow(image1, cmap='gray'), plt.title('Image de référence')
        plt.subplot(132), plt.imshow(image2, cmap='gray'), plt.title('Image à recaler')
        plt.subplot(133), plt.imshow(corrected_img, cmap='gray'), plt.title('Après correction rotation/échelle')
        plt.tight_layout()
        plt.show()
    
    # 9. Maintenant, trouvons la translation en utilisant la corrélation de phase
    f1 = np.fft.fft2(image1)
    f2 = np.fft.fft2(corrected_img)
    
    # Corrélation de phase
    product = f1 * np.conj(f2)
    cross_power_spectrum = product / (np.abs(product) + 1e-10)
    cc_image = np.fft.ifft2(cross_power_spectrum)
    cc_image = np.abs(cc_image)
    
    # Trouver le pic de corrélation
    ty, tx = np.unravel_index(np.argmax(cc_image), cc_image.shape)
    
    # Ajuster pour la taille FFT
    if ty > height // 2:
        ty -= height
    if tx > width // 2:
        tx -= width
    
    if plot_steps:
        print(f"Translation estimée: ({tx}, {ty}) pixels")
    
    # 10. Appliquer la translation
    transform = AffineTransform(translation=(-tx, -ty))
    registered_img = warp(corrected_img, transform, mode='wrap', preserve_range=True)
    
    if plot_steps:
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.imshow(image1, cmap='gray'), plt.title('Image de référence')
        plt.subplot(132), plt.imshow(corrected_img, cmap='gray'), plt.title('Après rotation/échelle')
        plt.subplot(133), plt.imshow(registered_img, cmap='gray'), plt.title('Image recalée')
        plt.tight_layout()
        plt.show()
    
    return registered_img, (scale, angle, tx, ty)


def refine_parameters(image1, image2, initial_params, method='Powell'):
    """
    Raffine les paramètres de transformation en utilisant l'optimisation.
    
    Parameters:
    -----------
    image1 : array_like
        Image de référence
    image2 : array_like
        Image à recaler
    initial_params : tuple
        Paramètres initiaux (scale, angle, tx, ty)
    method : str
        Méthode d'optimisation
        
    Returns:
    --------
    optimal_params : tuple
        Paramètres optimisés (scale, angle, tx, ty)
    """
    def objective_function(params):
        scale, angle, tx, ty = params
        
        # Appliquer la transformation
        temp = ndimage.zoom(image2, 1.0/scale)
        
        # Ajuster la taille
        h, w = image1.shape
        h2, w2 = temp.shape
        temp2 = np.zeros_like(image1)
        
        # Centrer l'image redimensionnée
        h_offset = max(0, (h - h2) // 2)
        w_offset = max(0, (w - w2) // 2)
        temp2[h_offset:min(h_offset+h2, h), w_offset:min(w_offset+w2, w)] = temp[:min(h2, h-h_offset), :min(w2, w-w_offset)]
        
        # Rotation
        temp2 = rotate(temp2, angle, resize=False, preserve_range=True)
        
        # Translation
        transform = AffineTransform(translation=(-tx, -ty))
        transformed = warp(temp2, transform, mode='wrap', preserve_range=True)
        
        # Calculer la différence (erreur)
        return -np.corrcoef(image1.flatten(), transformed.flatten())[0, 1]
    
    # Optimiser les paramètres
    result = minimize(objective_function, initial_params, method=method, 
                     options={'maxiter': 100, 'disp': False})
    
    return result.x


def apply_transformation(image, scale, angle, tx, ty):
    """
    Applique la transformation (échelle, rotation, translation) à l'image
    
    Parameters:
    -----------
    image : array_like
        Image à transformer
    scale : float
        Facteur d'échelle
    angle : float
        Angle de rotation en degrés
    tx, ty : int
        Translation en x et y
        
    Returns:
    --------
    transformed_image : array_like
        Image transformée
    """
    h, w = image.shape
    
    # Correction de l'échelle
    if scale != 1.0:
        temp = ndimage.zoom(image, 1.0/scale)
        
        # Ajuster la taille
        h2, w2 = temp.shape
        result = np.zeros_like(image)
        
        # Centrer l'image redimensionnée
        h_offset = max(0, (h - h2) // 2)
        w_offset = max(0, (w - w2) // 2)
        result[h_offset:min(h_offset+h2, h), w_offset:min(w_offset+w2, w)] = temp[:min(h2, h-h_offset), :min(w2, w-w_offset)]
    else:
        result = image.copy()
    
    # Correction de la rotation
    if angle != 0:
        result = rotate(result, angle, resize=False, preserve_range=True)
    
    # Correction de la translation
    if tx != 0 or ty != 0:
        transform = AffineTransform(translation=(-tx, -ty))
        result = warp(result, transform, mode='wrap', preserve_range=True)
    
    return result


# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les images
    reference_img = cv2.imread('C:/Users/khalid/OneDrive/Desktop/1.jpg', 0)  # Image de référence en niveaux de gris
    image_to_register = cv2.imread('C:/Users/khalid/OneDrive/Desktop/2.jpg', 0)  # Image à recaler
    
    if reference_img is None or image_to_register is None:
        print("Erreur: Impossible de charger les images.")
        exit()
    
    # Normalisation
    reference_img = reference_img.astype(np.float32) / 255.0
    image_to_register = image_to_register.astype(np.float32) / 255.0
    
    # Appliquer l'algorithme de Fourier-Mellin
    registered_img, (scale, angle, tx, ty) = fourier_mellin_registration(
        reference_img, image_to_register, plot_steps=True)
    
    print(f"Paramètres estimés: Échelle={scale:.3f}, Angle={angle:.2f}°, Translation=({tx}, {ty})")
    
    # Affichage des résultats
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(reference_img, cmap='gray'), plt.title('Image de référence')
    plt.subplot(132), plt.imshow(image_to_register, cmap='gray'), plt.title('Image à recaler')
    plt.subplot(133), plt.imshow(registered_img, cmap='gray'), plt.title('Image recalée')
    plt.tight_layout()
    plt.show()
    
    # Optionnel: Raffiner les paramètres
    refined_params = refine_parameters(reference_img, image_to_register, (scale, angle, tx, ty))
    refined_scale, refined_angle, refined_tx, refined_ty = refined_params
    
    print(f"Paramètres raffinés: Échelle={refined_scale:.3f}, Angle={refined_angle:.2f}°, "
          f"Translation=({refined_tx:.1f}, {refined_ty:.1f})")
    
    # Appliquer la transformation raffinée
    refined_img = apply_transformation(image_to_register, refined_scale, refined_angle, refined_tx, refined_ty)
    
    # Affichage des résultats raffinés
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(reference_img, cmap='gray'), plt.title('Image de référence')
    plt.subplot(132), plt.imshow(image_to_register, cmap='gray'), plt.title('Image à recaler')
    plt.subplot(133), plt.imshow(refined_img, cmap='gray'), plt.title('Image recalée (raffinée)')
    plt.tight_layout()
    plt.show()