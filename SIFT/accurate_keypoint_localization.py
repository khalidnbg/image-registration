import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters
from scipy import ndimage
import cv2

def gaussian_kernel(size, sigma):
    """Crée un noyau gaussien"""
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def build_gaussian_pyramid(image, num_octaves=4, scales_per_octave=5):
    """Construit la pyramide gaussienne"""
    k = 2**(1/scales_per_octave)
    sigma = 1.6
    
    pyramid = []
    
    for octave in range(num_octaves):
        if octave == 0:
            current_image = image.copy()
        else:
            current_image = ndimage.zoom(pyramid[octave-1][-3], 0.5)
        
        octave_images = []
        
        for scale in range(scales_per_octave + 3):
            current_sigma = sigma * (k ** scale)
            filtered = ndimage.gaussian_filter(current_image, current_sigma)
            octave_images.append(filtered)
        
        pyramid.append(octave_images)
    
    return pyramid

def compute_dog_pyramid(gaussian_pyramid):
    """Calcule la pyramide des Différences de Gaussiennes (DoG)"""
    dog_pyramid = []
    
    for octave in gaussian_pyramid:
        dog_octave = []
        for i in range(len(octave) - 1):
            dog = octave[i+1] - octave[i]
            dog_octave.append(dog)
        dog_pyramid.append(dog_octave)
    
    return dog_pyramid

def is_extremum(dog_pyramid, octave_idx, scale_idx, x, y):
    """Vérifie si un pixel est un extremum local dans l'espace échelle"""
    current = dog_pyramid[octave_idx][scale_idx]
    
    if (x <= 0 or x >= current.shape[0] - 1 or 
        y <= 0 or y >= current.shape[1] - 1 or
        scale_idx <= 0 or scale_idx >= len(dog_pyramid[octave_idx]) - 1):
        return False
    
    center_value = current[x, y]
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for ds in [-1, 0, 1]:
                if dx == 0 and dy == 0 and ds == 0:
                    continue
                
                neighbor_scale = scale_idx + ds
                if neighbor_scale < 0 or neighbor_scale >= len(dog_pyramid[octave_idx]):
                    continue
                
                neighbor_value = dog_pyramid[octave_idx][neighbor_scale][x + dx, y + dy]
                
                if center_value >= 0:
                    if neighbor_value >= center_value:
                        return False
                else:
                    if neighbor_value <= center_value:
                        return False
    
    return True

def detect_initial_extrema(dog_pyramid, contrast_threshold=0.005):
    """Détecte tous les extrema initiaux (étape 1)"""
    extrema = []
    
    for octave_idx, octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(octave) - 1):
            dog_image = octave[scale_idx]
            
            for x in range(1, dog_image.shape[0] - 1):
                for y in range(1, dog_image.shape[1] - 1):
                    
                    if abs(dog_image[x, y]) < contrast_threshold:
                        continue
                    
                    if is_extremum(dog_pyramid, octave_idx, scale_idx, x, y):
                        k = 2**(1/3)
                        sigma_base = 1.6
                        effective_sigma = sigma_base * (k ** scale_idx) * (2 ** octave_idx)
                        
                        extrema.append({
                            'octave': octave_idx,
                            'scale': scale_idx,
                            'x': x * (2 ** octave_idx),
                            'y': y * (2 ** octave_idx),
                            'local_x': x,
                            'local_y': y,
                            'sigma': effective_sigma,
                            'value': dog_image[x, y],
                            'response': abs(dog_image[x, y])
                        })
    
    return extrema

def compute_hessian_2d(image, x, y):
    """Calcule la matrice hessienne 2D en un point"""
    # Dérivées secondes par différences finies
    dxx = image[x-1, y] - 2*image[x, y] + image[x+1, y]
    dyy = image[x, y-1] - 2*image[x, y] + image[x, y+1]
    dxy = (image[x+1, y+1] - image[x+1, y-1] - image[x-1, y+1] + image[x-1, y-1]) / 4
    
    return np.array([[dxx, dxy], [dxy, dyy]])

def compute_gradient_2d(image, x, y):
    """Calcule le gradient 2D en un point"""
    dx = (image[x+1, y] - image[x-1, y]) / 2
    dy = (image[x, y+1] - image[x, y-1]) / 2
    return np.array([dx, dy])

def refine_keypoint_location(dog_pyramid, keypoint, max_iterations=5):
    """Affine la localisation d'un keypoint par interpolation quadratique"""
    octave_idx = keypoint['octave']
    scale_idx = keypoint['scale']
    x, y = keypoint['local_x'], keypoint['local_y']
    
    for iteration in range(max_iterations):
        # Vérifier les limites
        if (x <= 1 or x >= dog_pyramid[octave_idx][scale_idx].shape[0] - 2 or
            y <= 1 or y >= dog_pyramid[octave_idx][scale_idx].shape[1] - 2):
            return None
        
        # Images DoG aux trois échelles
        dog_prev = dog_pyramid[octave_idx][scale_idx - 1]
        dog_curr = dog_pyramid[octave_idx][scale_idx]
        dog_next = dog_pyramid[octave_idx][scale_idx + 1]
        
        # Gradient 3D (x, y, échelle)
        dx = (dog_curr[x+1, y] - dog_curr[x-1, y]) / 2
        dy = (dog_curr[x, y+1] - dog_curr[x, y-1]) / 2
        ds = (dog_next[x, y] - dog_prev[x, y]) / 2
        gradient = np.array([dx, dy, ds])
        
        # Matrice hessienne 3D
        dxx = dog_curr[x-1, y] - 2*dog_curr[x, y] + dog_curr[x+1, y]
        dyy = dog_curr[x, y-1] - 2*dog_curr[x, y] + dog_curr[x, y+1]
        dss = dog_prev[x, y] - 2*dog_curr[x, y] + dog_next[x, y]
        
        dxy = (dog_curr[x+1, y+1] - dog_curr[x+1, y-1] - 
               dog_curr[x-1, y+1] + dog_curr[x-1, y-1]) / 4
        dxs = (dog_next[x+1, y] - dog_next[x-1, y] - 
               dog_prev[x+1, y] + dog_prev[x-1, y]) / 4
        dys = (dog_next[x, y+1] - dog_next[x, y-1] - 
               dog_prev[x, y+1] + dog_prev[x, y-1]) / 4
        
        hessian = np.array([[dxx, dxy, dxs],
                           [dxy, dyy, dys],
                           [dxs, dys, dss]])
        
        # Résoudre le système linéaire
        try:
            offset = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        
        # Si l'offset est petit, on a convergé
        if np.max(np.abs(offset)) < 0.5:
            # Calculer la valeur interpolée
            interpolated_value = dog_curr[x, y] + 0.5 * np.dot(gradient, offset)
            
            # Retourner les coordonnées raffinées
            refined_x = (x + offset[0]) * (2 ** octave_idx)
            refined_y = (y + offset[1]) * (2 ** octave_idx)
            refined_scale = scale_idx + offset[2]
            
            # Calculer le sigma effectif pour l'échelle raffinée
            k = 2**(1/3)
            sigma_base = 1.6
            effective_sigma = sigma_base * (k ** refined_scale) * (2 ** octave_idx)
            
            return {
                'x': refined_x,
                'y': refined_y,
                'scale': refined_scale,
                'octave': octave_idx,
                'local_x': x + offset[0],
                'local_y': y + offset[1],
                'sigma': effective_sigma,
                'value': interpolated_value,
                'response': abs(interpolated_value)
            }
        
        # Mise à jour de la position pour la prochaine itération
        x = int(round(x + offset[0]))
        y = int(round(y + offset[1]))
        
        # Changer d'échelle si nécessaire
        if offset[2] > 0.5 and scale_idx < len(dog_pyramid[octave_idx]) - 2:
            scale_idx += 1
        elif offset[2] < -0.5 and scale_idx > 1:
            scale_idx -= 1
    
    return None  # Pas de convergence

def filter_low_contrast(keypoints, contrast_threshold=0.03):
    """Filtre les keypoints de faible contraste"""
    return [kp for kp in keypoints if abs(kp['value']) >= contrast_threshold]

def filter_edge_responses(dog_pyramid, keypoints, edge_threshold=10):
    """Filtre les keypoints sur les arêtes usando le ratio des courbures principales"""
    filtered_keypoints = []
    
    for kp in keypoints:
        octave_idx = kp['octave']
        scale_idx = int(round(kp['scale']))
        x = int(round(kp['local_x']))
        y = int(round(kp['local_y']))
        
        # Vérifier les limites
        if (scale_idx < 1 or scale_idx >= len(dog_pyramid[octave_idx]) - 1 or
            x <= 1 or x >= dog_pyramid[octave_idx][scale_idx].shape[0] - 2 or
            y <= 1 or y >= dog_pyramid[octave_idx][scale_idx].shape[1] - 2):
            continue
        
        # Calculer la matrice hessienne 2D
        dog_image = dog_pyramid[octave_idx][scale_idx]
        hessian = compute_hessian_2d(dog_image, x, y)
        
        # Calculer trace et déterminant
        trace = hessian[0, 0] + hessian[1, 1]
        det = hessian[0, 0] * hessian[1, 1] - hessian[0, 1] * hessian[1, 0]
        
        # Éviter les déterminants négatifs ou nuls
        if det <= 0:
            continue
        
        # Test du ratio des courbures principales
        ratio = trace * trace / det
        threshold_ratio = (edge_threshold + 1) ** 2 / edge_threshold
        
        if ratio < threshold_ratio:
            filtered_keypoints.append(kp)
    
    return filtered_keypoints

def visualize_keypoint_stages(image, initial_keypoints, contrast_filtered, final_keypoints):
    """Visualise les étapes de sélection des keypoints"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # (a) Image originale
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title(f'(a) Image originale {image.shape[1]}x{image.shape[0]} pixels')
    axes[0, 0].axis('off')
    
    # (b) Extrema initiaux
    axes[0, 1].imshow(image, cmap='gray')
    if initial_keypoints:
        x_coords = [kp['y'] for kp in initial_keypoints]
        y_coords = [kp['x'] for kp in initial_keypoints]
        axes[0, 1].plot(x_coords, y_coords, 'ro', markersize=1.5, alpha=0.7)
    axes[0, 1].set_title(f'(b) {len(initial_keypoints)} extrema initiaux')
    axes[0, 1].axis('off')
    
    # (c) Après filtrage par contraste
    axes[1, 0].imshow(image, cmap='gray')
    if contrast_filtered:
        x_coords = [kp['y'] for kp in contrast_filtered]
        y_coords = [kp['x'] for kp in contrast_filtered]
        axes[1, 0].plot(x_coords, y_coords, 'go', markersize=2, alpha=0.8)
    axes[1, 0].set_title(f'(c) {len(contrast_filtered)} keypoints après seuil de contraste')
    axes[1, 0].axis('off')
    
    # (d) Keypoints finaux
    axes[1, 1].imshow(image, cmap='gray')
    if final_keypoints:
        x_coords = [kp['y'] for kp in final_keypoints]
        y_coords = [kp['x'] for kp in final_keypoints]
        
        # Utiliser l'échelle pour la taille des cercles
        for kp in final_keypoints:
            x, y = kp['y'], kp['x']
            # Calculer le sigma si il n'existe pas
            if 'sigma' in kp:
                scale = kp['sigma']
            else:
                k = 2**(1/3)
                sigma_base = 1.6
                scale = sigma_base * (k ** kp['scale']) * (2 ** kp['octave'])
            
            # Dessiner le cercle avec rayon proportionnel à l'échelle
            circle = plt.Circle((x, y), scale*3, fill=False, color='blue', linewidth=1.5, alpha=0.8)
            axes[1, 1].add_patch(circle)
            axes[1, 1].plot(x, y, 'b.', markersize=3)
    
    axes[1, 1].set_title(f'(d) {len(final_keypoints)} keypoints finaux')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Charger une image de scikit-learn
    image = data.camera()
    
    print(f"Image chargée: {image.shape}")
    
    # Normaliser l'image
    image_float = image.astype(np.float64) / 255.0
    
    # Construire les pyramides
    print("Construction des pyramides...")
    gaussian_pyramid = build_gaussian_pyramid(image_float, num_octaves=3, scales_per_octave=3)
    dog_pyramid = compute_dog_pyramid(gaussian_pyramid)
    
    # Étape 1: Détection d'extrema initiaux
    print("Détection d'extrema initiaux...")
    initial_extrema = detect_initial_extrema(dog_pyramid, contrast_threshold=0.005)
    print(f"Extrema initiaux détectés: {len(initial_extrema)}")
    
    # Étape 2a: Localisation précise
    print("Localisation précise...")
    refined_keypoints = []
    for extremum in initial_extrema:
        refined = refine_keypoint_location(dog_pyramid, extremum)
        if refined is not None:
            refined_keypoints.append(refined)
    
    print(f"Keypoints après localisation précise: {len(refined_keypoints)}")
    
    # Étape 2b: Filtrage par contraste
    print("Filtrage par contraste...")
    contrast_filtered = filter_low_contrast(refined_keypoints, contrast_threshold=0.03)
    print(f"Keypoints après filtrage par contraste: {len(contrast_filtered)}")
    
    # Étape 2c: Élimination des réponses sur les arêtes
    print("Élimination des réponses sur les arêtes...")
    final_keypoints = filter_edge_responses(dog_pyramid, contrast_filtered, edge_threshold=10)
    print(f"Keypoints finaux: {len(final_keypoints)}")
    
    # Visualisation des étapes
    visualize_keypoint_stages(image, initial_extrema, contrast_filtered, final_keypoints)
    
    # Statistiques finales
    print(f"\n=== Résumé des étapes ===")
    print(f"1. Extrema initiaux: {len(initial_extrema)}")
    print(f"2. Après localisation précise: {len(refined_keypoints)}")
    print(f"3. Après seuil de contraste: {len(contrast_filtered)}")
    print(f"4. Keypoints finaux: {len(final_keypoints)}")
    
    reduction_contrast = (len(initial_extrema) - len(contrast_filtered)) / len(initial_extrema) * 100
    reduction_edge = (len(contrast_filtered) - len(final_keypoints)) / len(contrast_filtered) * 100
    
    print(f"\nRéduction par seuil de contraste: {reduction_contrast:.1f}%")
    print(f"Réduction par filtrage d'arêtes: {reduction_edge:.1f}%")
    print(f"Réduction totale: {(len(initial_extrema) - len(final_keypoints)) / len(initial_extrema) * 100:.1f}%")

if __name__ == "__main__":
    main()