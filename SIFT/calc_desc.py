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

def compute_gradient_magnitude_and_orientation(image):
    """Calcule la magnitude et l'orientation du gradient pour chaque pixel"""
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)
    
    # Gradients par différences finies (éviter les bords)
    dx[1:-1, 1:-1] = (image[1:-1, 2:] - image[1:-1, :-2]) / 2.0
    dy[1:-1, 1:-1] = (image[2:, 1:-1] - image[:-2, 1:-1]) / 2.0
    
    # Magnitude du gradient
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Orientation du gradient (en radians puis degrés)
    orientation = np.arctan2(dy, dx)
    orientation = np.degrees(orientation)
    orientation[orientation < 0] += 360
    
    return magnitude, orientation

def compute_sift_descriptor(gaussian_image, keypoint, magnitude, orientation):
    """
    Calcule le descripteur SIFT 128D pour un keypoint
    
    Le descripteur est composé de:
    - Une région 16x16 pixels autour du keypoint
    - Divisée en 4x4 sous-régions de 4x4 pixels chacune
    - Chaque sous-région génère un histogramme 8 bins (8 orientations)
    - Total: 4x4x8 = 128 dimensions
    """
    
    # Paramètres du descripteur SIFT
    DESCRIPTOR_SIZE = 16  # Taille de la région (16x16)
    SUBREGION_SIZE = 4    # Taille des sous-régions (4x4)
    NUM_BINS = 8          # Nombre de bins d'orientation par sous-région
    NUM_SUBREGIONS = 4    # 4x4 sous-régions
    
    # Position et orientation du keypoint
    kp_x = keypoint['local_x']
    kp_y = keypoint['local_y']
    kp_orientation = keypoint['orientation']
    kp_scale = keypoint['sigma']
    
    # Vérifier les limites
    half_size = DESCRIPTOR_SIZE // 2
    if (kp_x < half_size or kp_x >= gaussian_image.shape[1] - half_size or
        kp_y < half_size or kp_y >= gaussian_image.shape[0] - half_size):
        return None
    
    # Initialiser le descripteur (4x4x8 = 128 dimensions)
    descriptor = np.zeros((NUM_SUBREGIONS, NUM_SUBREGIONS, NUM_BINS))
    
    # Fenêtre gaussienne pour pondérer les contributions
    gaussian_sigma = DESCRIPTOR_SIZE / 2.0
    
    # Parcourir la région 16x16 autour du keypoint
    for i in range(-half_size, half_size):
        for j in range(-half_size, half_size):
            # Coordonnées dans l'image
            x = int(kp_x + j)
            y = int(kp_y + i)
            
            # Vérifier les limites
            if (x <= 0 or x >= gaussian_image.shape[1] - 1 or
                y <= 0 or y >= gaussian_image.shape[0] - 1):
                continue
            
            # Rotation des coordonnées relatives à l'orientation du keypoint
            cos_angle = np.cos(np.radians(-kp_orientation))
            sin_angle = np.sin(np.radians(-kp_orientation))
            
            # Coordonnées rotées
            rotated_x = cos_angle * j - sin_angle * i
            rotated_y = sin_angle * j + cos_angle * i
            
            # Déterminer la sous-région (0-3 pour x et y)
            subregion_x = int((rotated_x + half_size) / SUBREGION_SIZE)
            subregion_y = int((rotated_y + half_size) / SUBREGION_SIZE)
            
            # Vérifier les limites des sous-régions
            if (subregion_x < 0 or subregion_x >= NUM_SUBREGIONS or
                subregion_y < 0 or subregion_y >= NUM_SUBREGIONS):
                continue
            
            # Magnitude et orientation du gradient à cette position
            pixel_magnitude = magnitude[y, x]
            pixel_orientation = orientation[y, x]
            
            # Orientation relative (par rapport à l'orientation du keypoint)
            relative_orientation = pixel_orientation - kp_orientation
            if relative_orientation < 0:
                relative_orientation += 360
            
            # Bin d'orientation (0-7)
            orientation_bin = int(relative_orientation / (360.0 / NUM_BINS)) % NUM_BINS
            
            # Poids gaussien basé sur la distance au centre du keypoint
            distance_weight = np.exp(-(rotated_x**2 + rotated_y**2) / (2 * gaussian_sigma**2))
            
            # Contribution pondérée au descripteur
            weighted_magnitude = pixel_magnitude * distance_weight
            
            # Interpolation trilinéaire pour une meilleure précision
            # (Simplifiée ici - version basique)
            descriptor[subregion_y, subregion_x, orientation_bin] += weighted_magnitude
    
    # Aplatir le descripteur en vecteur 128D
    descriptor_vector = descriptor.flatten()
    
    # Normalisation L2
    norm = np.linalg.norm(descriptor_vector)
    if norm > 0:
        descriptor_vector = descriptor_vector / norm
    
    # Seuillage des valeurs importantes (> 0.2) pour réduire l'effet de l'éclairage
    descriptor_vector = np.minimum(descriptor_vector, 0.2)
    
    # Re-normalisation après seuillage
    norm = np.linalg.norm(descriptor_vector)
    if norm > 0:
        descriptor_vector = descriptor_vector / norm
    
    return descriptor_vector

def compute_all_descriptors(gaussian_pyramid, oriented_keypoints):
    """Calcule les descripteurs pour tous les keypoints orientés"""
    
    descriptors = []
    valid_keypoints = []
    
    for kp in oriented_keypoints:
        octave_idx = kp['octave']
        scale_idx = int(round(kp['scale']))
        
        # Vérifier les limites d'octave et d'échelle
        if (octave_idx < 0 or octave_idx >= len(gaussian_pyramid) or
            scale_idx < 0 or scale_idx >= len(gaussian_pyramid[octave_idx])):
            continue
        
        # Image gaussienne correspondante
        gaussian_image = gaussian_pyramid[octave_idx][scale_idx]
        
        # Calculer les gradients
        magnitude, orientation = compute_gradient_magnitude_and_orientation(gaussian_image)
        
        # Calculer le descripteur
        descriptor = compute_sift_descriptor(gaussian_image, kp, magnitude, orientation)
        
        if descriptor is not None:
            descriptors.append(descriptor)
            valid_keypoints.append(kp)
    
    return np.array(descriptors), valid_keypoints

def visualize_keypoint_and_descriptor(image, keypoint, descriptor, gaussian_image, magnitude, orientation):
    """Visualise un keypoint et son descripteur de manière détaillée"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Image originale avec le keypoint
    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray')
    
    # Dessiner le keypoint avec son orientation
    x, y = keypoint['y'], keypoint['x']  # Inversion pour matplotlib
    orientation_rad = np.radians(keypoint['orientation'])
    scale = keypoint['sigma']
    
    # Cercle représentant l'échelle
    circle = plt.Circle((x, y), scale*3, fill=False, color='red', linewidth=2)
    plt.gca().add_patch(circle)
    
    # Flèche d'orientation
    arrow_length = scale * 4
    dx = arrow_length * np.cos(orientation_rad)
    dy = arrow_length * np.sin(orientation_rad)
    plt.arrow(x, y, dx, dy, head_width=scale*1.5, head_length=scale*2, 
             fc='blue', ec='blue', linewidth=2)
    plt.plot(x, y, 'go', markersize=8)
    
    plt.title(f'Keypoint\nPos: ({keypoint["x"]:.1f}, {keypoint["y"]:.1f})\nOrient: {keypoint["orientation"]:.1f}°')
    plt.axis('off')
    
    # 2. Région 16x16 autour du keypoint
    plt.subplot(2, 4, 2)
    kp_x, kp_y = int(keypoint['local_x']), int(keypoint['local_y'])
    half_size = 8
    
    if (kp_x >= half_size and kp_x < gaussian_image.shape[1] - half_size and
        kp_y >= half_size and kp_y < gaussian_image.shape[0] - half_size):
        
        region = gaussian_image[kp_y-half_size:kp_y+half_size+1, 
                               kp_x-half_size:kp_x+half_size+1]
        plt.imshow(region, cmap='gray')
        
        # Grille 4x4 pour montrer les sous-régions
        for i in range(1, 4):
            plt.axhline(y=i*4-0.5, color='red', linewidth=1, alpha=0.7)
            plt.axvline(x=i*4-0.5, color='red', linewidth=1, alpha=0.7)
        
        plt.plot(half_size, half_size, 'go', markersize=8)
        plt.title('Région 16x16\n(Grille 4x4)')
    else:
        plt.text(0.5, 0.5, 'Région\nhors limites', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Région 16x16')
    plt.axis('off')
    
    # 3. Magnitude du gradient
    plt.subplot(2, 4, 3)
    if (kp_x >= half_size and kp_x < magnitude.shape[1] - half_size and
        kp_y >= half_size and kp_y < magnitude.shape[0] - half_size):
        
        mag_region = magnitude[kp_y-half_size:kp_y+half_size+1, 
                              kp_x-half_size:kp_x+half_size+1]
        plt.imshow(mag_region, cmap='hot')
        plt.colorbar(shrink=0.8)
        plt.plot(half_size, half_size, 'go', markersize=8)
    plt.title('Magnitude\ndu gradient')
    plt.axis('off')
    
    # 4. Orientation du gradient avec flèches
    plt.subplot(2, 4, 4)
    if (kp_x >= half_size and kp_x < orientation.shape[1] - half_size and
        kp_y >= half_size and kp_y < orientation.shape[0] - half_size):
        
        orient_region = orientation[kp_y-half_size:kp_y+half_size+1, 
                                   kp_x-half_size:kp_x+half_size+1]
        mag_region = magnitude[kp_y-half_size:kp_y+half_size+1, 
                              kp_x-half_size:kp_x+half_size+1]
        
        plt.imshow(region, cmap='gray', alpha=0.5)
        
        # Dessiner les flèches de gradient (sous-échantillonnage pour la lisibilité)
        step = 2
        for i in range(0, orient_region.shape[0], step):
            for j in range(0, orient_region.shape[1], step):
                if mag_region[i, j] > 0.01:  # Seuil pour éviter le bruit
                    angle = np.radians(orient_region[i, j])
                    length = mag_region[i, j] * 3
                    dx = length * np.cos(angle)
                    dy = length * np.sin(angle)
                    plt.arrow(j, i, dx, dy, head_width=0.5, head_length=0.5, 
                             fc='red', ec='red', alpha=0.8)
        
        plt.plot(half_size, half_size, 'go', markersize=8)
    plt.title('Orientations\ndu gradient')
    plt.axis('off')
    
    # 5-8. Descripteur sous forme de grille 4x4
    desc_reshaped = descriptor.reshape(4, 4, 8)
    
    for idx in range(4):
        plt.subplot(2, 4, 5 + idx)
        
        # Créer une visualisation en étoile pour chaque ligne de sous-régions
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        for j in range(4):
            values = desc_reshaped[idx, j, :]
            
            # Coordonnées polaires
            x_center, y_center = j, 0
            
            # Dessiner l'histogramme en étoile
            for k, (angle, value) in enumerate(zip(angles, values)):
                x_end = x_center + value * 50 * np.cos(angle)
                y_end = y_center + value * 50 * np.sin(angle)
                plt.plot([x_center, x_end], [y_center, y_end], 'b-', linewidth=2, alpha=0.7)
                plt.plot(x_end, y_end, 'ro', markersize=4)
            
            plt.plot(x_center, y_center, 'ko', markersize=6)
            plt.text(x_center, y_center-0.3, f'({idx},{j})', ha='center', fontsize=8)
        
        plt.xlim(-0.5, 3.5)
        plt.ylim(-0.5, 0.5)
        plt.title(f'Ligne {idx} du descripteur\n(4 sous-régions)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Analyse complète du keypoint SIFT\nDescripteur 128D (norme L2: {np.linalg.norm(descriptor):.4f})', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Affichage numérique détaillé
    print(f"\n=== Analyse détaillée du keypoint ===")
    print(f"Position: ({keypoint['x']:.1f}, {keypoint['y']:.1f})")
    print(f"Échelle (sigma): {keypoint['sigma']:.2f}")
    print(f"Orientation: {keypoint['orientation']:.1f}°")
    print(f"Octave: {keypoint['octave']}, Scale: {keypoint['scale']:.1f}")
    
    print(f"\n=== Propriétés du descripteur ===")
    print(f"Dimension: {len(descriptor)}")
    print(f"Norme L2: {np.linalg.norm(descriptor):.6f}")
    print(f"Somme: {np.sum(descriptor):.6f}")
    print(f"Min: {np.min(descriptor):.6f}, Max: {np.max(descriptor):.6f}")
    print(f"Moyenne: {np.mean(descriptor):.6f}, Std: {np.std(descriptor):.6f}")
    
    # Afficher quelques valeurs du descripteur par sous-région
    print(f"\n=== Valeurs par sous-région (premières 4 sous-régions) ===")
    for i in range(2):
        for j in range(2):
            subregion_values = desc_reshaped[i, j, :]
            print(f"Sous-région ({i},{j}): [{', '.join([f'{v:.4f}' for v in subregion_values])}]")
    
    return desc_reshaped

def analyze_descriptor_properties(descriptors):
    """Analyse les propriétés statistiques des descripteurs"""
    
    if len(descriptors) == 0:
        print("Aucun descripteur à analyser")
        return
    
    print(f"=== Analyse des descripteurs ===")
    print(f"Nombre de descripteurs: {len(descriptors)}")
    print(f"Dimension: {descriptors.shape[1]}")
    
    # Statistiques globales
    mean_values = np.mean(descriptors, axis=0)
    std_values = np.std(descriptors, axis=0)
    
    print(f"Valeur moyenne: {np.mean(mean_values):.4f}")
    print(f"Écart-type moyen: {np.mean(std_values):.4f}")
    print(f"Valeur max: {np.max(descriptors):.4f}")
    print(f"Valeur min: {np.min(descriptors):.4f}")
    
    # Vérifier la normalisation L2
    norms = np.linalg.norm(descriptors, axis=1)
    print(f"Normes L2 - Min: {np.min(norms):.4f}, Max: {np.max(norms):.4f}, Moyenne: {np.mean(norms):.4f}")
    
    # Analyse de la distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(descriptors.flatten(), bins=50, alpha=0.7, color='lightblue')
    plt.title('Distribution des valeurs')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')
    
    plt.subplot(1, 3, 2)
    plt.plot(mean_values)
    plt.title('Profil moyen du descripteur')
    plt.xlabel('Dimension')
    plt.ylabel('Valeur moyenne')
    
    plt.subplot(1, 3, 3)
    plt.hist(norms, bins=20, alpha=0.7, color='lightgreen')
    plt.title('Distribution des normes L2')
    plt.xlabel('Norme L2')
    plt.ylabel('Fréquence')
    
    plt.tight_layout()
    plt.show()

def load_sample_oriented_keypoints():
    """Génère quelques keypoints orientés d'exemple"""
    sample_keypoints = [
        {
            'octave': 0, 'scale': 2, 'x': 100, 'y': 100, 
            'local_x': 100, 'local_y': 100, 'sigma': 3.2, 
            'orientation': 94.7, 'value': 0.05
        },
        {
            'octave': 0, 'scale': 2, 'x': 200, 'y': 150, 
            'local_x': 200, 'local_y': 150, 'sigma': 3.2, 
            'orientation': 163.3, 'value': -0.04
        },
        {
            'octave': 0, 'scale': 3, 'x': 300, 'y': 200, 
            'local_x': 300, 'local_y': 200, 'sigma': 5.1, 
            'orientation': 0.8, 'value': 0.06
        }
    ]
    return sample_keypoints

def main():
    # Charger une image de test
    image = data.camera()
    print(f"Image chargée: {image.shape}")
    
    # Normaliser l'image
    image_float = image.astype(np.float64) / 255.0
    
    # Construire la pyramide gaussienne
    print("Construction de la pyramide gaussienne...")
    gaussian_pyramid = build_gaussian_pyramid(image_float, num_octaves=3, scales_per_octave=3)
    
    # Charger des keypoints orientés d'exemple
    print("Chargement des keypoints orientés...")
    oriented_keypoints = load_sample_oriented_keypoints()
    
    print(f"Keypoints orientés: {len(oriented_keypoints)}")
    
    # Étape 4: Calcul des descripteurs SIFT
    print("Calcul des descripteurs SIFT...")
    descriptors, valid_keypoints = compute_all_descriptors(gaussian_pyramid, oriented_keypoints)
    
    print(f"Descripteurs calculés: {len(descriptors)}")
    
    if len(descriptors) > 0:
        print(f"Dimension des descripteurs: {descriptors.shape[1]}")
        
        # Afficher quelques exemples de descripteurs
        print(f"\n=== Exemples de descripteurs ===")
        for i, (desc, kp) in enumerate(zip(descriptors[:3], valid_keypoints[:3])):
            print(f"Descripteur {i+1}: Position ({kp['x']:.1f}, {kp['y']:.1f})")
            print(f"  Norme L2: {np.linalg.norm(desc):.4f}")
            print(f"  Valeurs [0:5]: {desc[:5]}")
            print(f"  Valeurs [-5:]: {desc[-5:]}")
        
        # Visualiser le premier keypoint en détail
        if len(valid_keypoints) > 0:
            print(f"\nVisualisation détaillée du premier keypoint...")
            
            # Récupérer les données nécessaires pour la visualisation
            kp = valid_keypoints[0]
            octave_idx = kp['octave']
            scale_idx = int(round(kp['scale']))
            
            if (octave_idx < len(gaussian_pyramid) and 
                scale_idx < len(gaussian_pyramid[octave_idx])):
                
                gaussian_img = gaussian_pyramid[octave_idx][scale_idx]
                magnitude, orientation = compute_gradient_magnitude_and_orientation(gaussian_img)
                
                # Visualisation complète
                desc_reshaped = visualize_keypoint_and_descriptor(
                    image, kp, descriptors[0], gaussian_img, magnitude, orientation
                )
        
        # Analyse des propriétés des descripteurs
        print(f"\n" + "="*50)
        analyze_descriptor_properties(descriptors)
        
        # Calculer la similarité entre descripteurs
        if len(descriptors) > 1:
            print(f"\n=== Analyse de similarité ===")
            for i in range(len(descriptors)):
                for j in range(i+1, len(descriptors)):
                    # Distance euclidienne
                    euclidean_dist = np.linalg.norm(descriptors[i] - descriptors[j])
                    # Similarité cosinus
                    cosine_sim = np.dot(descriptors[i], descriptors[j])
                    
                    print(f"Keypoints {i+1}-{j+1}: "
                          f"Distance euclidienne: {euclidean_dist:.4f}, "
                          f"Similarité cosinus: {cosine_sim:.4f}")
    
    else:
        print("Aucun descripteur n'a pu être calculé. Vérifiez les positions des keypoints.")

if __name__ == "__main__":
    main()