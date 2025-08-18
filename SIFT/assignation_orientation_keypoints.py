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

def compute_gradient_magnitude_and_orientation(image):
    """Calcule la magnitude et l'orientation du gradient pour chaque pixel"""
    # Calculer les gradients en x et y
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)
    
    # Gradients par différences finies (éviter les bords)
    dx[1:-1, 1:-1] = (image[1:-1, 2:] - image[1:-1, :-2]) / 2.0
    dy[1:-1, 1:-1] = (image[2:, 1:-1] - image[:-2, 1:-1]) / 2.0
    
    # Magnitude du gradient
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Orientation du gradient (en radians)
    orientation = np.arctan2(dy, dx)
    
    # Convertir en degrés [0, 360)
    orientation = np.degrees(orientation)
    orientation[orientation < 0] += 360
    
    return magnitude, orientation

def create_orientation_histogram(magnitude, orientation, keypoint_x, keypoint_y, 
                                scale, region_radius, num_bins=36):
    """Crée un histogramme d'orientations pondéré par une gaussienne"""
    
    # Initialiser l'histogramme (36 bins pour 360°)
    histogram = np.zeros(num_bins)
    bin_width = 360.0 / num_bins  # 10° par bin
    
    # Paramètres de la fenêtre gaussienne
    gaussian_sigma = 1.5 * scale
    
    # Parcourir la région autour du keypoint
    for dy in range(-region_radius, region_radius + 1):
        for dx in range(-region_radius, region_radius + 1):
            y = int(keypoint_y + dy)
            x = int(keypoint_x + dx)
            
            # Vérifier les limites de l'image
            if (x <= 0 or x >= magnitude.shape[1] - 1 or 
                y <= 0 or y >= magnitude.shape[0] - 1):
                continue
            
            # Distance au centre du keypoint
            distance = np.sqrt(dx**2 + dy**2)
            
            # Ignorer les pixels hors du rayon circulaire
            if distance > region_radius:
                continue
            
            # Poids gaussien
            weight = np.exp(-(dx**2 + dy**2) / (2 * gaussian_sigma**2))
            
            # Contribution pondérée à l'histogramme
            pixel_magnitude = magnitude[y, x]
            pixel_orientation = orientation[y, x]
            
            # Trouver le bin correspondant
            bin_index = int(pixel_orientation / bin_width) % num_bins
            
            # Ajouter la contribution pondérée
            histogram[bin_index] += pixel_magnitude * weight
    
    return histogram

def find_orientation_peaks(histogram, peak_threshold=0.8):
    """Trouve les pics d'orientation dans l'histogramme"""
    
    # Lisser l'histogramme (convolution circulaire)
    # Filtre simple : [1, 1, 1] normalisé
    smoothed = np.convolve(np.concatenate([histogram[-1:], histogram, histogram[:1]]), 
                          [1/3, 1/3, 1/3], mode='valid')
    
    # Trouver le maximum global
    max_value = np.max(smoothed)
    threshold = peak_threshold * max_value
    
    orientations = []
    
    # Chercher tous les pics au-dessus du seuil
    for i in range(len(smoothed)):
        # Vérifier si c'est un maximum local
        prev_idx = (i - 1) % len(smoothed)
        next_idx = (i + 1) % len(smoothed)
        
        if (smoothed[i] > smoothed[prev_idx] and 
            smoothed[i] > smoothed[next_idx] and 
            smoothed[i] >= threshold):
            
            # Interpolation parabolique pour plus de précision
            refined_orientation = refine_orientation_peak(smoothed, i)
            orientations.append(refined_orientation)
    
    return orientations

def refine_orientation_peak(histogram, peak_index):
    """Affine la position du pic par interpolation parabolique"""
    n = len(histogram)
    
    # Indices des voisins (gestion circulaire)
    prev_idx = (peak_index - 1) % n
    next_idx = (peak_index + 1) % n
    
    # Valeurs de l'histogramme
    prev_val = histogram[prev_idx]
    curr_val = histogram[peak_index]
    next_val = histogram[next_idx]
    
    # Interpolation parabolique
    numerator = prev_val - next_val
    denominator = 2 * (prev_val - 2*curr_val + next_val)
    
    if abs(denominator) < 1e-10:
        offset = 0
    else:
        offset = numerator / denominator
    
    # Position raffinée
    refined_peak = peak_index + offset
    
    # Convertir en degrés
    bin_width = 360.0 / n
    orientation = (refined_peak * bin_width) % 360
    
    return orientation

def assign_keypoint_orientations(gaussian_pyramid, keypoints):
    """Assigne les orientations aux keypoints"""
    
    oriented_keypoints = []
    
    for kp in keypoints:
        octave_idx = kp['octave']
        scale_idx = int(round(kp['scale']))
        
        # Coordonnées locales dans l'octave
        local_x = kp['local_x']
        local_y = kp['local_y']
        
        # Vérifier les limites
        if (scale_idx < 0 or scale_idx >= len(gaussian_pyramid[octave_idx]) or
            local_x < 3 or local_y < 3):
            continue
        
        # Image gaussienne correspondante
        gaussian_image = gaussian_pyramid[octave_idx][scale_idx]
        
        if (local_x >= gaussian_image.shape[1] - 3 or 
            local_y >= gaussian_image.shape[0] - 3):
            continue
        
        # Calculer magnitude et orientation du gradient
        magnitude, orientation = compute_gradient_magnitude_and_orientation(gaussian_image)
        
        # Rayon de la région d'analyse (proportionnel à l'échelle)
        region_radius = int(round(3 * kp['sigma']))
        region_radius = max(region_radius, 3)  # Minimum 3 pixels
        
        # Créer l'histogramme d'orientations
        histogram = create_orientation_histogram(
            magnitude, orientation, local_x, local_y, 
            kp['sigma'], region_radius
        )
        
        # Trouver les orientations dominantes
        dominant_orientations = find_orientation_peaks(histogram)
        
        # Créer un keypoint pour chaque orientation dominante
        for orient in dominant_orientations:
            oriented_kp = kp.copy()
            oriented_kp['orientation'] = orient
            oriented_kp['histogram'] = histogram.copy()  # Pour la visualisation
            oriented_keypoints.append(oriented_kp)
    
    return oriented_keypoints

def visualize_keypoints_with_orientations(image, keypoints, title="Keypoints avec orientations"):
    """Visualise les keypoints avec leurs orientations"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap='gray')
    
    for kp in keypoints:
        x, y = kp['y'], kp['x']  # Note: inversion pour matplotlib
        orientation = kp['orientation']
        scale = kp.get('sigma', 3)
        
        # Dessiner le cercle représentant l'échelle
        circle = plt.Circle((x, y), scale*2, fill=False, color='red', linewidth=1.5, alpha=0.7)
        plt.gca().add_patch(circle)
        
        # Dessiner la flèche d'orientation
        arrow_length = scale * 3
        dx = arrow_length * np.cos(np.radians(orientation))
        dy = arrow_length * np.sin(np.radians(orientation))
        
        plt.arrow(x, y, dx, dy, head_width=scale*0.8, head_length=scale*1.2, 
                 fc='blue', ec='blue', linewidth=2, alpha=0.8)
        
        # Point central
        plt.plot(x, y, 'go', markersize=3)
    
    plt.title(f'{title} ({len(keypoints)} keypoints)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_orientation_histogram(keypoint, title="Histogramme d'orientations"):
    """Visualise l'histogramme d'orientations d'un keypoint"""
    if 'histogram' not in keypoint:
        print("Pas d'histogramme disponible pour ce keypoint")
        return
    
    histogram = keypoint['histogram']
    orientations = np.arange(0, 360, 360/len(histogram))
    
    plt.figure(figsize=(10, 6))
    plt.bar(orientations, histogram, width=360/len(histogram), alpha=0.7, 
            color='skyblue', edgecolor='black')
    
    # Marquer l'orientation assignée
    assigned_orientation = keypoint['orientation']
    max_height = np.max(histogram)
    plt.axvline(x=assigned_orientation, color='red', linestyle='--', linewidth=2, 
                label=f'Orientation assignée: {assigned_orientation:.1f}°')
    
    plt.xlabel('Orientation (degrés)')
    plt.ylabel('Magnitude pondérée')
    plt.title(title)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 45))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation avec keypoints de l'étape 2
def load_sample_keypoints():
    """Génère quelques keypoints d'exemple pour tester"""
    # Ces keypoints seraient normalement issus de l'étape 2
    sample_keypoints = [
        {'octave': 0, 'scale': 2, 'x': 100, 'y': 100, 'local_x': 100, 'local_y': 100, 'sigma': 3.2, 'value': 0.05},
        {'octave': 0, 'scale': 2, 'x': 200, 'y': 150, 'local_x': 200, 'local_y': 150, 'sigma': 3.2, 'value': -0.04},
        {'octave': 0, 'scale': 3, 'x': 300, 'y': 200, 'local_x': 300, 'local_y': 200, 'sigma': 5.1, 'value': 0.06},
        {'octave': 1, 'scale': 1, 'x': 150, 'y': 250, 'local_x': 75, 'local_y': 125, 'sigma': 2.5, 'value': 0.03},
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
    
    # Utiliser des keypoints d'exemple (en pratique, ils viennent de l'étape 2)
    print("Chargement des keypoints d'exemple...")
    keypoints = load_sample_keypoints()
    
    print(f"Keypoints d'entrée: {len(keypoints)}")
    
    # Étape 3: Assignation d'orientations
    print("Assignation des orientations...")
    oriented_keypoints = assign_keypoint_orientations(gaussian_pyramid, keypoints)
    
    print(f"Keypoints avec orientations: {len(oriented_keypoints)}")
    
    # Statistiques sur les orientations multiples
    original_count = len(keypoints)
    oriented_count = len(oriented_keypoints)
    multiple_orientations = oriented_count - original_count
    
    print(f"\n=== Statistiques d'orientation ===")
    print(f"Keypoints originaux: {original_count}")
    print(f"Keypoints avec orientations: {oriented_count}")
    print(f"Orientations multiples générées: {multiple_orientations}")
    print(f"Ratio d'expansion: {oriented_count/original_count:.2f}")
    
    # Afficher quelques orientations
    print(f"\n=== Exemples d'orientations assignées ===")
    for i, kp in enumerate(oriented_keypoints[:5]):
        print(f"Keypoint {i+1}: Position ({kp['x']:.1f}, {kp['y']:.1f}), "
              f"Échelle: {kp['sigma']:.2f}, Orientation: {kp['orientation']:.1f}°")
    
    # Visualisation
    print("\nVisualisation des keypoints avec orientations...")
    visualize_keypoints_with_orientations(image, oriented_keypoints)
    
    # Visualiser l'histogramme d'un keypoint spécifique
    if oriented_keypoints:
        print(f"\nHistogramme d'orientation du premier keypoint:")
        visualize_orientation_histogram(oriented_keypoints[0])

if __name__ == "__main__":
    main()