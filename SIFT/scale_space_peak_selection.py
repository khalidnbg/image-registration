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
    # Sigma initial
    k = 2**(1/scales_per_octave)
    sigma = 1.6
    
    pyramid = []
    
    for octave in range(num_octaves):
        # Redimensionner l'image pour cette octave
        if octave == 0:
            current_image = image.copy()
        else:
            # Sous-échantillonner par 2
            current_image = ndimage.zoom(pyramid[octave-1][-3], 0.5)
        
        octave_images = []
        
        for scale in range(scales_per_octave + 3):  # +3 pour les images additionnelles nécessaires au DoG
            current_sigma = sigma * (k ** scale)
            
            # Appliquer le filtre gaussien
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
            # Différence entre échelles gaussiennes consécutives
            dog = octave[i+1] - octave[i]
            dog_octave.append(dog)
        dog_pyramid.append(dog_octave)
    
    return dog_pyramid

def is_extremum(dog_pyramid, octave_idx, scale_idx, x, y):
    """Vérifie si un pixel est un extremum local dans l'espace échelle"""
    current = dog_pyramid[octave_idx][scale_idx]
    
    # Vérifier les limites
    if (x <= 0 or x >= current.shape[0] - 1 or 
        y <= 0 or y >= current.shape[1] - 1 or
        scale_idx <= 0 or scale_idx >= len(dog_pyramid[octave_idx]) - 1):
        return False
    
    center_value = current[x, y]
    
    # Vérifier dans le voisinage 3x3x3 (x, y, échelle)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for ds in [-1, 0, 1]:
                if dx == 0 and dy == 0 and ds == 0:
                    continue
                
                neighbor_scale = scale_idx + ds
                if neighbor_scale < 0 or neighbor_scale >= len(dog_pyramid[octave_idx]):
                    continue
                
                neighbor_value = dog_pyramid[octave_idx][neighbor_scale][x + dx, y + dy]
                
                # Si ce n'est ni un maximum ni un minimum local, ce n'est pas un extremum
                if center_value >= 0:  # Test pour maximum
                    if neighbor_value >= center_value:
                        return False
                else:  # Test pour minimum
                    if neighbor_value <= center_value:
                        return False
    
    return True

def detect_extrema(dog_pyramid, contrast_threshold=0.03):
    """Détecte tous les extrema dans la pyramide DoG"""
    extrema = []
    
    for octave_idx, octave in enumerate(dog_pyramid):
        for scale_idx in range(1, len(octave) - 1):  # Éviter les échelles extrêmes
            dog_image = octave[scale_idx]
            
            # Parcourir chaque pixel
            for x in range(1, dog_image.shape[0] - 1):
                for y in range(1, dog_image.shape[1] - 1):
                    
                    # Filtrage par seuil de contraste
                    if abs(dog_image[x, y]) < contrast_threshold:
                        continue
                    
                    # Vérifier si c'est un extremum
                    if is_extremum(dog_pyramid, octave_idx, scale_idx, x, y):
                        # Calculer l'échelle effective
                        k = 2**(1/3)  # scales_per_octave = 3
                        sigma_base = 1.6
                        effective_sigma = sigma_base * (k ** scale_idx) * (2 ** octave_idx)
                        
                        extrema.append({
                            'octave': octave_idx,
                            'scale': scale_idx,
                            'x': x * (2 ** octave_idx),  # Coordonnées dans l'image originale
                            'y': y * (2 ** octave_idx),
                            'sigma': effective_sigma,
                            'value': dog_image[x, y],
                            'response': abs(dog_image[x, y])
                        })
    
    return extrema

# Exemple principal avec une image de scikit-learn
def main():
    # Charger une image de scikit-learn (convertir en niveaux de gris)
    image = data.camera()  # ou data.coins(), data.checkerboard(), etc.
    
    print(f"Image chargée: {image.shape}")
    print(f"Type: {image.dtype}, Min: {image.min()}, Max: {image.max()}")
    
    # Normaliser l'image
    image = image.astype(np.float64) / 255.0
    
    # Étape 1: Construire la pyramide gaussienne
    print("Construction de la pyramide gaussienne...")
    gaussian_pyramid = build_gaussian_pyramid(image, num_octaves=3, scales_per_octave=3)
    
    # Étape 2: Calculer la pyramide DoG
    print("Calcul de la pyramide DoG...")
    dog_pyramid = compute_dog_pyramid(gaussian_pyramid)
    
    # Étape 3: Détecter les extrema
    print("Détection des extrema...")
    extrema = detect_extrema(dog_pyramid, contrast_threshold=0.01)
    
    print(f"Nombre d'extrema détectés: {len(extrema)}")
    
    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Image originale
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Image originale')
    axes[0, 0].axis('off')
    
    # Quelques images de la pyramide gaussienne
    for i, ax in enumerate(axes[0, 1:]):
        if i < len(gaussian_pyramid[0]):
            ax.imshow(gaussian_pyramid[0][i], cmap='gray')
            ax.set_title(f'Gaussienne échelle {i}')
        ax.axis('off')
    
    # Quelques images DoG
    for i, ax in enumerate(axes[1, :3]):
        if i < len(dog_pyramid[0]):
            ax.imshow(dog_pyramid[0][i], cmap='RdBu_r', vmin=-0.1, vmax=0.1)
            ax.set_title(f'DoG échelle {i}')
        ax.axis('off')
    
    # Affichage des extrema sur l'image originale
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    
    # Afficher les extrema (seulement pour la première octave pour la clarté)
    octave_0_extrema = [e for e in extrema if e['octave'] == 0]
    
    # Séparer maxima et minima
    maxima = [e for e in octave_0_extrema if e['value'] > 0]
    minima = [e for e in octave_0_extrema if e['value'] < 0]
    
    if maxima:
        max_x = [e['y']/(2**e['octave']) for e in maxima]  # Ajustement pour les coordonnées
        max_y = [e['x']/(2**e['octave']) for e in maxima]
        plt.plot(max_x, max_y, 'ro', markersize=3, label=f'Maxima ({len(maxima)})')
    
    if minima:
        min_x = [e['y']/(2**e['octave']) for e in minima]
        min_y = [e['x']/(2**e['octave']) for e in minima]
        plt.plot(min_x, min_y, 'bo', markersize=3, label=f'Minima ({len(minima)})')
    
    plt.title('Extrema détectés (Octave 0)')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Statistiques des extrema
    print(f"\nStatistiques des extrema:")
    print(f"Total: {len(extrema)}")
    print(f"Maxima: {len([e for e in extrema if e['value'] > 0])}")
    print(f"Minima: {len([e for e in extrema if e['value'] < 0])}")
    
    # Distribution par octave
    octave_counts = {}
    for e in extrema:
        octave_counts[e['octave']] = octave_counts.get(e['octave'], 0) + 1
    
    print("Distribution par octave:")
    for octave, count in sorted(octave_counts.items()):
        print(f"  Octave {octave}: {count} extrema")

if __name__ == "__main__":
    main()