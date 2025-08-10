import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time

class ImageRegistration:
    """
    Classe pour le recalage d'images avec deux méthodes de corrélation
    """
    
    def __init__(self):
        pass
    
    def load_images(self, img1_path, img2_path):
        """
        Charge deux images depuis des fichiers
        """
        try:
            # Charger les images
            img1 = plt.imread(img1_path)
            img2 = plt.imread(img2_path)
            
            # Convertir en niveaux de gris si nécessaire
            if len(img1.shape) == 3:
                img1 = np.mean(img1, axis=2)
            if len(img2.shape) == 3:
                img2 = np.mean(img2, axis=2)
            
            # Normaliser entre 0 et 1
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            
            if img1.max() > 1:
                img1 = img1 / 255.0
            if img2.max() > 1:
                img2 = img2 / 255.0
            
            return img1, img2
            
        except Exception as e:
            print(f"Erreur lors du chargement des images: {e}")
            return None, None
    
    def load_images_from_arrays(self, img1, img2):
        """
        Utilise des arrays NumPy comme images d'entrée
        """
        # Convertir en niveaux de gris si nécessaire
        if len(img1.shape) == 3:
            img1 = np.mean(img1, axis=2)
        if len(img2.shape) == 3:
            img2 = np.mean(img2, axis=2)
        
        # Normaliser entre 0 et 1
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        if img1.max() > 1:
            img1 = img1 / 255.0
        if img2.max() > 1:
            img2 = img2 / 255.0
        
        return img1, img2
    
    def create_test_images(self, size=256):
        """
        Crée des images de test avec une translation connue (pour les tests)
        """
        # Image de référence (cercles concentriques)
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Créer des cercles concentriques
        reference = np.zeros((size, size))
        for r in np.linspace(0.2, 0.8, 4):
            circle = (X**2 + Y**2) < r**2
            reference += circle.astype(float) * 0.3
        
        # Ajouter du bruit gaussien
        reference += np.random.normal(0, 0.1, (size, size))
        
        # Translation connue
        dx_true, dy_true = 15, -8  # pixels
        
        # Image déplacée
        shifted = ndimage.shift(reference, (dy_true, dx_true), mode='constant', cval=0)
        
        return reference, shifted, (dx_true, dy_true)
    
    def correlation_fft(self, img1, img2):
        """
        Corrélation croisée classique avec FFT
        """
        start_time = time.time()
        
        # Transformée de Fourier des images
        F1 = np.fft.fft2(img1)
        F2 = np.fft.fft2(img2)
        
        # Corrélation croisée dans le domaine fréquentiel
        # Conjugué de F2 pour la corrélation croisée
        cross_correlation = F1 * np.conj(F2)
        
        # Retour dans le domaine spatial
        correlation_map = np.fft.ifft2(cross_correlation)
        correlation_map = np.abs(correlation_map)
        
        # Trouver le pic de corrélation
        peak_pos = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        
        # Convertir en décalage (gestion du wrapping)
        h, w = img1.shape
        dy = peak_pos[0] if peak_pos[0] <= h//2 else peak_pos[0] - h
        dx = peak_pos[1] if peak_pos[1] <= w//2 else peak_pos[1] - w
        
        execution_time = time.time() - start_time
        
        return dx, dy, correlation_map, execution_time
    
    def phase_correlation(self, img1, img2):
        """
        Corrélation de phase pour une meilleure précision sub-pixellique
        """
        start_time = time.time()
        
        # Transformée de Fourier des images
        F1 = np.fft.fft2(img1)
        F2 = np.fft.fft2(img2)
        
        # Corrélation de phase normalisée
        cross_power_spectrum = F1 * np.conj(F2)
        
        # Normalisation par l'amplitude (phase seulement)
        cross_power_spectrum = cross_power_spectrum / (np.abs(cross_power_spectrum) + 1e-8)
        
        # Transformée inverse pour obtenir la fonction de corrélation de phase
        phase_correlation_map = np.fft.ifft2(cross_power_spectrum)
        phase_correlation_map = np.abs(phase_correlation_map)
        
        # Trouver le pic avec précision sub-pixellique
        peak_pos = np.unravel_index(np.argmax(phase_correlation_map), phase_correlation_map.shape)
        
        # Convertir en décalage
        h, w = img1.shape
        dy = peak_pos[0] if peak_pos[0] <= h//2 else peak_pos[0] - h
        dx = peak_pos[1] if peak_pos[1] <= w//2 else peak_pos[1] - w
        
        execution_time = time.time() - start_time
        
        return dx, dy, phase_correlation_map, execution_time
    
    def compare_methods(self, img1, img2, true_translation=None):
        """
        Compare les deux méthodes de corrélation
        """
        print("=== COMPARAISON DES MÉTHODES DE CORRÉLATION ===\n")
        
        # Méthode 1: Corrélation FFT classique
        dx_fft, dy_fft, corr_map_fft, time_fft = self.correlation_fft(img1, img2)
        
        # Méthode 2: Corrélation de phase
        dx_phase, dy_phase, corr_map_phase, time_phase = self.phase_correlation(img1, img2)
        
        # Affichage des résultats
        print("1. CORRÉLATION FFT CLASSIQUE:")
        print(f"   Translation estimée: dx = {dx_fft:.2f}, dy = {dy_fft:.2f}")
        print(f"   Temps d'exécution: {time_fft:.4f} secondes")
        
        print("\n2. CORRÉLATION DE PHASE:")
        print(f"   Translation estimée: dx = {dx_phase:.2f}, dy = {dy_phase:.2f}")
        print(f"   Temps d'exécution: {time_phase:.4f} secondes")
        
        if true_translation:
            dx_true, dy_true = true_translation
            error_fft = np.sqrt((dx_fft - dx_true)**2 + (dy_fft - dy_true)**2)
            error_phase = np.sqrt((dx_phase - dx_true)**2 + (dy_phase - dy_true)**2)
            
            print(f"\n3. COMPARAISON AVEC LA VÉRITÉ TERRAIN:")
            print(f"   Translation vraie: dx = {dx_true}, dy = {dy_true}")
            print(f"   Erreur FFT: {error_fft:.2f} pixels")
            print(f"   Erreur Phase: {error_phase:.2f} pixels")
        
        return {
            'fft': {'dx': dx_fft, 'dy': dy_fft, 'time': time_fft, 'map': corr_map_fft},
            'phase': {'dx': dx_phase, 'dy': dy_phase, 'time': time_phase, 'map': corr_map_phase}
        }
    
    def visualize_results(self, img1, img2, results, true_translation=None):
        """
        Visualise les résultats des deux méthodes
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Images originales
        axes[0, 0].imshow(img1, cmap='gray')
        axes[0, 0].set_title('Image de référence')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2, cmap='gray')
        axes[0, 1].set_title('Image déplacée')
        axes[0, 1].axis('off')
        
        # Carte de corrélation FFT
        axes[0, 2].imshow(np.fft.fftshift(results['fft']['map']), cmap='hot')
        axes[0, 2].set_title(f'Corrélation FFT\n(dx={results["fft"]["dx"]:.1f}, dy={results["fft"]["dy"]:.1f})')
        axes[0, 2].axis('off')
        
        # Carte de corrélation de phase
        axes[1, 0].imshow(np.fft.fftshift(results['phase']['map']), cmap='hot')
        axes[1, 0].set_title(f'Corrélation de phase\n(dx={results["phase"]["dx"]:.1f}, dy={results["phase"]["dy"]:.1f})')
        axes[1, 0].axis('off')
        
        # Graphique de comparaison des performances
        methods = ['FFT', 'Phase']
        times = [results['fft']['time'], results['phase']['time']]
        
        axes[1, 1].bar(methods, times, color=['blue', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('Temps (secondes)')
        axes[1, 1].set_title('Temps d\'exécution')
        
        # Graphique de précision si vérité terrain disponible
        if true_translation:
            dx_true, dy_true = true_translation
            errors = [
                np.sqrt((results['fft']['dx'] - dx_true)**2 + (results['fft']['dy'] - dy_true)**2),
                np.sqrt((results['phase']['dx'] - dx_true)**2 + (results['phase']['dy'] - dy_true)**2)
            ]
            
            axes[1, 2].bar(methods, errors, color=['blue', 'red'], alpha=0.7)
            axes[1, 2].set_ylabel('Erreur (pixels)')
            axes[1, 2].set_title('Précision de l\'estimation')
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

# Exemple d'utilisation avec vos propres images
if __name__ == "__main__":
    # Créer une instance de la classe
    reg = ImageRegistration()
    
    # OPTION 1: Charger vos images depuis des fichiers
    print("=== UTILISATION AVEC VOS IMAGES ===")
    print("Chargement des images...")
    
    # Remplacez ces chemins par vos vrais chemins d'images
    img1_path = "images/tisdrin.jpg"  # Image de référence
    img2_path = "images/tisdrin_bright.png"  # Image à recaler
    
    # Charger les images
    img1, img2 = reg.load_images(img1_path, img2_path)
    
    # Vérifier si le chargement a réussi
    if img1 is not None and img2 is not None:
        print(f"Images chargées avec succès:")
        print(f"- Image 1: {img1.shape}")
        print(f"- Image 2: {img2.shape}")
        
        # Comparer les méthodes (sans vérité terrain)
        results = reg.compare_methods(img1, img2)
        
        # Visualiser les résultats
        reg.visualize_results(img1, img2, results)
        
    else:
        print("Erreur: Impossible de charger les images.")
        print("Utilisation d'images de test à la place...")
        
        # OPTION 2: Fallback avec images de test
        img_ref, img_shifted, true_translation = reg.create_test_images(size=128)
        results = reg.compare_methods(img_ref, img_shifted, true_translation)
        reg.visualize_results(img_ref, img_shifted, results, true_translation)
    
    # OPTION 3: Si vous avez déjà vos images en arrays NumPy
    """
    # Exemple d'utilisation avec des arrays NumPy existants:
    # img1, img2 = reg.load_images_from_arrays(your_image1_array, your_image2_array)
    # results = reg.compare_methods(img1, img2)
    # reg.visualize_results(img1, img2, results)
    """
    
    print("\n=== COMMENT UTILISER AVEC VOS IMAGES ===")
    print("1. Remplacez 'image1.jpg' et 'image2.jpg' par vos vrais chemins")
    print("2. Ou utilisez load_images_from_arrays() si vous avez des arrays NumPy")
    print("3. Formats supportés: JPG, PNG, TIFF, BMP, etc.")
    
    print("\n=== AVANTAGES ET INCONVÉNIENTS ===")
    print("\nCorrélation FFT classique:")
    print("+ Simple à implémenter")
    print("+ Robuste au bruit")
    print("- Précision limitée (pixel entier)")
    print("- Sensible aux variations d'illumination")
    
    print("\nCorrélation de phase:")
    print("+ Précision sub-pixellique possible")
    print("+ Moins sensible aux variations d'illumination")
    print("+ Meilleure performance avec des objets bien contrastés")
    print("- Plus sensible au bruit")
    print("- Peut être instable avec des images peu texturées")