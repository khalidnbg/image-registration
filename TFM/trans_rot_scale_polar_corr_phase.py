import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.ndimage import rotate, zoom, shift

class RobustTransformDetector:
    def __init__(self):
        self.debug = True
    
    def preprocess_image(self, img):
        """Prétraitement simple et efficace"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Conversion en float et normalisation
        img = img.astype(np.float64)
        img = (img - np.mean(img)) / (np.std(img) + 1e-10)
        
        return img
    
    def phase_correlation(self, img1, img2):
        """Corrélation de phase standard"""
        # FFT
        F1 = np.fft.fft2(img1)
        F2 = np.fft.fft2(img2)
        
        # Cross power spectrum
        cross_power = F1 * np.conj(F2)
        
        # Normalisation
        magnitude = np.abs(cross_power)
        phase_corr = cross_power / (magnitude + 1e-10)
        
        # IFFT pour obtenir la corrélation
        correlation = np.abs(np.fft.ifft2(phase_corr))
        
        return correlation
    
    def find_translation_peak(self, correlation):
        """Trouve le pic de translation avec correction périodique"""
        peak_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
        dy, dx = peak_pos
        h, w = correlation.shape
        
        # Correction périodique
        if dy > h // 2:
            dy = dy - h
        if dx > w // 2:
            dx = dx - w
        
        return dy, dx
    
    def get_log_polar_coords(self, shape, center=None):
        """Génère les coordonnées pour la transformation log-polaire"""
        h, w = shape
        if center is None:
            center = (h//2, w//2)
        
        cy, cx = center
        max_radius = min(cy, cx, h-cy, w-cx) - 1
        
        # Créer les grilles log-polaires
        log_r = np.linspace(0, np.log(max_radius), h)
        theta = np.linspace(0, 2*np.pi, w, endpoint=False)
        
        LOG_R, THETA = np.meshgrid(log_r, theta, indexing='ij')
        R = np.exp(LOG_R)
        
        # Coordonnées cartésiennes
        Y = cy + R * np.sin(THETA)
        X = cx + R * np.cos(THETA)
        
        return Y, X, max_radius
    
    def transform_to_log_polar(self, img):
        """Transformation log-polaire avec interpolation"""
        Y, X, max_radius = self.get_log_polar_coords(img.shape)
        
        # Interpolation avec gestion des bords
        h, w = img.shape
        valid_mask = (X >= 0) & (X < w-1) & (Y >= 0) & (Y < h-1)
        
        log_polar = np.zeros_like(Y)
        
        # Interpolation bilinéaire manuelle pour plus de contrôle
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if valid_mask[i, j]:
                    y, x = Y[i, j], X[i, j]
                    
                    # Indices entiers
                    y1, x1 = int(y), int(x)
                    y2, x2 = y1 + 1, x1 + 1
                    
                    if y2 < h and x2 < w:
                        # Poids d'interpolation
                        wy = y - y1
                        wx = x - x1
                        
                        # Interpolation bilinéaire
                        log_polar[i, j] = (img[y1, x1] * (1-wy) * (1-wx) +
                                          img[y1, x2] * (1-wy) * wx +
                                          img[y2, x1] * wy * (1-wx) +
                                          img[y2, x2] * wy * wx)
        
        return log_polar, max_radius
    
    def detect_rotation_scale_fourier(self, img1, img2):
        """Détection rotation/échelle via spectres de magnitude et log-polaire"""
        if self.debug:
            print("  Calcul des spectres de magnitude...")
        
        # Prétraitement
        img1_prep = self.preprocess_image(img1)
        img2_prep = self.preprocess_image(img2)
        
        # Spectres de magnitude (invariants à la translation)
        fft1 = np.fft.fftshift(np.fft.fft2(img1_prep))
        fft2 = np.fft.fftshift(np.fft.fft2(img2_prep))
        
        mag1 = np.abs(fft1)
        mag2 = np.abs(fft2)
        
        # Appliquer un filtre passe-haut pour éliminer la composante DC
        h, w = mag1.shape
        cy, cx = h//2, w//2
        y, x = np.ogrid[:h, :w]
        mask = ((y - cy)**2 + (x - cx)**2) > (min(h, w)/20)**2
        
        mag1 = mag1 * mask
        mag2 = mag2 * mask
        
        # Compression logarithmique
        mag1 = np.log(mag1 + 1)
        mag2 = np.log(mag2 + 1)
        
        if self.debug:
            print("  Transformation log-polaire...")
        
        # Transformation log-polaire
        lp1, max_r = self.transform_to_log_polar(mag1)
        lp2, _ = self.transform_to_log_polar(mag2)
        
        if self.debug:
            print("  Corrélation de phase en log-polaire...")
        
        # Corrélation de phase
        correlation = self.phase_correlation(lp1, lp2)
        
        # Trouver le pic
        delta_log_r, delta_theta = self.find_translation_peak(correlation)
        
        if self.debug:
            print(f"  Delta log_r: {delta_log_r}, Delta theta: {delta_theta}")
        
        # Conversion en paramètres physiques
        # Échelle : exp(delta_log_r * facteur_échelle)
        log_r_range = np.log(max_r)
        scale_factor = np.exp(-delta_log_r * log_r_range / lp1.shape[0])
        
        # Rotation : delta_theta * 2π / largeur
        rotation_angle = -delta_theta * 2 * np.pi / lp1.shape[1]
        
        return scale_factor, rotation_angle, correlation
    
    def detect_translation_direct(self, img1, img2):
        """Détection directe de translation par corrélation de phase"""
        if self.debug:
            print("  Corrélation de phase pour translation...")
        
        # Prétraitement
        img1_prep = self.preprocess_image(img1)
        img2_prep = self.preprocess_image(img2)
        
        # Corrélation de phase
        correlation = self.phase_correlation(img1_prep, img2_prep)
        
        # Trouver la translation
        dy, dx = self.find_translation_peak(correlation)
        
        return (dy, dx), correlation
    
    def apply_inverse_transform(self, img, scale=1.0, rotation=0.0, translation=(0, 0)):
        """Applique la transformation géométrique inverse"""
        h, w = img.shape[:2]
        center = (w//2, h//2)
        
        # Matrice de transformation inverse
        M = cv2.getRotationMatrix2D(center, -np.degrees(rotation), 1.0/scale)
        M[0, 2] -= translation[1]  # compensation dx
        M[1, 2] -= translation[0]  # compensation dy
        
        # Application
        result = cv2.warpAffine(img.astype(np.float32), M, (w, h), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    def refine_parameters_iteratively(self, img1, img2, initial_scale, initial_rotation):
        """Raffinement itératif des paramètres par optimisation"""
        
        def objective_function(params):
            scale, rotation = params
            corrected = self.apply_inverse_transform(img2, scale, rotation)
            
            # Calculer la corrélation normalisée
            img1_norm = self.preprocess_image(img1)
            corrected_norm = self.preprocess_image(corrected)
            
            # Corrélation croisée normalisée
            correlation = cv2.matchTemplate(img1_norm.astype(np.float32), 
                                          corrected_norm.astype(np.float32), 
                                          cv2.TM_CCOEFF_NORMED)
            
            return -np.max(correlation)  # Minimiser l'inverse de la corrélation
        
        # Recherche locale autour des valeurs initiales
        best_scale = initial_scale
        best_rotation = initial_rotation
        best_score = float('inf')
        
        # Grille de recherche fine
        scale_range = np.linspace(initial_scale * 0.9, initial_scale * 1.1, 21)
        rotation_range = np.linspace(initial_rotation - np.radians(5), 
                                   initial_rotation + np.radians(5), 21)
        
        for scale in scale_range:
            for rotation in rotation_range:
                score = objective_function([scale, rotation])
                if score < best_score:
                    best_score = score
                    best_scale = scale
                    best_rotation = rotation
        
        return best_scale, best_rotation
    
    def detect_all_parameters(self, img1, img2):
        """Détection complète avec raffinement"""
        print("=== DÉTECTION ROBUSTE DES TRANSFORMATIONS ===")
        
        # Étape 1: Détection initiale rotation/échelle
        print("\n1. Détection initiale rotation/échelle (log-polaire):")
        scale_init, rotation_init, rot_corr = self.detect_rotation_scale_fourier(img1, img2)
        
        print(f"   Échelle initiale: {scale_init:.4f}")
        print(f"   Rotation initiale: {np.degrees(rotation_init):.2f}°")
        
        # Étape 2: Raffinement par optimisation
        print("\n2. Raffinement des paramètres:")
        scale_refined, rotation_refined = self.refine_parameters_iteratively(
            img1, img2, scale_init, rotation_init)
        
        print(f"   Échelle raffinée: {scale_refined:.4f}")
        print(f"   Rotation raffinée: {np.degrees(rotation_refined):.2f}°")
        
        # Étape 3: Correction et détection de translation
        print("\n3. Correction et détection de translation:")
        img2_corrected = self.apply_inverse_transform(img2, scale_refined, rotation_refined)
        
        translation, trans_corr = self.detect_translation_direct(img1, img2_corrected)
        
        print(f"   Translation: dx={translation[1]:.2f}, dy={translation[0]:.2f}")
        
        # Étape 4: Correction finale
        final_corrected = self.apply_inverse_transform(img2_corrected, 1.0, 0.0, translation)
        
        return {
            'scale': scale_refined,
            'rotation': rotation_refined,
            'translation': translation,
            'initial_estimates': (scale_init, rotation_init),
            'rotation_scale_correlation': rot_corr,
            'translation_correlation': trans_corr,
            'intermediate_correction': img2_corrected,
            'final_correction': final_corrected
        }

def create_test_with_known_transform():
    """Crée des images de test avec transformation connue"""
    # Image de base avec motifs distincts
    img1 = np.zeros((300, 300), dtype=np.uint8)
    
    # Motifs géométriques bien définis
    # Rectangle principal
    cv2.rectangle(img1, (100, 100), (200, 200), 255, -1)
    cv2.rectangle(img1, (120, 120), (180, 180), 0, -1)
    
    # Cercles aux coins
    cv2.circle(img1, (70, 70), 30, 200, -1)
    cv2.circle(img1, (230, 70), 25, 150, -1)
    cv2.circle(img1, (70, 230), 25, 150, -1)
    cv2.circle(img1, (230, 230), 30, 100, -1)
    
    # Triangles
    pts1 = np.array([[50, 150], [90, 150], [70, 120]], np.int32)
    cv2.fillPoly(img1, [pts1], 180)
    
    pts2 = np.array([[210, 150], [250, 150], [230, 180]], np.int32)
    cv2.fillPoly(img1, [pts2], 120)
    
    # Lignes directionnelles
    cv2.line(img1, (50, 50), (100, 100), 160, 3)
    cv2.line(img1, (200, 200), (250, 250), 140, 3)
    
    # Paramètres de transformation exacts
    true_params = {
        'scale': 1.2,
        'rotation': 18,  # degrés
        'translation': (15, 7)  # dx, dy
    }
    
    # Application de la transformation
    h, w = img1.shape
    center = (w//2, h//2)
    
    M = cv2.getRotationMatrix2D(center, true_params['rotation'], true_params['scale'])
    M[0, 2] += true_params['translation'][0]  # dx
    M[1, 2] += true_params['translation'][1]  # dy
    
    img2 = cv2.warpAffine(img1, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return img1, img2, true_params

def calculate_errors(detected, true_params):
    """Calcule les erreurs de détection"""
    scale_error = abs(detected['scale'] - true_params['scale']) / true_params['scale'] * 100
    rotation_error = abs(np.degrees(detected['rotation']) - true_params['rotation'])
    
    detected_trans = (detected['translation'][1], detected['translation'][0])  # dx, dy
    true_trans = true_params['translation']
    
    translation_error = np.sqrt((detected_trans[0] - true_trans[0])**2 + 
                               (detected_trans[1] - true_trans[1])**2)
    
    return scale_error, rotation_error, translation_error

def main():
    """Test principal avec images de référence"""
    detector = RobustTransformDetector()
    
    print("Création d'images de test avec transformation connue...")
    img1, img2, true_params = create_test_with_known_transform()
    
    print(f"\n📍 PARAMÈTRES DE RÉFÉRENCE:")
    print(f"   Échelle: {true_params['scale']}")
    print(f"   Rotation: {true_params['rotation']}°")
    print(f"   Translation: dx={true_params['translation'][0]}, dy={true_params['translation'][1]}")
    
    # Détection
    results = detector.detect_all_parameters(img1, img2)
    
    # Affichage des résultats
    print(f"\n{'='*60}")
    print("📊 RÉSULTATS DE DÉTECTION:")
    print(f"{'='*60}")
    
    print(f"Échelle    | Détectée: {results['scale']:7.4f} | Vraie: {true_params['scale']:7.4f}")
    print(f"Rotation   | Détectée: {np.degrees(results['rotation']):7.2f}° | Vraie: {true_params['rotation']:7.2f}°")
    print(f"Translation| Détectée: dx={results['translation'][1]:5.1f}, dy={results['translation'][0]:5.1f} | Vraie: dx={true_params['translation'][0]:5.1f}, dy={true_params['translation'][1]:5.1f}")
    
    # Calcul et affichage des erreurs
    scale_error, rotation_error, translation_error = calculate_errors(results, true_params)
    
    print(f"\n🎯 PRÉCISION:")
    print(f"   Erreur d'échelle: {scale_error:.2f}%")
    print(f"   Erreur de rotation: {rotation_error:.2f}°")
    print(f"   Erreur de translation: {translation_error:.2f} pixels")
    
    # Évaluation de la qualité
    overall_quality = "Excellente" if (scale_error < 2 and rotation_error < 1 and translation_error < 2) else \
                     "Bonne" if (scale_error < 5 and rotation_error < 3 and translation_error < 5) else \
                     "Correcte" if (scale_error < 10 and rotation_error < 5 and translation_error < 10) else "À améliorer"
    
    print(f"   Qualité globale: {overall_quality}")
    
    # Visualisation complète
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Ligne 1: Images principales
    axes[0,0].imshow(img1, cmap='gray')
    axes[0,0].set_title('Image 1 (Référence)', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img2, cmap='gray')
    axes[0,1].set_title(f'Image 2 Transformée\n(S:{true_params["scale"]}, R:{true_params["rotation"]}°, T:{true_params["translation"]})', fontsize=10)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(results['intermediate_correction'], cmap='gray')
    axes[0,2].set_title('Après correction R/S', fontsize=12)
    axes[0,2].axis('off')
    
    axes[0,3].imshow(results['final_correction'], cmap='gray')
    axes[0,3].set_title('Correction finale', fontsize=12, fontweight='bold')
    axes[0,3].axis('off')
    
    # Ligne 2: Analyses de corrélation
    axes[1,0].imshow(np.log(results['rotation_scale_correlation'] + 1), cmap='jet')
    axes[1,0].set_title('Corrélation Log-Polaire', fontsize=12)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(np.log(results['translation_correlation'] + 1), cmap='jet')
    axes[1,1].set_title('Corrélation Translation', fontsize=12)
    axes[1,1].axis('off')
    
    # Différence finale
    diff_final = np.abs(img1.astype(float) - results['final_correction'].astype(float))
    im_diff = axes[1,2].imshow(diff_final, cmap='hot')
    axes[1,2].set_title(f'Différence finale\n(MSE: {np.mean(diff_final**2):.2f})', fontsize=12)
    axes[1,2].axis('off')
    plt.colorbar(im_diff, ax=axes[1,2], fraction=0.046, pad=0.04)
    
    # Superposition
    overlay = np.zeros((img1.shape[0], img1.shape[1], 3))
    overlay[:,:,0] = img1 / 255.0  # Rouge pour référence
    overlay[:,:,1] = results['final_correction'] / 255.0  # Vert pour correction
    axes[1,3].imshow(overlay)
    axes[1,3].set_title('Superposition\n(Rouge: Réf, Vert: Corrigée)', fontsize=12)
    axes[1,3].axis('off')
    
    # Ligne 3: Graphiques d'analyse
    # Erreurs par paramètre
    errors = [scale_error, rotation_error, translation_error]
    labels = ['Échelle\n(%)', 'Rotation\n(°)', 'Translation\n(pixels)']
    colors = ['green' if e < 2 else 'orange' if e < 5 else 'red' for e in errors]
    
    bars = axes[2,0].bar(labels, errors, color=colors, alpha=0.7)
    axes[2,0].set_title('Erreurs de détection', fontsize=12, fontweight='bold')
    axes[2,0].set_ylabel('Erreur')
    axes[2,0].grid(True, alpha=0.3)
    
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        axes[2,0].text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.02,
                      f'{error:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Évolution des paramètres (initial vs raffiné)
    param_names = ['Échelle', 'Rotation (°)']
    initial_vals = [results['initial_estimates'][0], np.degrees(results['initial_estimates'][1])]
    refined_vals = [results['scale'], np.degrees(results['rotation'])]
    true_vals = [true_params['scale'], true_params['rotation']]
    
    x_pos = np.arange(len(param_names))
    width = 0.25
    
    axes[2,1].bar(x_pos - width, initial_vals, width, label='Initial', alpha=0.7, color='lightblue')
    axes[2,1].bar(x_pos, refined_vals, width, label='Raffiné', alpha=0.7, color='blue')
    axes[2,1].bar(x_pos + width, true_vals, width, label='Vrai', alpha=0.7, color='green')
    
    axes[2,1].set_xlabel('Paramètres')
    axes[2,1].set_ylabel('Valeurs')
    axes[2,1].set_title('Évolution des paramètres', fontsize=12, fontweight='bold')
    axes[2,1].set_xticks(x_pos)
    axes[2,1].set_xticklabels(param_names)
    axes[2,1].legend()
    axes[2,1].grid(True, alpha=0.3)
    
    # Histogramme de qualité
    quality_scores = [100 - scale_error, 100 - min(rotation_error * 10, 100), 100 - min(translation_error * 10, 100)]
    quality_labels = ['Échelle', 'Rotation', 'Translation']
    quality_colors = ['green' if s > 90 else 'orange' if s > 70 else 'red' for s in quality_scores]
    
    bars_quality = axes[2,2].bar(quality_labels, quality_scores, color=quality_colors, alpha=0.7)
    axes[2,2].set_title('Score de qualité (%)', fontsize=12, fontweight='bold')
    axes[2,2].set_ylabel('Score (%)')
    axes[2,2].set_ylim(0, 100)
    axes[2,2].grid(True, alpha=0.3)
    
    for bar, score in zip(bars_quality, quality_scores):
        height = bar.get_height()
        axes[2,2].text(bar.get_x() + bar.get_width()/2., height + 2,
                      f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Résumé textuel
    axes[2,3].text(0.1, 0.9, '📋 RÉSUMÉ DES RÉSULTATS', fontsize=14, fontweight='bold', transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.7, f'Qualité globale: {overall_quality}', fontsize=12, transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.6, f'Erreur d\'échelle: {scale_error:.2f}%', fontsize=11, transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.5, f'Erreur de rotation: {rotation_error:.2f}°', fontsize=11, transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.4, f'Erreur de translation: {translation_error:.2f}px', fontsize=11, transform=axes[2,3].transAxes)
    axes[2,3].text(0.1, 0.2, f'MSE finale: {np.mean(diff_final**2):.2f}', fontsize=11, transform=axes[2,3].transAxes)
    axes[2,3].set_xlim(0, 1)
    axes[2,3].set_ylim(0, 1)
    axes[2,3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()