import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data, transform

class SimpleImageRegistration:
    """
    Classe simplifiée pour le recalage d'images par SIFT + RANSAC
    """
    
    def __init__(self):
        # Initialisation du détecteur SIFT
        self.sift = cv2.SIFT_create()
        self.lowe_ratio = 0.75  # Ratio de Lowe pour filtrer les correspondances
    
    def detect_features(self, image):
        """Détection et description des features SIFT"""
        # Conversion en uint8 si nécessaire
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            img_uint8 = image
            
        keypoints, descriptors = self.sift.detectAndCompute(img_uint8, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Appariement des features avec test du ratio de Lowe"""
        # Matcher FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Appariement k=2 plus proches voisins
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Test du ratio de Lowe
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.lowe_ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_homography(self, kp1, kp2, matches):
        """Estimation de l'homographie avec RANSAC"""
        if len(matches) < 4:
            return None, None
        
        # Extraction des points correspondants
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimation RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return H, mask
    
    def apply_transformation(self, image, H, output_shape):
        """Application de la transformation homographique"""
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            img_uint8 = image
        
        # Transformation de l'image
        transformed = cv2.warpPerspective(img_uint8, H, (output_shape[1], output_shape[0]))
        
        # Reconversion si nécessaire
        if image.max() <= 1.0:
            transformed = transformed.astype(np.float64) / 255.0
            
        return transformed
    
    def register_images(self, ref_image, mov_image):
        """
        Recalage complet de deux images
        
        Args:
            ref_image: Image de référence (fixe)
            mov_image: Image mobile à recaler
            
        Returns:
            registered_image: Image recalée
            info: Informations sur le recalage
        """
        print("Détection des features...")
        
        # 1. Détection des features
        kp_ref, desc_ref = self.detect_features(ref_image)
        kp_mov, desc_mov = self.detect_features(mov_image)
        
        if desc_ref is None or desc_mov is None:
            print("Erreur: Aucune feature détectée")
            return None, None
        
        print(f"Features détectées - Référence: {len(kp_ref)}, Mobile: {len(kp_mov)}")
        
        # 2. Appariement des features
        print("Appariement des features...")
        matches = self.match_features(desc_mov, desc_ref)
        
        if len(matches) < 4:
            print("Erreur: Pas assez de correspondances")
            return None, None
        
        print(f"Correspondances trouvées: {len(matches)}")
        
        # 3. Estimation de l'homographie
        print("Estimation de l'homographie avec RANSAC...")
        H, mask = self.estimate_homography(kp_mov, kp_ref, matches)
        
        if H is None:
            print("Erreur: Impossible d'estimer l'homographie")
            return None, None
        
        inliers = np.sum(mask) if mask is not None else 0
        print(f"Inliers: {inliers}/{len(matches)} ({inliers/len(matches)*100:.1f}%)")
        
        # 4. Application de la transformation
        print("Application de la transformation...")
        registered_image = self.apply_transformation(mov_image, H, ref_image.shape)
        
        # Informations de retour
        info = {
            'homography': H,
            'matches_total': len(matches),
            'inliers': inliers,
            'inlier_ratio': inliers/len(matches) if len(matches) > 0 else 0,
            'keypoints_ref': kp_ref,
            'keypoints_mov': kp_mov,
            'matches': matches,
            'mask': mask,
            'descriptors_ref': desc_ref,
            'descriptors_mov': desc_mov
        }
        
        return registered_image, info

def create_test_images():
    """Création d'images de test"""
    # Image originale
    original = data.camera().astype(np.float64) / 255.0
    
    # Paramètres de transformation connus
    true_angle = 20  # degrés
    true_scale = 0.9
    true_tx = 30
    true_ty = 20
    
    # Transformation : rotation + échelle + translation
    tform = transform.SimilarityTransform(
        rotation=np.radians(true_angle),
        scale=true_scale,
        translation=(true_tx, true_ty)
    )
    
    # Application de la transformation
    transformed = transform.warp(original, tform.inverse, output_shape=original.shape)
    
    true_params = {
        'angle': true_angle,
        'scale': true_scale,
        'tx': true_tx,
        'ty': true_ty
    }
    
    return original, transformed, true_params

def decompose_homography(H):
    """Décomposition de l'homographie pour extraire les paramètres de transformation"""
    if H is None:
        return None
    
    # Calcul de l'homographie inverse (référence → mobile)
    H_inv = np.linalg.inv(H)
    
    # Extraction des paramètres de la transformation
    a, b = H_inv[0,0], H_inv[0,1]
    c, d = H_inv[1,0], H_inv[1,1]
    tx, ty = H_inv[0,2], H_inv[1,2]
    
    # Facteurs d'échelle
    scale_x = np.sqrt(a**2 + b**2)
    scale_y = np.sqrt(c**2 + d**2)
    scale_avg = (scale_x + scale_y) / 2
    
    # Rotation
    theta_rad = np.arctan2(b, a)
    theta_deg = np.degrees(theta_rad)
    
    return {
        'angle': abs(theta_deg),
        'scale': scale_avg,
        'tx': tx,
        'ty': ty
    }

def plot_keypoint_orientations(keypoints, title):
    """Affichage de l'histogramme des orientations des keypoints"""
    orientations = [kp.angle for kp in keypoints]
    
    plt.figure(figsize=(10, 6))
    
    # Histogramme des orientations
    plt.subplot(1, 2, 1)
    plt.hist(orientations, bins=36, range=(0, 360), alpha=0.7, edgecolor='black')
    plt.xlabel('Orientation (degrés)')
    plt.ylabel('Nombre de keypoints')
    plt.title(f'Distribution des orientations\n{title}')
    plt.grid(True, alpha=0.3)
    
    # Histogramme polaire
    plt.subplot(1, 2, 2, projection='polar')
    theta = np.radians(orientations)
    plt.hist(theta, bins=36, alpha=0.7)
    plt.title(f'Orientations (polaire)\n{title}')
    
    plt.tight_layout()

def plot_sift_descriptor(descriptor, keypoint_info=""):
    """Visualisation d'un descripteur SIFT (128 dimensions)"""
    plt.figure(figsize=(12, 8))
    
    # Visualisation 1D du descripteur
    plt.subplot(2, 2, 1)
    plt.plot(descriptor, 'b-', linewidth=2)
    plt.xlabel('Dimension')
    plt.ylabel('Valeur')
    plt.title(f'Descripteur SIFT 128D {keypoint_info}')
    plt.grid(True, alpha=0.3)
    
    # Visualisation 2D du descripteur (4x4 grille de 8 orientations)
    plt.subplot(2, 2, 2)
    desc_2d = descriptor.reshape(4, 4, 8).sum(axis=2)  # Somme sur les 8 orientations
    plt.imshow(desc_2d, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Descripteur 2D (4x4 régions)')
    
    # Histogramme des valeurs du descripteur
    plt.subplot(2, 2, 3)
    plt.hist(descriptor, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')
    plt.title('Distribution des valeurs')
    plt.grid(True, alpha=0.3)
    
    # Visualisation des 8 orientations pour chaque région 4x4
    plt.subplot(2, 2, 4)
    desc_orientations = descriptor.reshape(16, 8)  # 16 régions x 8 orientations
    plt.imshow(desc_orientations.T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Région (4x4)')
    plt.ylabel('Orientation')
    plt.title('Histogrammes d\'orientations par région')
    
    plt.tight_layout()

def visualize_results(ref_image, mov_image, reg_image, info, true_params=None):
    """Visualisation complète des résultats"""
    
    # Figure 1: Images et différences
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle('Recalage d\'images avec SIFT + RANSAC', fontsize=16)
    
    # Images
    axes[0,0].imshow(ref_image, cmap='gray')
    axes[0,0].set_title('Image de référence')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(mov_image, cmap='gray')
    axes[0,1].set_title('Image mobile')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(reg_image, cmap='gray')
    axes[0,2].set_title('Image recalée')
    axes[0,2].axis('off')
    
    # Différences
    diff_before = np.abs(ref_image - mov_image)
    diff_after = np.abs(ref_image - reg_image)
    
    axes[1,0].imshow(diff_before, cmap='hot')
    axes[1,0].set_title('Différence avant recalage')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(diff_after, cmap='hot')
    axes[1,1].set_title('Différence après recalage')
    axes[1,1].axis('off')
    
    # Superposition
    overlay = np.zeros((ref_image.shape[0], ref_image.shape[1], 3))
    overlay[:,:,0] = ref_image
    overlay[:,:,1] = reg_image
    axes[1,2].imshow(overlay)
    axes[1,2].set_title('Superposition (R: réf, V: recalée)')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    # Figure 2: Comparaison paramètres vrais vs estimés
    if true_params is not None:
        estimated_params = decompose_homography(info['homography'])
        
        if estimated_params is not None:
            fig2, ax = plt.subplots(figsize=(12, 6))
            
            params = ['Rotation (°)', 'Échelle', 'Translation X', 'Translation Y']
            true_vals = [true_params['angle'], true_params['scale'], 
                        true_params['tx'], true_params['ty']]
            est_vals = [estimated_params['angle'], estimated_params['scale'],
                       estimated_params['tx'], estimated_params['ty']]
            
            x = np.arange(len(params))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, true_vals, width, label='Valeurs réelles', 
                          alpha=0.8, color='blue')
            bars2 = ax.bar(x + width/2, est_vals, width, label='Valeurs estimées', 
                          alpha=0.8, color='red')
            
            ax.set_xlabel('Paramètres')
            ax.set_ylabel('Valeurs')
            ax.set_title('Comparaison Paramètres Réels vs Estimés')
            ax.set_xticks(x)
            ax.set_xticklabels(params)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Ajout des valeurs sur les barres
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Affichage des erreurs
            errors = {
                'rotation': abs(true_params['angle'] - estimated_params['angle']),
                'scale': abs(true_params['scale'] - estimated_params['scale']),
                'tx': abs(true_params['tx'] - estimated_params['tx']),
                'ty': abs(true_params['ty'] - estimated_params['ty'])
            }
            
            print(f"\n📊 Erreurs d'estimation:")
            print(f"- Rotation: {errors['rotation']:.2f}°")
            print(f"- Échelle: {errors['scale']:.4f}")
            print(f"- Translation X: {errors['tx']:.1f} pixels")
            print(f"- Translation Y: {errors['ty']:.1f} pixels")
    
    # Figure 3: Histogrammes d'orientations des keypoints
    plot_keypoint_orientations(info['keypoints_ref'], 'Image de référence')
    plot_keypoint_orientations(info['keypoints_mov'], 'Image mobile')
    
    # Figure 4: Visualisation d'un descripteur SIFT
    if len(info['keypoints_ref']) > 0:
        # Prendre le keypoint le plus fort (plus grande réponse)
        responses = [kp.response for kp in info['keypoints_ref']]
        best_idx = np.argmax(responses)
        best_keypoint = info['keypoints_ref'][best_idx]
        best_descriptor = info['descriptors_ref'][best_idx]
        
        keypoint_info = f"(x={best_keypoint.pt[0]:.0f}, y={best_keypoint.pt[1]:.0f}, réponse={best_keypoint.response:.3f})"
        plot_sift_descriptor(best_descriptor, keypoint_info)
    
    plt.show()

def main():
    """Fonction principale"""
    print("=== Recalage d'images avec SIFT + RANSAC ===\n")
    
    # 1. Création des images de test
    print("Création des images de test...")
    ref_image, mov_image, true_params = create_test_images()
    print(f"Taille des images: {ref_image.shape}")
    print(f"Paramètres réels: Rotation={true_params['angle']}°, Échelle={true_params['scale']}, Translation=({true_params['tx']}, {true_params['ty']})\n")
    
    # 2. Initialisation du système de recalage
    registrator = SimpleImageRegistration()
    
    # 3. Recalage
    registered_image, info = registrator.register_images(ref_image, mov_image)
    
    if registered_image is not None:
        print("\n✅ Recalage réussi!")
        
        # 4. Estimation des paramètres
        estimated_params = decompose_homography(info['homography'])
        if estimated_params is not None:
            print(f"\nParamètres estimés:")
            print(f"- Rotation: {estimated_params['angle']:.2f}°")
            print(f"- Échelle: {estimated_params['scale']:.4f}")
            print(f"- Translation: ({estimated_params['tx']:.1f}, {estimated_params['ty']:.1f})")
        
        # 5. Évaluation simple
        mse_before = np.mean((ref_image - mov_image)**2)
        mse_after = np.mean((ref_image - registered_image)**2)
        improvement = (mse_before - mse_after) / mse_before * 100
        
        print(f"\nRésultats:")
        print(f"- MSE avant recalage: {mse_before:.6f}")
        print(f"- MSE après recalage: {mse_after:.6f}")
        print(f"- Amélioration: {improvement:.1f}%")
        
        # 6. Visualisation complète
        visualize_results(ref_image, mov_image, registered_image, info, true_params)
        
    else:
        print("❌ Échec du recalage")

if __name__ == "__main__":
    main()