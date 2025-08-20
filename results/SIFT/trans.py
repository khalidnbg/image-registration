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
    
    def register_images(self, ref_image, mov_image):
        """Recalage complet de deux images"""
        
        # 1. Détection des features
        kp_ref, desc_ref = self.detect_features(ref_image)
        kp_mov, desc_mov = self.detect_features(mov_image)
        
        if desc_ref is None or desc_mov is None:
            return None, None
        
        # 2. Appariement des features
        matches = self.match_features(desc_mov, desc_ref)
        
        if len(matches) < 4:
            return None, None
        
        # 3. Estimation de l'homographie
        H, mask = self.estimate_homography(kp_mov, kp_ref, matches)
        
        if H is None:
            return None, None
        
        inliers = np.sum(mask) if mask is not None else 0
        
        # Informations de retour
        info = {
            'homography': H,
            'matches_total': len(matches),
            'inliers': inliers,
            'inlier_ratio': inliers/len(matches) if len(matches) > 0 else 0
        }
        
        return None, info  # Pas besoin de l'image recalée pour les résultats

def create_translated_image(original, translation):
    """Création d'une image avec translation"""
    tx, ty = translation
    
    # Matrice de transformation
    tform = transform.AffineTransform(translation=(tx, ty))
    
    # Application de la transformation
    transformed = transform.warp(original, tform.inverse, output_shape=original.shape)
    
    true_params = {
        'angle': 0,
        'scale': 1.0,
        'tx': tx,
        'ty': ty
    }
    
    return transformed, true_params

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

def test_translations():
    """Test avec différentes translations"""
    
    # Image originale
    original = cv2.imread('results/brain.jpg')

    # Translations à tester
    translations = [(10, 5), (-15, 8), (20, -10), (-25, -15), (30, 12)]
    
    # Initialisation du système de recalage
    registrator = SimpleImageRegistration()
    
    print("=== SIFT + RANSAC - Test des Translations ===\n")
    print(f"{'Translation':<15} {'Réel TX':<10} {'Réel TY':<10} {'Est. TX':<10} {'Est. TY':<10} {'Erreur TX':<12} {'Erreur TY':<12} {'Inliers':<8}")
    print("-" * 90)
    
    results = []
    
    for i, translation in enumerate(translations):
        # Création de l'image transformée
        mov_image, true_params = create_translated_image(original, translation)
        
        # Recalage
        _, info = registrator.register_images(original, mov_image)
        
        if info is not None:
            # Estimation des paramètres
            estimated_params = decompose_homography(info['homography'])
            
            if estimated_params is not None:
                # Calcul des erreurs
                error_tx = abs(true_params['tx'] - estimated_params['tx'])
                error_ty = abs(true_params['ty'] - estimated_params['ty'])
                
                # Stockage des résultats
                results.append({
                    'translation': translation,
                    'true_tx': true_params['tx'],
                    'true_ty': true_params['ty'],
                    'est_tx': estimated_params['tx'],
                    'est_ty': estimated_params['ty'],
                    'error_tx': error_tx,
                    'error_ty': error_ty,
                    'inliers': info['inliers'],
                    'total_matches': info['matches_total']
                })
                
                # Affichage des résultats
                print(f"{str(translation):<15} {true_params['tx']:<10.1f} {true_params['ty']:<10.1f} "
                      f"{estimated_params['tx']:<10.1f} {estimated_params['ty']:<10.1f} "
                      f"{error_tx:<12.2f} {error_ty:<12.2f} {info['inliers']:<8}")
            else:
                print(f"{str(translation):<15} {'ÉCHEC - Impossible de décomposer l\'homographie'}")
        else:
            print(f"{str(translation):<15} {'ÉCHEC - Recalage impossible'}")
    
    # Résumé statistique
    if results:
        print("\n=== Résumé Statistique ===")
        errors_tx = [r['error_tx'] for r in results]
        errors_ty = [r['error_ty'] for r in results]
        
        print(f"Erreur moyenne TX: {np.mean(errors_tx):.2f} ± {np.std(errors_tx):.2f} pixels")
        print(f"Erreur moyenne TY: {np.mean(errors_ty):.2f} ± {np.std(errors_ty):.2f} pixels")
        print(f"Erreur max TX: {np.max(errors_tx):.2f} pixels")
        print(f"Erreur max TY: {np.max(errors_ty):.2f} pixels")
        print(f"Nombre moyen d'inliers: {np.mean([r['inliers'] for r in results]):.0f}")
    
    # Visualisation simple
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graphique 1: Comparaison TX
        translations_labels = [str(r['translation']) for r in results]
        true_tx = [r['true_tx'] for r in results]
        est_tx = [r['est_tx'] for r in results]
        
        x = np.arange(len(results))
        width = 0.35
        
        ax1.bar(x - width/2, true_tx, width, label='Réel TX', alpha=0.8, color='blue')
        ax1.bar(x + width/2, est_tx, width, label='Estimé TX', alpha=0.8, color='red')
        ax1.set_xlabel('Translations')
        ax1.set_ylabel('Translation X (pixels)')
        ax1.set_title('Comparaison Translation X')
        ax1.set_xticks(x)
        ax1.set_xticklabels(translations_labels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Comparaison TY
        true_ty = [r['true_ty'] for r in results]
        est_ty = [r['est_ty'] for r in results]
        
        ax2.bar(x - width/2, true_ty, width, label='Réel TY', alpha=0.8, color='green')
        ax2.bar(x + width/2, est_ty, width, label='Estimé TY', alpha=0.8, color='orange')
        ax2.set_xlabel('Translations')
        ax2.set_ylabel('Translation Y (pixels)')
        ax2.set_title('Comparaison Translation Y')
        ax2.set_xticks(x)
        ax2.set_xticklabels(translations_labels, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_translations()