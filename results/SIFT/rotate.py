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
            img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.ast(np.uint8)
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
        """
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
        
        # 4. Application de la transformation
        registered_image = self.apply_transformation(mov_image, H, ref_image.shape)
        
        # Informations de retour
        info = {
            'homography': H,
            'matches_total': len(matches),
            'inliers': np.sum(mask) if mask is not None else 0,
            'inlier_ratio': np.sum(mask)/len(matches) if len(matches) > 0 else 0,
        }
        
        return registered_image, info

def create_rotated_image(image, angle_degrees):
    """
    Crée une image tournée avec rotation centrée correcte
    """
    # Centre de l'image
    center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    
    # Matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Appliquer la rotation sans translation supplémentaire
    img_rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return img_rotated

def decompose_homography_simple(H):
    """
    Décomposition simplifiée de l'homographie pour extraire l'angle de rotation
    """
    if H is None:
        return None
    
    # Calcul de l'homographie inverse (référence → mobile)
    H_inv = np.linalg.inv(H)
    
    # Extraction des paramètres de la transformation
    a, b = H_inv[0,0], H_inv[0,1]
    
    # Rotation
    theta_rad = np.arctan2(b, a)
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg

def test_rotations():
    """
    Test de différentes rotations pures
    """
        # Charger l'image de référence - SECTION MODIFIÉE
    img_ref = cv2.imread('results/brain.jpg')
    if img_ref is None:
        print("❌ Erreur: Impossible de charger l'image 'results/brain.jpg'")
        print("Vérifiez le chemin du fichier")
        return None, None, None
    
    # Conversion en niveaux de gris si l'image est en couleur
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    
    # Normalisation entre 0 et 1
    original = img_ref.astype(np.float64) / 255.0
    
    # Rotations à tester
    rotations_reelles = [15, 35, -15.50, 40, -20]
    rotations_estimées = []
    erreurs = []
    
    # Initialisation du recalage
    registrator = SimpleImageRegistration()
    
    print("=== Test de rotations pures avec SIFT + RANSAC ===")
    print("=" * 50)
    
    for i, angle in enumerate(rotations_reelles):
        print(f"\nTest {i+1}: Rotation de {angle}°")
        
        # Créer l'image mobile par rotation
        img_mov = create_rotated_image(original, angle)
        
        # Recalage
        registered_image, info = registrator.register_images(original, img_mov)
        
        if registered_image is not None and info is not None:
            # Estimation de la rotation
            angle_estime = decompose_homography_simple(info['homography'])
            rotations_estimées.append(angle_estime)
            
            erreur = abs(angle_estime - angle)
            erreurs.append(erreur)
            
            print(f"  Rotation réelle: {angle:6.2f}°")
            print(f"  Rotation estimée: {angle_estime:6.2f}°")
            print(f"  Erreur: {erreur:6.2f}°")
            print(f"  Inliers: {info['inliers']}/{info['matches_total']} ({info['inlier_ratio']*100:.1f}%)")
        else:
            print(f"  ❌ Échec du recalage pour {angle}°")
            rotations_estimées.append(None)
            erreurs.append(None)
    
    # Affichage des résultats
    print("\n" + "=" * 50)
    print("RÉSULTATS FINAUX:")
    print("=" * 50)
    
    for i, (angle_reel, angle_est, erreur) in enumerate(zip(rotations_reelles, rotations_estimées, erreurs)):
        if angle_est is not None:
            print(f"Test {i+1}: {angle_reel:6.1f}° → {angle_est:6.1f}° (erreur: {erreur:5.2f}°)")
        else:
            print(f"Test {i+1}: {angle_reel:6.1f}° → ÉCHEC")
    
    # Calcul des statistiques
    erreurs_valides = [e for e in erreurs if e is not None]
    if erreurs_valides:
        erreur_moyenne = np.mean(erreurs_valides)
        erreur_max = np.max(erreurs_valides)
        succes = len(erreurs_valides) / len(rotations_reelles) * 100
        
        print("\n" + "=" * 50)
        print(f"STATISTIQUES:")
        print(f"Taux de succès: {succes:.1f}%")
        print(f"Erreur moyenne: {erreur_moyenne:.2f}°")
        print(f"Erreur maximale: {erreur_max:.2f}°")
    
    # Graphique de comparaison
    plt.figure(figsize=(10, 6))
    
    # Données valides pour le graphique
    indices_valides = [i for i, angle in enumerate(rotations_estimées) if angle is not None]
    angles_reels_valides = [rotations_reelles[i] for i in indices_valides]
    angles_estimes_valides = [rotations_estimées[i] for i in indices_valides]
    
    plt.plot(angles_reels_valides, angles_estimes_valides, 'bo-', markersize=8, 
             label='Rotations estimées', linewidth=2)
    plt.plot(rotations_reelles, rotations_reelles, 'r--', 
             label='Rotations réelles', linewidth=2)
    
    plt.xlabel('Rotation appliquée (°)')
    plt.ylabel('Rotation estimée (°)')
    plt.title('Comparaison des rotations estimées vs réelles\n(SIFT + RANSAC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ajouter les valeurs d'erreur
    for i, (x, y, err) in enumerate(zip(angles_reels_valides, angles_estimes_valides, 
                                       [e for e in erreurs if e is not None])):
        plt.annotate(f'err: {err:.1f}°', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.show()
    
    return rotations_reelles, rotations_estimées, erreurs

# Fonction principale
if __name__ == "__main__":
    # Test des rotations
    rotations_reelles, rotations_estimées, erreurs = test_rotations()
    
    # Affichage détaillé dans la console
    print("\n\nDÉTAIL DES TESTS:")
    print("-" * 60)
    for i, (reel, estime, erreur) in enumerate(zip(rotations_reelles, rotations_estimées, erreurs)):
        if estime is not None:
            status = "✓ SUCCÈS" if abs(erreur) < 5 else "⚠ PRÉCISION MOYENNE"
            print(f"Test {i+1}: {reel:6.1f}° → {estime:6.1f}° | Erreur: {erreur:5.1f}° | {status}")
        else:
            print(f"Test {i+1}: {reel:6.1f}° → ÉCHEC | Erreur: N/A | ❌ ÉCHEC")