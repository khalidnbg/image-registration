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
    """Création d'images de test avec votre propre image"""
    # Charger votre image
    image_path = "results/brain.jpg"
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    
    if original is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    original_shape = original.shape
    
    # 1. Redimensionnement de l'image de référence (-30% mais même résolution)
    scale_factor = 1  # Réduction de 30%
    
    # Redimensionner l'image
    resized_height = int(original_shape[0] * scale_factor)
    resized_width = int(original_shape[1] * scale_factor)
    resized = transform.resize(original, (resized_height, resized_width), anti_aliasing=True)
    
    # Créer une image de la taille originale avec l'image redimensionnée centrée
    ref_image = np.zeros(original_shape)
    
    # Calculer les positions pour centrer l'image redimensionnée
    start_y = (original_shape[0] - resized_height) // 2
    start_x = (original_shape[1] - resized_width) // 2
    end_y = start_y + resized_height
    end_x = start_x + resized_width
    
    # Placer l'image redimensionnée au centre
    ref_image[start_y:end_y, start_x:end_x] = resized
    
    # 2. Paramètres de transformation à appliquer sur l'image de référence
    true_angle = 35  # degrés
    true_scale = 1.25
    true_tx = 15.55
    true_ty = 20.60
    
    # Transformation : rotation + échelle + translation
    tform = transform.SimilarityTransform(
        rotation=np.radians(true_angle),
        scale=true_scale,
        translation=(true_tx, true_ty)
    )
    
    # 3. Application de la transformation sur l'image de référence redimensionnée
    transformed = transform.warp(ref_image, tform.inverse, output_shape=original_shape)
    
    true_params = {
        'angle': true_angle,
        'scale': true_scale,
        'tx': true_tx,
        'ty': true_ty
    }
    
    return ref_image, transformed, true_params

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

def visualize_matches(ref_image, mov_image, info):
    """Visualisation des correspondances entre images de référence et mobile"""
    
    # Conversion en uint8 pour OpenCV si nécessaire
    if ref_image.dtype != np.uint8:
        ref_uint8 = (ref_image * 255).astype(np.uint8) if ref_image.max() <= 1.0 else ref_image.astype(np.uint8)
    else:
        ref_uint8 = ref_image
        
    if mov_image.dtype != np.uint8:
        mov_uint8 = (mov_image * 255).astype(np.uint8) if mov_image.max() <= 1.0 else mov_image.astype(np.uint8)
    else:
        mov_uint8 = mov_image
    
    # Conversion en BGR pour OpenCV
    if len(ref_uint8.shape) == 2:
        ref_bgr = cv2.cvtColor(ref_uint8, cv2.COLOR_GRAY2BGR)
    else:
        ref_bgr = ref_uint8
        
    if len(mov_uint8.shape) == 2:
        mov_bgr = cv2.cvtColor(mov_uint8, cv2.COLOR_GRAY2BGR)
    else:
        mov_bgr = mov_uint8
    
    # Figure 1: Toutes les correspondances (avec OpenCV)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Visualisation des Correspondances SIFT', fontsize=16, fontweight='bold')
    
    # 1.1 Correspondances avec OpenCV (échantillon)
    sample_matches = info['matches'][::max(1, len(info['matches'])//50)]  # Max 50 correspondances
    img_matches = cv2.drawMatches(mov_bgr, info['keypoints_mov'], 
                                 ref_bgr, info['keypoints_ref'], 
                                 sample_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    axes[0,0].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title(f'Correspondances SIFT (échantillon de {len(sample_matches)})')
    axes[0,0].axis('off')
    
    # 1.2 Distinction Inliers/Outliers
    h1, w1 = ref_image.shape[:2]
    h2, w2 = mov_image.shape[:2]
    
    # Image combinée
    combined = np.zeros((max(h1, h2), w1 + w2, 3))
    combined[:h1, :w1] = np.stack([ref_image]*3, axis=-1) if len(ref_image.shape) == 2 else ref_image
    combined[:h2, w1:] = np.stack([mov_image]*3, axis=-1) if len(mov_image.shape) == 2 else mov_image
    
    axes[0,1].imshow(combined)
    
    # Affichage des inliers (vert)
    inlier_matches = [info['matches'][i] for i in range(len(info['matches'])) if info['mask'][i]]
    sample_inliers = inlier_matches[::max(1, len(inlier_matches)//30)]
    
    for match in sample_inliers:
        kp_ref = info['keypoints_ref'][match.trainIdx]
        kp_mov = info['keypoints_mov'][match.queryIdx]
        axes[0,1].plot([kp_ref.pt[0], kp_mov.pt[0] + w1], 
                      [kp_ref.pt[1], kp_mov.pt[1]], 'g-', alpha=0.7, linewidth=1.5)
        axes[0,1].plot(kp_ref.pt[0], kp_ref.pt[1], 'go', markersize=4)
        axes[0,1].plot(kp_mov.pt[0] + w1, kp_mov.pt[1], 'go', markersize=4)
    
    # Affichage des outliers (rouge)
    outlier_matches = [info['matches'][i] for i in range(len(info['matches'])) if not info['mask'][i]]
    if len(outlier_matches) > 0:
        sample_outliers = outlier_matches[::max(1, len(outlier_matches)//10)]
        for match in sample_outliers:
            kp_ref = info['keypoints_ref'][match.trainIdx]
            kp_mov = info['keypoints_mov'][match.queryIdx]
            axes[0,1].plot([kp_ref.pt[0], kp_mov.pt[0] + w1], 
                          [kp_ref.pt[1], kp_mov.pt[1]], 'r-', alpha=0.8, linewidth=1)
            axes[0,1].plot(kp_ref.pt[0], kp_ref.pt[1], 'ro', markersize=3)
            axes[0,1].plot(kp_mov.pt[0] + w1, kp_mov.pt[1], 'ro', markersize=3)
    
    axes[0,1].set_title(f'Inliers (vert): {len(inlier_matches)} | Outliers (rouge): {len(outlier_matches)}')
    axes[0,1].axis('off')
    
    # Ajout d'une légende
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label=f'Inliers ({len(inlier_matches)})'),
                      Patch(facecolor='red', alpha=0.7, label=f'Outliers ({len(outlier_matches)})')] 
    axes[0,1].legend(handles=legend_elements, loc='upper right')
    
    # 1.3 Distribution des distances d'appariement
    axes[1,0].hist([m.distance for m in inlier_matches], bins=20, alpha=0.7, 
                   color='green', label=f'Inliers ({len(inlier_matches)})', edgecolor='black')
    
    if len(outlier_matches) > 0:
        axes[1,0].hist([m.distance for m in outlier_matches], bins=20, alpha=0.7, 
                       color='red', label=f'Outliers ({len(outlier_matches)})', edgecolor='black')
    
    axes[1,0].set_xlabel('Distance euclidienne des descripteurs')
    axes[1,0].set_ylabel('Fréquence')
    axes[1,0].set_title('Distribution des distances d\'appariement')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Ligne verticale pour la distance moyenne des inliers
    mean_inlier_dist = np.mean([m.distance for m in inlier_matches])
    axes[1,0].axvline(x=mean_inlier_dist, color='green', linestyle='--', alpha=0.8,
                      label=f'Moyenne inliers: {mean_inlier_dist:.1f}')
    
    # 1.4 Répartition spatiale des keypoints dans l'image de référence
    axes[1,1].imshow(ref_image, cmap='gray', alpha=0.7)
    
    # Keypoints inliers
    if len(inlier_matches) > 0:
        inlier_coords = [(info['keypoints_ref'][m.trainIdx].pt[0], 
                         info['keypoints_ref'][m.trainIdx].pt[1]) for m in inlier_matches]
        inlier_x, inlier_y = zip(*inlier_coords)
        axes[1,1].scatter(inlier_x, inlier_y, c='green', s=30, alpha=0.8, 
                         label=f'Keypoints inliers ({len(inlier_x)})')
    
    # Keypoints outliers  
    if len(outlier_matches) > 0:
        outlier_coords = [(info['keypoints_ref'][m.trainIdx].pt[0], 
                          info['keypoints_ref'][m.trainIdx].pt[1]) for m in outlier_matches]
        outlier_x, outlier_y = zip(*outlier_coords)
        axes[1,1].scatter(outlier_x, outlier_y, c='red', s=30, alpha=0.8,
                         label=f'Keypoints outliers ({len(outlier_x)})')
    
    axes[1,1].set_title('Répartition spatiale des keypoints\n(Image de référence)')
    axes[1,1].legend()
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    # Figure 2: Analyse détaillée des correspondances
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Analyse Détaillée des Correspondances', fontsize=16, fontweight='bold')
    
    # 2.1 Matrice d'homographie visualisée
    H = info['homography']
    im = axes2[0].imshow(H, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=axes2[0], shrink=0.8)
    
    # Annotations des valeurs
    for i in range(3):
        for j in range(3):
            axes2[0].text(j, i, f'{H[i,j]:.3f}', ha='center', va='center', 
                         fontsize=12, fontweight='bold', 
                         color='white' if abs(H[i,j]) > 0.5 else 'black')
    
    axes2[0].set_title('Matrice d\'Homographie\n(Mobile → Référence)')
    axes2[0].set_xticks([0,1,2])
    axes2[0].set_yticks([0,1,2])
    
    # 2.2 Statistiques des correspondances
    stats_data = [
        len(info['keypoints_ref']),
        len(info['keypoints_mov']),
        info['matches_total'],
        len(inlier_matches),
        len(outlier_matches)
    ]
    
    stats_labels = ['Keypoints\nRéférence', 'Keypoints\nMobile', 'Correspondances\nTotales', 
                   'Inliers', 'Outliers']
    colors = ['lightblue', 'lightcoral', 'lightgray', 'lightgreen', 'salmon']
    
    bars = axes2[1].bar(stats_labels, stats_data, color=colors, alpha=0.8, edgecolor='black')
    axes2[1].set_ylabel('Nombre')
    axes2[1].set_title('Statistiques des Correspondances')
    axes2[1].grid(True, alpha=0.3, axis='y')
    
    # Annotations sur les barres
    for bar, value in zip(bars, stats_data):
        height = bar.get_height()
        axes2[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # 2.3 Évolution du taux d'inliers (simulation)
    # Simulation de l'évolution RANSAC (pour illustration)
    iterations = np.arange(1, 101)
    inlier_ratio = info['inlier_ratio']
    
    # Simulation d'une convergence RANSAC
    simulated_ratios = []
    current_best = 0.3  # Démarrage avec 30%
    for i in iterations:
        if i < 20:
            # Phase d'exploration
            improvement = np.random.normal(0.02, 0.01) * (i/20)
        else:
            # Phase de convergence
            improvement = np.random.normal(0.001, 0.005)
        
        current_best = min(inlier_ratio, current_best + improvement + np.random.normal(0, 0.005))
        simulated_ratios.append(current_best)
    
    axes2[2].plot(iterations, np.array(simulated_ratios) * 100, 'b-', linewidth=2, alpha=0.8)
    axes2[2].axhline(y=inlier_ratio * 100, color='red', linestyle='--', linewidth=2,
                    label=f'Taux final: {inlier_ratio*100:.1f}%')
    axes2[2].set_xlabel('Itérations RANSAC (simulées)')
    axes2[2].set_ylabel('Taux d\'inliers (%)')
    axes2[2].set_title('Convergence RANSAC\n(Illustration)')
    axes2[2].grid(True, alpha=0.3)
    axes2[2].legend()
    axes2[2].set_ylim(20, 100)
    
    plt.tight_layout()
    
    return fig, fig2
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
    print(f"Image de référence: redimensionnée à 70% puis centrée")
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

        visualize_matches(ref_image, mov_image, info)
        plt.show() 
        
    else:
        print("❌ Échec du recalage")

    

if __name__ == "__main__":
    main()