import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data, transform
from sklearn.metrics import mean_squared_error
import time
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches

class SIFTImageRegistration:
    """
    Classe complète pour le recalage d'images par SIFT avec analyse avancée
    """
    
    def __init__(self, nfeatures=5000, contrastThreshold=0.03, edgeThreshold=10, sigma=1.6):
        """
        Initialise le détecteur SIFT avec les paramètres
        
        Args:
            nfeatures: Nombre maximum de features à détecter
            contrastThreshold: Seuil de contraste pour filtrer les keypoints faibles
            edgeThreshold: Seuil pour éliminer les réponses sur les arêtes
            sigma: Écart-type du noyau gaussien de base
        """
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )
        
        # Paramètres pour l'appariement
        self.lowe_ratio = 0.8
        
        # Paramètres pour RANSAC
        self.ransac_threshold = 5.0
        self.ransac_maxIters = 5000
        self.ransac_confidence = 0.99
    
    def detect_and_compute(self, image):
        """
        Détection des keypoints et calcul des descripteurs SIFT
        
        Args:
            image: Image en niveaux de gris
            
        Returns:
            keypoints: Liste des keypoints détectés
            descriptors: Descripteurs SIFT (array 128D pour chaque keypoint)
        """
        # Conversion en uint8 si nécessaire
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            image_uint8 = image
            
        # Détection et description SIFT
        keypoints, descriptors = self.sift.detectAndCompute(image_uint8, None)
        
        print(f"Nombre de keypoints détectés : {len(keypoints)}")
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Appariement des caractéristiques avec le test du ratio de Lowe
        
        Args:
            desc1, desc2: Descripteurs des deux images
            
        Returns:
            good_matches: Liste des appariements valides
            all_matches: Tous les appariements avant filtrage
        """
        # Matcher FLANN (Fast Library for Approximate Nearest Neighbors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Recherche des k=2 plus proches voisins pour chaque descripteur
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Test du ratio de Lowe
        good_matches = []
        ratio_values = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                ratio = m.distance / n.distance
                ratio_values.append(ratio)
                # Si la distance au premier voisin est suffisamment plus petite
                # que la distance au second voisin
                if ratio < self.lowe_ratio:
                    good_matches.append(m)
        
        print(f"Nombre d'appariements après filtrage : {len(good_matches)}")
        
        return good_matches, matches, ratio_values
    
    def estimate_homography_ransac(self, kp1, kp2, matches):
        """
        Estimation robuste de l'homographie par RANSAC
        
        Args:
            kp1, kp2: Keypoints des deux images
            matches: Appariements entre les keypoints
            
        Returns:
            H: Matrice d'homographie 3x3
            mask: Masque des inliers
            inlier_matches: Appariements inliers
            outlier_matches: Appariements outliers
        """
        if len(matches) < 4:
            print("Erreur: Moins de 4 appariements, impossible d'estimer l'homographie")
            return None, None, [], []
        
        # Extraction des coordonnées des points appariés
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimation de l'homographie par RANSAC
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            self.ransac_threshold,
            maxIters=self.ransac_maxIters,
            confidence=self.ransac_confidence
        )
        
        if H is not None:
            # Comptage des inliers
            inliers_count = np.sum(mask)
            print(f"Homographie estimée avec {inliers_count}/{len(matches)} inliers")
            
            # Séparation inliers/outliers
            inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
            outlier_matches = [matches[i] for i in range(len(matches)) if not mask[i]]
            
            return H, mask, inlier_matches, outlier_matches
        else:
            print("Erreur: Impossible d'estimer une homographie robuste")
            return None, None, [], []
    
    def analyze_homography_precision(self, H, true_params):
        """
        Analyse de précision de l'homographie estimée
        
        Args:
            H: Homographie estimée
            true_params: Paramètres de transformation vrais
            
        Returns:
            analysis: Dictionnaire d'analyse complète
        """
        if H is None:
            return None
        
        # Calcul de l'homographie inverse (référence → mobile)
        H_inv = np.linalg.inv(H)
        
        def decompose_similarity_homography(H_mat):
            """Décomposition d'une homographie proche d'une similarité"""
            a, b = H_mat[0,0], H_mat[0,1] 
            c, d = H_mat[1,0], H_mat[1,1]
            tx, ty = H_mat[0,2], H_mat[1,2]
            
            # Facteurs d'échelle
            sx = np.sqrt(a**2 + b**2)
            sy = np.sqrt(c**2 + d**2)
            
            # Rotation (à partir de la matrice de rotation normalisée)
            cos_theta = a / sx
            sin_theta = b / sx  
            theta_rad = np.arctan2(sin_theta, cos_theta)
            theta_deg = np.degrees(theta_rad)
            
            return {
                'scale_x': sx,
                'scale_y': sy, 
                'rotation_deg': theta_deg,
                'translation_x': tx,
                'translation_y': ty,
                'scale_avg': (sx + sy) / 2
            }
        
        # Décomposition des transformations
        decomp_direct = decompose_similarity_homography(H)      # mobile → référence
        decomp_inverse = decompose_similarity_homography(H_inv)  # référence → mobile
        
        # Comparaison avec ground truth (référence → mobile)
        estimated_angle = abs(decomp_inverse['rotation_deg'])
        true_angle = abs(true_params['angle'])
        
        errors = {
            'rotation_error': abs(true_angle - estimated_angle),
            'scale_error': abs(true_params['scale'] - decomp_inverse['scale_avg']),
            'translation_x_error': abs(true_params['tx'] - decomp_inverse['translation_x']),
            'translation_y_error': abs(true_params['ty'] - decomp_inverse['translation_y'])
        }
        
        # Score de précision
        score_rotation = max(0, 100 - errors['rotation_error'] * 10)
        score_scale = max(0, 100 - errors['scale_error'] * 1000)
        score_translation = max(0, 100 - (errors['translation_x_error'] + errors['translation_y_error']) * 5)
        score_global = (score_rotation + score_scale + score_translation) / 3
        
        analysis = {
            'homography': H,
            'homography_inverse': H_inv,
            'decomposition_direct': decomp_direct,
            'decomposition_inverse': decomp_inverse,
            'errors': errors,
            'precision_scores': {
                'rotation': score_rotation,
                'scale': score_scale, 
                'translation': score_translation,
                'global': score_global
            },
            'estimated_params': {
                'angle': decomp_inverse['rotation_deg'],
                'scale': decomp_inverse['scale_avg'],
                'tx': decomp_inverse['translation_x'],
                'ty': decomp_inverse['translation_y']
            }
        }
        
        return analysis
    
    def apply_transformation(self, image, H, output_shape):
        """
        Application de la transformation homographique à l'image
        
        Args:
            image: Image à transformer
            H: Matrice d'homographie
            output_shape: Forme de l'image de sortie (height, width)
            
        Returns:
            transformed_image: Image transformée
        """
        # Conversion en uint8 si nécessaire
        if image.dtype != np.uint8:
            image_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            image_uint8 = image
        
        # Application de la transformation
        transformed = cv2.warpPerspective(image_uint8, H, (output_shape[1], output_shape[0]))
        
        # Retour au type float dans [0,1] si nécessaire
        if image.max() <= 1.0:
            transformed = transformed.astype(np.float64) / 255.0
            
        return transformed
    
    def register_images(self, image_ref, image_mov, true_params=None):
        """
        Recalage complet de deux images avec analyse de précision
        
        Args:
            image_ref: Image de référence (fixe)
            image_mov: Image à recaler (mobile)
            true_params: Paramètres de transformation vrais (pour évaluation)
            
        Returns:
            registered_image: Image recalée
            registration_info: Dictionnaire complet avec informations du recalage
        """
        print("=== Début du recalage d'images par SIFT ===")
        start_time = time.time()
        
        # Étape 1: Détection et description des caractéristiques
        print("\nÉtape 1: Détection des caractéristiques SIFT")
        kp_ref, desc_ref = self.detect_and_compute(image_ref)
        kp_mov, desc_mov = self.detect_and_compute(image_mov)
        
        if desc_ref is None or desc_mov is None:
            print("Erreur: Aucune caractéristique détectée")
            return None, None
        
        # Étape 2: Appariement des caractéristiques
        print("\nÉtape 2: Appariement des caractéristiques")
        matches, all_matches, ratio_values = self.match_features(desc_mov, desc_ref)
        
        if len(matches) < 4:
            print("Erreur: Pas assez d'appariements pour l'estimation d'homographie")
            return None, None
        
        # Étape 3: Estimation robuste de la transformation
        print("\nÉtape 3: Estimation de l'homographie par RANSAC")
        H, mask, inlier_matches, outlier_matches = self.estimate_homography_ransac(kp_mov, kp_ref, matches)
        
        if H is None:
            print("Erreur: Échec de l'estimation d'homographie")
            return None, None
        
        # Étape 4: Analyse de précision
        precision_analysis = None
        if true_params is not None:
            print("\nÉtape 4: Analyse de précision")
            precision_analysis = self.analyze_homography_precision(H, true_params)
        
        # Étape 5: Application de la transformation
        print(f"\nÉtape {'5' if true_params is not None else '4'}: Application de la transformation")
        registered_image = self.apply_transformation(image_mov, H, image_ref.shape)
        
        # Calcul du temps de traitement
        processing_time = time.time() - start_time
        
        # Informations complètes de recalage
        registration_info = {
            'processing_time': processing_time,
            'keypoints_ref': len(kp_ref),
            'keypoints_mov': len(kp_mov),
            'matches_total': len(matches),
            'matches_inliers': len(inlier_matches),
            'matches_outliers': len(outlier_matches),
            'inlier_ratio': len(inlier_matches) / len(matches) if len(matches) > 0 else 0,
            'homography_matrix': H,
            'keypoints_ref_obj': kp_ref,
            'keypoints_mov_obj': kp_mov,
            'inlier_matches': inlier_matches,
            'outlier_matches': outlier_matches,
            'all_matches': all_matches,
            'ratio_values': ratio_values,
            'precision_analysis': precision_analysis,
            'true_params': true_params
        }
        
        print(f"\n=== Recalage terminé en {processing_time:.2f} secondes ===")
        print(f"Taux d'inliers: {registration_info['inlier_ratio']:.2%}")
        
        if precision_analysis:
            score = precision_analysis['precision_scores']['global']
            print(f"Score de précision: {score:.1f}/100")
        
        return registered_image, registration_info

def create_transformed_image(image, angle=30, scale=0.8, tx=50, ty=30):
    """
    Création d'une version transformée de l'image pour simulation
    
    Args:
        image: Image originale
        angle: Rotation en degrés
        scale: Facteur d'échelle
        tx, ty: Translation en pixels
        
    Returns:
        transformed_image: Image transformée
        true_params: Paramètres de transformation
    """
    # Matrice de transformation combinée (similarity)
    tform = transform.SimilarityTransform(
        rotation=np.radians(angle),
        scale=scale,
        translation=(tx, ty)
    )
    
    # Application de la transformation
    transformed = transform.warp(image, tform.inverse, output_shape=image.shape)
    
    true_params = {
        'angle': angle,
        'scale': scale,
        'tx': tx,
        'ty': ty,
        'transform_object': tform
    }
    
    return transformed, true_params

def evaluate_registration(image_ref, image_registered):
    """
    Évaluation de la qualité du recalage
    
    Args:
        image_ref: Image de référence
        image_registered: Image recalée
        
    Returns:
        metrics: Dictionnaire des métriques d'évaluation
    """
    # Erreur quadratique moyenne (MSE)
    mse = mean_squared_error(image_ref.flatten(), image_registered.flatten())
    
    # Signal-to-Noise Ratio (PSNR)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    # Coefficient de corrélation
    correlation = np.corrcoef(image_ref.flatten(), image_registered.flatten())[0, 1]
    
    # Erreur absolue moyenne (MAE)
    mae = np.mean(np.abs(image_ref - image_registered))
    
    # Coefficient de détermination (R²)
    ss_res = np.sum((image_ref - image_registered) ** 2)
    ss_tot = np.sum((image_ref - np.mean(image_ref)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mse': mse,
        'psnr': psnr,
        'correlation': correlation,
        'mae': mae,
        'r2_score': r2_score
    }

def create_academic_visualizations(image_ref, image_mov, image_registered, registration_info):
    """
    Création de visualisations académiques avancées
    """
    
    # Figure 1: Pipeline complet du recalage SIFT
    fig1 = plt.figure(figsize=(20, 12))
    fig1.suptitle('Pipeline complet du recalage d\'images par SIFT', fontsize=16, fontweight='bold')
    
    # Conversion en uint8 pour visualisation avec keypoints OpenCV
    img_ref_uint8 = (image_ref * 255).astype(np.uint8) if image_ref.max() <= 1.0 else image_ref.astype(np.uint8)
    img_mov_uint8 = (image_mov * 255).astype(np.uint8) if image_mov.max() <= 1.0 else image_mov.astype(np.uint8)
    
    # 1.1: Image de référence avec keypoints
    plt.subplot(2, 4, 1)
    img_ref_kp = cv2.drawKeypoints(img_ref_uint8, registration_info['keypoints_ref_obj'][:50], None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(img_ref_kp, cv2.COLOR_BGR2RGB))
    plt.title(f'Image de référence\n{registration_info["keypoints_ref"]} keypoints SIFT')
    plt.axis('off')
    
    # 1.2: Image mobile avec keypoints  
    plt.subplot(2, 4, 2)
    img_mov_kp = cv2.drawKeypoints(img_mov_uint8, registration_info['keypoints_mov_obj'][:50], None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(img_mov_kp, cv2.COLOR_BGR2RGB))
    plt.title(f'Image mobile\n{registration_info["keypoints_mov"]} keypoints SIFT')
    plt.axis('off')
    
    # 1.3: Appariements bruts
    plt.subplot(2, 4, 3)
    h1, w1 = image_ref.shape
    h2, w2 = image_mov.shape
    combined_raw = np.zeros((max(h1, h2), w1 + w2))
    combined_raw[:h1, :w1] = image_ref
    combined_raw[:h2, w1:] = image_mov
    plt.imshow(combined_raw, cmap='gray')
    
    # Affichage d'un échantillon d'appariements bruts
    sample_matches = registration_info['inlier_matches'][::max(1, len(registration_info['inlier_matches'])//20)]
    for match in sample_matches:
        kp_ref = registration_info['keypoints_ref_obj'][match.trainIdx]
        kp_mov = registration_info['keypoints_mov_obj'][match.queryIdx]
        plt.plot([kp_ref.pt[0], kp_mov.pt[0] + w1], 
                [kp_ref.pt[1], kp_mov.pt[1]], 'g-', alpha=0.6, linewidth=1)
    
    plt.title(f'Appariements inliers\n{registration_info["matches_inliers"]} correspondances')
    plt.axis('off')
    
    # 1.4: Matrice d'homographie
    plt.subplot(2, 4, 4)
    H_display = registration_info['homography_matrix'].copy()
    im = plt.imshow(H_display, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, shrink=0.8)
    
    # Annotations des valeurs
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{H_display[i,j]:.3f}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.title('Matrice d\'homographie H\n(mobile → référence)')
    plt.xticks([0,1,2])
    plt.yticks([0,1,2])
    
    # 1.5: Image recalée
    plt.subplot(2, 4, 5)
    plt.imshow(image_registered, cmap='gray')
    plt.title('Image recalée')
    plt.axis('off')
    
    # 1.6: Différence avant/après
    plt.subplot(2, 4, 6)
    diff_before = np.abs(image_ref - image_mov)
    diff_after = np.abs(image_ref - image_registered)
    
    plt.imshow(diff_before, cmap='hot', alpha=0.7)
    plt.title('Différence avant recalage')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    plt.subplot(2, 4, 7) 
    plt.imshow(diff_after, cmap='hot', alpha=0.7)
    plt.title('Différence après recalage')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    # 1.7: Superposition finale
    plt.subplot(2, 4, 8)
    overlay = np.zeros((image_ref.shape[0], image_ref.shape[1], 3))
    overlay[:,:,0] = image_ref
    overlay[:,:,1] = image_registered
    overlay[:,:,2] = image_registered * 0.5
    plt.imshow(overlay)
    plt.title('Superposition\n(Rouge: Réf, Vert: Recalée)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Figure 2: Analyse de précision et statistiques
    if registration_info['precision_analysis'] is not None:
        fig2 = plt.figure(figsize=(16, 10))
        fig2.suptitle('Analyse de précision du recalage SIFT', fontsize=16, fontweight='bold')
        
        precision = registration_info['precision_analysis']
        
        # 2.1: Comparaison paramètres vrais vs estimés
        plt.subplot(2, 3, 1)
        params = ['Rotation\n(degrés)', 'Échelle', 'Translation X\n(pixels)', 'Translation Y\n(pixels)']
        true_vals = [
            registration_info['true_params']['angle'],
            registration_info['true_params']['scale'],
            registration_info['true_params']['tx'],
            registration_info['true_params']['ty']
        ]
        est_vals = [
            abs(precision['estimated_params']['angle']),  # Valeur absolue pour rotation
            precision['estimated_params']['scale'],
            precision['estimated_params']['tx'], 
            precision['estimated_params']['ty']
        ]
        
        x = np.arange(len(params))
        width = 0.35
        
        plt.bar(x - width/2, true_vals, width, label='Vrai', alpha=0.7, color='blue')
        plt.bar(x + width/2, est_vals, width, label='Estimé', alpha=0.7, color='red')
        
        plt.xlabel('Paramètres')
        plt.ylabel('Valeurs')
        plt.title('Comparaison Vrai vs Estimé')
        plt.xticks(x, params, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2.2: Erreurs par paramètre  
        plt.subplot(2, 3, 2)
        errors = [
            precision['errors']['rotation_error'],
            precision['errors']['scale_error'] * 100,  # En pourcentage
            precision['errors']['translation_x_error'],
            precision['errors']['translation_y_error']
        ]
        error_labels = ['Rotation\n(degrés)', 'Échelle\n(%)', 'Trans. X\n(pixels)', 'Trans. Y\n(pixels)']
        
        bars = plt.bar(error_labels, errors, color=['red' if e > 1 else 'orange' if e > 0.1 else 'green' for e in errors])
        plt.ylabel('Erreur absolue')
        plt.title('Erreurs de recalage')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Annotations
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2.3: Scores de précision
        plt.subplot(2, 3, 3)
        scores = list(precision['precision_scores'].values())
        score_labels = ['Rotation', 'Échelle', 'Translation', 'Global']
        colors = ['green' if s >= 90 else 'orange' if s >= 70 else 'red' for s in scores]
        
        bars = plt.bar(score_labels, scores, color=colors, alpha=0.7)
        plt.ylabel('Score (/100)')
        plt.title('Scores de précision')
        plt.ylim(0, 100)
        plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent')
        plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Bon') 
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Annotations
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
        # 2.4: Distribution du ratio de Lowe
        plt.subplot(2, 3, 4)
        plt.hist(registration_info['ratio_values'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label=f'Seuil Lowe = {0.8}')
        plt.xlabel('Ratio distance 1er/2ème voisin')
        plt.ylabel('Fréquence')
        plt.title('Distribution des ratios de Lowe')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2.5: Statistiques temporelles et spatiales
        plt.subplot(2, 3, 5)
        stats_data = [
            registration_info['processing_time'],
            registration_info['keypoints_ref'] / 100,  # Normalisé par 100
            registration_info['keypoints_mov'] / 100,  # Normalisé par 100
            registration_info['inlier_ratio'] * 100    # En pourcentage
        ]
        stats_labels = ['Temps\n(secondes)', 'Keypoints Réf\n(×100)', 'Keypoints Mob\n(×100)', 'Taux inliers\n(%)']
        
        plt.bar(stats_labels, stats_data, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
        plt.ylabel('Valeurs')
        plt.title('Statistiques de performance')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Annotations
        for i, (label, value) in enumerate(zip(stats_labels, stats_data)):
            plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2.6: Matrice d'homographie décomposée
        plt.subplot(2, 3, 6)
        decomp = precision['decomposition_inverse']
        
        decomp_data = [decomp['scale_avg'], abs(decomp['rotation_deg']), 
                      abs(decomp['translation_x'])/10, abs(decomp['translation_y'])/10]
        decomp_labels = ['Échelle', 'Rotation\n(degrés)', 'Trans. X\n(÷10)', 'Trans. Y\n(÷10)']
        
        plt.bar(decomp_labels, decomp_data, color='skyblue', alpha=0.7, edgecolor='black')
        plt.ylabel('Valeurs normalisées')
        plt.title('Décomposition homographie\n(référence → mobile)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    # Figure 3: Analyse détaillée des correspondances
    fig3 = plt.figure(figsize=(18, 8))
    fig3.suptitle('Analyse détaillée des correspondances SIFT', fontsize=16, fontweight='bold')
    
    # 3.1: Correspondances avec distinction inliers/outliers
    plt.subplot(1, 3, 1)
    combined_detailed = np.zeros((max(h1, h2), w1 + w2, 3))
    combined_detailed[:h1, :w1, :] = np.stack([image_ref]*3, axis=-1)
    combined_detailed[:h2, w1:, :] = np.stack([image_mov]*3, axis=-1)
    plt.imshow(combined_detailed)
    
    # Affichage des inliers en vert
    sample_inliers = registration_info['inlier_matches'][::max(1, len(registration_info['inlier_matches'])//30)]
    for match in sample_inliers:
        kp_ref = registration_info['keypoints_ref_obj'][match.trainIdx]
        kp_mov = registration_info['keypoints_mov_obj'][match.queryIdx]
        plt.plot([kp_ref.pt[0], kp_mov.pt[0] + w1], 
                [kp_ref.pt[1], kp_mov.pt[1]], 'g-', alpha=0.7, linewidth=1.5)
        plt.plot(kp_ref.pt[0], kp_ref.pt[1], 'go', markersize=3)
        plt.plot(kp_mov.pt[0] + w1, kp_mov.pt[1], 'go', markersize=3)
    
    # Affichage des outliers en rouge
    if len(registration_info['outlier_matches']) > 0:
        sample_outliers = registration_info['outlier_matches'][::max(1, len(registration_info['outlier_matches'])//10)]
        for match in sample_outliers:
            kp_ref = registration_info['keypoints_ref_obj'][match.trainIdx]
            kp_mov = registration_info['keypoints_mov_obj'][match.queryIdx]
            plt.plot([kp_ref.pt[0], kp_mov.pt[0] + w1], 
                    [kp_ref.pt[1], kp_mov.pt[1]], 'r-', alpha=0.8, linewidth=1)
            plt.plot(kp_ref.pt[0], kp_ref.pt[1], 'ro', markersize=3)
            plt.plot(kp_mov.pt[0] + w1, kp_mov.pt[1], 'ro', markersize=3)
    
    # Légende
    green_patch = mpatches.Patch(color='green', label=f'Inliers ({registration_info["matches_inliers"]})')
    red_patch = mpatches.Patch(color='red', label=f'Outliers ({registration_info["matches_outliers"]})')
    plt.legend(handles=[green_patch, red_patch], loc='upper right')
    
    plt.title('Correspondances SIFT\n(Vert: Inliers, Rouge: Outliers)')
    plt.axis('off')
    
    # 3.2: Distribution des distances d'appariement
    plt.subplot(1, 3, 2)
    inlier_distances = [m.distance for m in registration_info['inlier_matches']]
    if len(registration_info['outlier_matches']) > 0:
        outlier_distances = [m.distance for m in registration_info['outlier_matches']]
        plt.hist(outlier_distances, bins=30, alpha=0.6, label=f'Outliers ({len(outlier_distances)})', 
                color='red', edgecolor='black')
    
    plt.hist(inlier_distances, bins=30, alpha=0.7, label=f'Inliers ({len(inlier_distances)})', 
            color='green', edgecolor='black')
    
    plt.xlabel('Distance euclidienne des descripteurs')
    plt.ylabel('Fréquence')
    plt.title('Distribution des distances\nd\'appariement SIFT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistiques
    mean_inlier = np.mean(inlier_distances)
    plt.axvline(x=mean_inlier, color='green', linestyle='--', alpha=0.8, 
               label=f'Moyenne inliers: {mean_inlier:.1f}')
    
    # 3.3: Répartition spatiale des keypoints
    plt.subplot(1, 3, 3)
    plt.imshow(image_ref, cmap='gray', alpha=0.7)
    
    # Keypoints inliers
    inlier_ref_coords = [(registration_info['keypoints_ref_obj'][m.trainIdx].pt[0], 
                         registration_info['keypoints_ref_obj'][m.trainIdx].pt[1]) 
                        for m in registration_info['inlier_matches']]
    if inlier_ref_coords:
        inlier_x, inlier_y = zip(*inlier_ref_coords)
        plt.scatter(inlier_x, inlier_y, c='green', s=20, alpha=0.8, label=f'Keypoints inliers ({len(inlier_x)})')
    
    # Keypoints outliers
    if len(registration_info['outlier_matches']) > 0:
        outlier_ref_coords = [(registration_info['keypoints_ref_obj'][m.trainIdx].pt[0], 
                             registration_info['keypoints_ref_obj'][m.trainIdx].pt[1]) 
                            for m in registration_info['outlier_matches']]
        if outlier_ref_coords:
            outlier_x, outlier_y = zip(*outlier_ref_coords)
            plt.scatter(outlier_x, outlier_y, c='red', s=20, alpha=0.8, label=f'Keypoints outliers ({len(outlier_x)})')
    
    plt.title('Répartition spatiale des keypoints\ndans l\'image de référence')
    plt.legend()
    plt.axis('off')
    
    plt.tight_layout()
    
    return fig1, fig2 if registration_info['precision_analysis'] is not None else None, fig3

def print_detailed_analysis(registration_info):
    """
    Affichage détaillé de l'analyse de recalage
    """
    print("\n" + "="*80)
    print("ANALYSE DÉTAILLÉE DU RECALAGE SIFT")
    print("="*80)
    
    # Informations générales
    print(f"\n📊 STATISTIQUES GÉNÉRALES")
    print(f"├─ Temps de traitement: {registration_info['processing_time']:.2f} secondes")
    print(f"├─ Keypoints détectés - Référence: {registration_info['keypoints_ref']}")
    print(f"├─ Keypoints détectés - Mobile: {registration_info['keypoints_mov']}")
    print(f"├─ Appariements totaux: {registration_info['matches_total']}")
    print(f"├─ Appariements inliers: {registration_info['matches_inliers']}")
    print(f"├─ Appariements outliers: {registration_info['matches_outliers']}")
    print(f"└─ Taux d'inliers: {registration_info['inlier_ratio']:.2%}")
    
    # Analyse de la matrice d'homographie
    print(f"\n🎯 ANALYSE DE L'HOMOGRAPHIE")
    H = registration_info['homography_matrix']
    print(f"Matrice H (mobile → référence):")
    print(f"┌─ {H[0,0]:8.4f} {H[0,1]:8.4f} {H[0,2]:8.1f} ─┐")
    print(f"│  {H[1,0]:8.4f} {H[1,1]:8.4f} {H[1,2]:8.1f}  │")
    print(f"└─ {H[2,0]:8.6f} {H[2,1]:8.6f} {H[2,2]:8.4f} ─┘")
    
    # Déterminant et conditionnement
    det_H = np.linalg.det(H)
    cond_H = np.linalg.cond(H)
    print(f"├─ Déterminant: {det_H:.6f}")
    print(f"└─ Nombre de condition: {cond_H:.2f} ({'Bien conditionné' if cond_H < 100 else 'Mal conditionné'})")
    
    # Analyse de précision si disponible
    if registration_info['precision_analysis'] is not None:
        precision = registration_info['precision_analysis']
        print(f"\n🔍 ANALYSE DE PRÉCISION")
        
        print(f"Paramètres estimés (référence → mobile):")
        est_params = precision['estimated_params']
        true_params = registration_info['true_params']
        
        print(f"├─ Rotation: {est_params['angle']:.2f}° (vrai: {true_params['angle']:.1f}°)")
        print(f"├─ Échelle: {est_params['scale']:.4f} (vrai: {true_params['scale']:.3f})")
        print(f"├─ Translation X: {est_params['tx']:.1f}px (vrai: {true_params['tx']:.1f}px)")
        print(f"└─ Translation Y: {est_params['ty']:.1f}px (vrai: {true_params['ty']:.1f}px)")
        
        print(f"\nErreurs absolues:")
        errors = precision['errors']
        print(f"├─ Rotation: {errors['rotation_error']:.3f}°")
        print(f"├─ Échelle: {errors['scale_error']:.4f}")
        print(f"├─ Translation X: {errors['translation_x_error']:.1f}px")
        print(f"└─ Translation Y: {errors['translation_y_error']:.1f}px")
        
        print(f"\nScores de précision (/100):")
        scores = precision['precision_scores']
        print(f"├─ Rotation: {scores['rotation']:.1f}")
        print(f"├─ Échelle: {scores['scale']:.1f}")
        print(f"├─ Translation: {scores['translation']:.1f}")
        print(f"└─ Global: {scores['global']:.1f}")
        
        # Évaluation qualitative
        global_score = scores['global']
        if global_score >= 95:
            quality = "🟢 EXCELLENT"
        elif global_score >= 85:
            quality = "🟢 TRÈS BON"
        elif global_score >= 70:
            quality = "🟡 BON"
        elif global_score >= 50:
            quality = "🟠 MOYEN"
        else:
            quality = "🔴 FAIBLE"
        
        print(f"\n🏆 QUALITÉ DU RECALAGE: {quality}")
    
    # Statistiques des appariements
    print(f"\n📈 STATISTIQUES DES APPARIEMENTS")
    if registration_info['ratio_values']:
        ratios = registration_info['ratio_values']
        print(f"├─ Ratio de Lowe moyen: {np.mean(ratios):.3f}")
        print(f"├─ Écart-type des ratios: {np.std(ratios):.3f}")
        print(f"├─ Ratio minimum: {np.min(ratios):.3f}")
        print(f"└─ Ratio maximum: {np.max(ratios):.3f}")
    
    # Performance et recommandations
    print(f"\n💡 ÉVALUATION ET RECOMMANDATIONS")
    performance_notes = []
    
    # Évaluation du taux d'inliers
    inlier_ratio = registration_info['inlier_ratio']
    if inlier_ratio >= 0.8:
        performance_notes.append("✅ Excellent taux d'inliers - Transformation bien estimée")
    elif inlier_ratio >= 0.6:
        performance_notes.append("⚠️ Taux d'inliers acceptable - Transformation correcte")
    else:
        performance_notes.append("❌ Faible taux d'inliers - Vérifier les paramètres ou la qualité des images")
    
    # Évaluation du nombre de keypoints
    total_kp = registration_info['keypoints_ref'] + registration_info['keypoints_mov']
    if total_kp >= 1000:
        performance_notes.append("✅ Nombre suffisant de keypoints détectés")
    elif total_kp >= 500:
        performance_notes.append("⚠️ Nombre modéré de keypoints - Acceptable pour la plupart des cas")
    else:
        performance_notes.append("❌ Peu de keypoints détectés - Considérer ajuster les paramètres SIFT")
    
    # Évaluation du temps de traitement
    if registration_info['processing_time'] <= 0.5:
        performance_notes.append("✅ Temps de traitement excellent (temps réel)")
    elif registration_info['processing_time'] <= 2.0:
        performance_notes.append("⚠️ Temps de traitement acceptable")
    else:
        performance_notes.append("❌ Temps de traitement élevé - Considérer optimisation")
    
    for note in performance_notes:
        print(f"├─ {note}")
    
    print("└─" + "─"*75)

# =========================
# SCRIPT PRINCIPAL AMÉLIORÉ
# =========================

def main():
    print("🚀 SYSTÈME DE RECALAGE D'IMAGES PAR SIFT - VERSION ACADÉMIQUE")
    print("="*80)
    
    # Chargement de l'image camera
    print("📂 Chargement de l'image camera de scikit-image...")
    image_original = data.camera()
    
    # Conversion en float [0,1] pour cohérence
    image_original = image_original.astype(np.float64) / 255.0
    
    print(f"├─ Taille de l'image: {image_original.shape}")
    print(f"└─ Type de données: {image_original.dtype}")
    
    # Création d'une version transformée pour simulation
    print("\n🔄 Création d'une version transformée pour simulation...")
    transformation_params = {
        'angle': 12,      # Rotation de 25°
        'scale': 1.25,    # Réduction de 15%
        'tx': 10,         # Translation de 40px en X
        'ty': -15.5         # Translation de -30px en Y
    }
    
    image_transformed, true_params = create_transformed_image(
        image_original, 
        **transformation_params
    )
    
    print(f"├─ Rotation appliquée: {transformation_params['angle']}°")
    print(f"├─ Échelle appliquée: {transformation_params['scale']}")
    print(f"└─ Translation appliquée: ({transformation_params['tx']}, {transformation_params['ty']})")
    
    # Initialisation du système de recalage SIFT
    print("\n⚙️ Initialisation du système de recalage SIFT...")
    registrator = SIFTImageRegistration(
        nfeatures=3000,           # Nombre max de features
        contrastThreshold=0.04,   # Seuil de contraste
        edgeThreshold=10,         # Seuil pour éliminer les arêtes
        sigma=1.6                 # Sigma de base
    )
    
    print(f"├─ Paramètres SIFT configurés")
    print(f"├─ Nombre max de features: 3000")
    print(f"├─ Seuil de contraste: 0.04")
    print(f"└─ Ratio de Lowe: 0.8")
    
    # Recalage des images
    print(f"\n🎯 Lancement du recalage complet...")
    registered_image, registration_info = registrator.register_images(
        image_original,      # Image de référence
        image_transformed,   # Image à recaler
        true_params          # Paramètres vrais pour analyse
    )
    
    if registered_image is not None:
        # Évaluation du recalage
        print(f"\n📊 Évaluation de la qualité du recalage...")
        metrics = evaluate_registration(image_original, registered_image)
        
        print(f"├─ MSE: {metrics['mse']:.6f}")
        print(f"├─ PSNR: {metrics['psnr']:.2f} dB")
        print(f"├─ Corrélation: {metrics['correlation']:.4f}")
        print(f"├─ MAE: {metrics['mae']:.6f}")
        print(f"└─ R² Score: {metrics['r2_score']:.4f}")
        
        # Analyse détaillée
        print_detailed_analysis(registration_info)
        
        # Création des visualisations académiques
        print(f"\n🎨 Génération des visualisations académiques...")
        fig1, fig2, fig3 = create_academic_visualizations(
            image_original, image_transformed, registered_image, registration_info
        )
        
        print(f"├─ Figure 1: Pipeline complet du recalage SIFT")
        print(f"├─ Figure 2: Analyse de précision et statistiques")
        print(f"└─ Figure 3: Analyse détaillée des correspondances")
        
        # Affichage des figures
        plt.show()
        
        print(f"\n✅ RECALAGE TERMINÉ AVEC SUCCÈS!")
        print(f"🏆 Score de précision global: {registration_info['precision_analysis']['precision_scores']['global']:.1f}/100")
        
    else:
        print("❌ ÉCHEC du recalage d'images")
        print("💡 Suggestions:")
        print("├─ Vérifier la qualité des images d'entrée")
        print("├─ Ajuster les paramètres SIFT")
        print("└─ Réduire l'amplitude de la transformation")

if __name__ == "__main__":
    main()