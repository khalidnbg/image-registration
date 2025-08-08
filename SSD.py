import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ssd_cost(params, img_ref, img_mov, mask=None):
    """Calcule le coût SSD entre image de référence et image mobile transformée"""
    tx, ty, angle = params
    
    # Matrice de transformation
    M = cv2.getRotationMatrix2D((img_mov.shape[1]//2, img_mov.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Transformation de l'image mobile
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]))
    
    # Calcul SSD
    if mask is not None:
        diff = (img_ref - img_transformed) * mask
    else:
        diff = img_ref - img_transformed
    
    return np.sum(diff**2)

def register_images_ssd(img_ref, img_mov, initial_params=[0, 0, 0]):
    """Recalage par optimisation SSD"""
    
    # Conversion en niveaux de gris si nécessaire
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    # Optimisation
    result = minimize(ssd_cost, initial_params, 
                     args=(img_ref, img_mov),
                     method='Powell',
                     options={'maxiter': 1000})
    
    return result.x

def apply_transformation(img, params):
    """Applique la transformation trouvée"""
    tx, ty, angle = params
    
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger vos images
    img_ref = cv2.imread('images/tisdrin.png')  # image fixe
    img_mov = cv2.imread('images/tisdrin_translated_12_27.jpg')     # image à recaler
    
    # Recalage
    params = register_images_ssd(img_ref, img_mov)
    print(f"Paramètres trouvés - tx: {params[0]:.2f}, ty: {params[1]:.2f}, angle: {params[2]:.2f}°")
    
    # Application de la transformation
    img_registered = apply_transformation(img_mov, params)
    
    # Affichage
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title('Image de référence')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB))
    plt.title('Image mobile')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB))
    plt.title('Image recalée')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

"""
# Optimisation SSD pour le recalage d'images
    result = minimize(
        # 1. FONCTION À MINIMISER
        ssd_cost,                    # Notre fonction qui calcule le coût SSD
                                     # minimize() va appeler cette fonction avec 
                                     # différentes valeurs de paramètres jusqu'à
                                     # trouver celles qui donnent le plus petit coût
        
        # 2. POINT DE DÉPART  
        initial_params,              # [0, 0, 0] = pas de translation, pas de rotation
                                     # L'algorithme commence ici et explore autour
                                     # Peut influencer la vitesse de convergence
        
        # 3. ARGUMENTS SUPPLÉMENTAIRES
        args=(img_ref, img_mov),     # Ces arguments sont passés à ssd_cost() après params
                                     # Équivalent à : ssd_cost(params, img_ref, img_mov)
                                     # Permet de "figer" les images pendant l'optimisation
        
        # 4. ALGORITHME D'OPTIMISATION
        method='Powell',             # Algorithme de Powell (sans gradient)
                                     # AVANTAGES :
                                     # - Pas besoin de calculer les dérivées
                                     # - Robuste pour fonctions "bruyantes"
                                     # - Bon pour 2-3 paramètres
                                     # ALTERNATIVES :
                                     # - 'Nelder-Mead' : simplex, robuste
                                     # - 'BFGS' : plus rapide mais besoin gradient
        
        # 5. OPTIONS DE CONTRÔLE
        options={'maxiter': 1000}    # Maximum 1000 évaluations de la fonction
                                     # Sécurité pour éviter boucles infinies
                                     # En pratique, converge en 50-200 évaluations 

                                    #
                                    
    # ***********QUE FAIT minimize() EN INTERNE ? (Algorithme Powell simplifié)************
    Étapes approximatives de l'algorithme Powell :
    
    1. Départ : params = [0, 0, 0]
    2. Phase 1 - Optimisation de tx :
       - Teste : [-1, 0, 0] → SSD = 1250
       - Teste : [+1, 0, 0] → SSD = 1100  ← Mieux !
       - Teste : [+2, 0, 0] → SSD = 950   ← Encore mieux !
       - Teste : [+3, 0, 0] → SSD = 980   ← Pire, stop
       - Meilleur tx ≈ 2
    
    3. Phase 2 - Optimisation de ty (tx fixé à 2) :
       - Teste : [2, -1, 0] → SSD = 920
       - Teste : [2, +1, 0] → SSD = 800   ← Mieux !
       - Continue jusqu'à trouver meilleur ty
    
    4. Phase 3 - Optimisation angle (tx, ty fixés) :
       - Teste différents angles...
       - Trouve le meilleur
    
    5. Répète les phases jusqu'à convergence
       (quand l'amélioration devient négligeable)
    
    Résultat : Solution optimale en ~100 tests au lieu de 20,000+ !
"""