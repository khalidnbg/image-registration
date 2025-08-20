# Importation des bibliothèques nécessaires
import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Fonctions MI avec rotation centrée corrigée ---
def calculate_histogram_2d(img1, img2, bins=256):
    img1_int = (img1 * (bins-1)).astype(np.int32)
    img2_int = (img2 * (bins-1)).astype(np.int32)
    hist_joint, _, _ = np.histogram2d(img1_int.flatten(), img2_int.flatten(),
                                      bins=bins, range=[[0,bins-1],[0,bins-1]])
    return hist_joint

def mutual_information_manual(img1, img2, bins=64):
    hist_joint = calculate_histogram_2d(img1, img2, bins)
    prob_joint = hist_joint / np.sum(hist_joint)
    prob_img1 = np.sum(prob_joint, axis=1)
    prob_img2 = np.sum(prob_joint, axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if prob_joint[i,j] > 0 and prob_img1[i] > 0 and prob_img2[j] > 0:
                mi += prob_joint[i,j] * np.log(prob_joint[i,j] / (prob_img1[i]*prob_img2[j]))
    return mi

def mi_cost(params, img_ref, img_mov):
    tx, ty, angle = params
    center = (img_mov.shape[1] / 2.0, img_mov.shape[0] / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0,2] += tx
    M[1,2] += ty
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return -mutual_information_manual(img_ref, img_transformed, bins=64)

def coarse_rotation_search(img_ref, img_mov, step=10):
    """Recherche grossière de la rotation optimale par pas de 'step' degrés."""
    best_angle = 0
    best_mi = -np.inf
    for angle in range(-90, 91, step):
        M = cv2.getRotationMatrix2D((img_mov.shape[1]/2, img_mov.shape[0]/2), angle, 1.0)
        img_rot = cv2.warpAffine(img_mov, M, (img_mov.shape[1], img_mov.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mi = mutual_information_manual(img_ref, img_rot, bins=64)
        if mi > best_mi:
            best_mi = mi
            best_angle = angle
    return best_angle

def register_images_mi(img_ref, img_mov, initial_params=[0,0,0]):
    if len(img_ref.shape) == 3: img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3: img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32)/255.0
    img_mov = img_mov.astype(np.float32)/255.0
    
    # Recherche grossière pour initialiser l'angle
    coarse_angle = coarse_rotation_search(img_ref, img_mov, step=10)
    starts = [
        [0,0,coarse_angle],
        [0,0,coarse_angle+5],
        [0,0,coarse_angle-5]
    ]
    
    best_params = initial_params
    best_mi = -np.inf
    
    for start in starts:
        try:
            result = minimize(mi_cost, start, args=(img_ref, img_mov), method='Powell', options={'maxiter':500})
            current_mi = -result.fun
            if result.success and current_mi > best_mi:
                best_mi = current_mi
                best_params = result.x
        except:
            continue
    
    return best_params

def apply_transformation(img, params):
    tx, ty, angle = params
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0,2] += tx
    M[1,2] += ty
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def create_rotated_image_with_centered_rotation(img, angle_degrees):
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    img_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img_rotated

def calculate_mi_score(img_ref, img_mov):
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    return mutual_information_manual(img_ref, img_mov, bins=64)

# --- Test ---
if __name__ == "__main__":
    img_ref = cv2.imread('results/brain.jpg')
    if img_ref is None:
        print("Image non trouvée, création d'une image de test...")
        img_ref = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img_ref, (50, 50), (200, 200), (255, 0, 0), -1)
        cv2.circle(img_ref, (128, 128), 40, (0, 255, 0), -1)
        cv2.putText(img_ref, 'TEST', (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    rotations_reelles = [15, 35, -15.5, 40.0, -20.0]  
    rotations_estimées, erreurs, mi_scores = [], [], []

    for angle in rotations_reelles:
        img_mov = create_rotated_image_with_centered_rotation(img_ref, angle)
        mi_avant = calculate_mi_score(img_ref, img_mov)
        params_est = register_images_mi(img_ref, img_mov)
        rotation_estimee = params_est[2]
        rotations_estimées.append(rotation_estimee)
        erreurs.append(abs(rotation_estimee - angle))
        img_recalee = apply_transformation(img_mov, params_est)
        mi_apres = calculate_mi_score(img_ref, img_recalee)
        mi_scores.append((mi_avant, mi_apres))
        print(f"Rotation réelle={angle:.2f}°, estimée={rotation_estimee:.2f}°, erreur={abs(rotation_estimee-angle):.2f}°")

