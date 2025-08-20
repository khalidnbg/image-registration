import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from skimage import data, transform

# ================================
# Fonction de coût SSD
# ================================
def ssd_cost(params, img_ref, img_mov, mask=None):
    tx, ty, angle = params

    # Matrice de transformation
    M = cv2.getRotationMatrix2D((img_mov.shape[1] // 2, img_mov.shape[0] // 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty

    img_transformed = cv2.warpAffine(
        img_mov, M, (img_ref.shape[1], img_ref.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    if mask is not None:
        diff = (img_ref - img_transformed) * mask
    else:
        diff = img_ref - img_transformed

    return np.sum(diff ** 2)

# ================================
# Recalage par SSD
# ================================
def register_images_ssd(img_ref, img_mov, initial_params=[0.0, 0.0, 0.0]):
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)

    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0

    scales = [0.25, 0.5, 1.0]
    params = np.array(initial_params, dtype=np.float64)

    for scale in scales:
        if scale != 1.0:
            ref_scaled = cv2.resize(img_ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mov_scaled = cv2.resize(img_mov, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            ref_scaled = img_ref
            mov_scaled = img_mov

        params_scaled = params.copy()
        params_scaled[:2] *= scale

        starts = [
            params_scaled,
            params_scaled + [5.0, 5.0, 10.0],
            params_scaled + [-5.0, -5.0, -10.0],
            params_scaled + [10.0, 0.0, 15.0],
            params_scaled + [0.0, 10.0, -15.0]
        ]

        best_params = params_scaled
        best_ssd = np.inf

        for start in starts:
            try:
                result = minimize(
                    ssd_cost, start,
                    args=(ref_scaled, mov_scaled),
                    method='Powell',
                    options={'maxiter': 500, 'ftol': 1e-6}
                )
                if result.success and result.fun < best_ssd:
                    best_ssd = result.fun
                    best_params = result.x
            except:
                continue

        params[:2] = best_params[:2] / scale
        params[2] = best_params[2]

    return params

# ================================
# Appliquer la transformation
# ================================
def apply_transformation(img, params):
    tx, ty, angle = params
    M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# ================================
# Calcul du SSD
# ================================
def calculate_ssd_score(img_ref, img_mov):
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    return np.sum((img_ref - img_mov) ** 2)

# ================================
# Création d'images de test
# ================================
def create_test_images():
    original = data.camera().astype(np.float32) / 255.0
    original_shape = original.shape

    # Redimensionnement de l’image (-30%) mais centrée dans même taille
    scale_factor = 0.7
    resized_h = int(original_shape[0] * scale_factor)
    resized_w = int(original_shape[1] * scale_factor)
    resized = transform.resize(original, (resized_h, resized_w), anti_aliasing=True)

    ref_image = np.zeros(original_shape)
    start_y = (original_shape[0] - resized_h) // 2
    start_x = (original_shape[1] - resized_w) // 2
    ref_image[start_y:start_y+resized_h, start_x:start_x+resized_w] = resized

    # Appliquer rotation + translation
    true_angle = 10
    true_tx, true_ty = 55, 10
    tform = transform.SimilarityTransform(
        rotation=np.radians(true_angle),
        translation=(true_tx, true_ty)
    )
    transformed = transform.warp(ref_image, tform.inverse, output_shape=original_shape)

    true_params = {
        "angle": true_angle,
        "tx": true_tx,
        "ty": true_ty
    }

    return ref_image, transformed, true_params

# ================================
# Exemple d'utilisation
# ================================
if __name__ == "__main__":
    img_ref, img_mov, true_params = create_test_images()

    # Conversion float -> 8bit pour OpenCV
    img_ref_cv = (img_ref * 255).astype(np.uint8)
    img_mov_cv = (img_mov * 255).astype(np.uint8)

    ssd_before = calculate_ssd_score(img_ref_cv, img_mov_cv)
    print(f"SSD avant recalage: {ssd_before:.2f}")

    params = register_images_ssd(img_ref_cv, img_mov_cv)
    print(f"Paramètres trouvés - tx: {params[0]:.2f}, ty: {params[1]:.2f}, angle: {params[2]:.2f}°")
    print(f"Paramètres réels - {true_params}")

    img_registered = apply_transformation(img_mov_cv, params)

    ssd_after = calculate_ssd_score(img_ref_cv, img_registered)
    print(f"SSD après recalage: {ssd_after:.2f}")
    print(f"Amélioration: {ssd_before - ssd_after:.2f}")

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(img_ref, cmap="gray")
    plt.title("Référence")
    plt.subplot(132)
    plt.imshow(img_mov, cmap="gray")
    plt.title(f"Mobile (SSD: {ssd_before:.2f})")
    plt.subplot(133)
    plt.imshow(img_registered, cmap="gray")
    plt.title(f"Recalée (SSD: {ssd_after:.2f})")
    plt.show()
