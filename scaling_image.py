import cv2
original = cv2.imread('images/image_test.jpg')
if original is not None:
    scaled = cv2.resize(original, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('images/zoomed_test_image.jpg', scaled)
    print(f"Nouvelle taille de zoomed_test_image.jpg: {scaled.shape}")
else:
    print("Erreur : Impossible de charger image_test.jpg")