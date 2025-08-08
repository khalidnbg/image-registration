from PIL import Image

# Charger l'image
img = Image.open("images/tisdrin.jpg").convert("RGB")

# Obtenir toutes les couleurs uniques
couleurs_uniques = img.getcolors(maxcolors=img.width * img.height)

# Afficher le nombre de couleurs réellement utilisées
if couleurs_uniques is not None:
    print(f"Nombre de couleurs utilisées : {len(couleurs_uniques)}")
else:
    print("Trop de couleurs pour être comptées directement.")
