from PIL import Image
from PIL.ExifTags import TAGS

def extraire_info_image(chemin_image):
    # Ouvrir l'image
    image = Image.open(chemin_image)
    
    # Extraire les métadonnées EXIF
    exif_data = image._getexif()
    
    if exif_data is not None:
        info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            info[tag] = value
        return info
    else:
        return "Aucune métadonnée EXIF trouvée"

# Exemple d'utilisation
chemin = 'images/brain.jpg'  # Remplacez par le chemin de votre image
infos = extraire_info_image(chemin)
for cle, valeur in infos.items():
    print(f"{cle} : {valeur}")
