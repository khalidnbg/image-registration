import numpy as np
import matplotlib.pyplot as plt

# Paramètres réels
tx_real, ty_real = 15.55, 20.60
angle_real = 35
scale_real = 1.25

# Méthodes et estimations
methods = {
    "SSD": {"tx": 17.55, "ty": 0.94, "angle": 5.31, "scale": 0.86},
    "NCC": {"tx": 0.68, "ty": 20.73, "angle": 34.98, "scale": 0.80},
    "MI":  {"tx": 10.00, "ty": 15.00, "angle": 30.00, "scale": 1.20},
    "PhaseCorr": {"tx": -158.95, "ty": 642.05, "angle": 34.9922, "scale": 1.2356},
    "SIFT+RANSAC": {"tx": 15.7, "ty": 20.4, "angle": 35.00, "scale": 1.2499}
}

parameters = ["tx", "ty", "angle", "scale"]

# Préparer les données pour le tracé
fig, axs = plt.subplots(2, 2, figsize=(12,10))
axs = axs.flatten()

for i, param in enumerate(parameters):
    real_value = eval(f"{param}_real") if param != "scale" else scale_real
    values = [methods[m][param] for m in methods]
    axs[i].bar(methods.keys(), values, color='skyblue', alpha=0.7)
    axs[i].axhline(real_value, color='red', linestyle='--', label='Valeur réelle')
    axs[i].set_title(f"Comparaison de {param}")
    axs[i].set_ylabel(param)
    axs[i].legend()
    axs[i].grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Comparaison des méthodes de recalage pour translation, rotation et échelle", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
