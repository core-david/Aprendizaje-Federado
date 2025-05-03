import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import os

# Cargar la base de datos MNIST
print("Cargando el conjunto de datos MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X.astype('float32')
X /= 255.0  # Normalizar valores de píxeles

# Convertir etiquetas a enteros si es necesario
y = y.astype(int)

# Mostrar la distribución original
original_distribution = Counter(y)
print(f"Distribución original de clases en MNIST: {original_distribution}")

# Utilizar StratifiedKFold para crear 5 divisiones manteniendo la proporción de clases
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Almacenar índices de cada pliegue
folds = []
for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
    folds.append(test_index)
    
    # Verificar distribución de este pliegue
    fold_distribution = Counter(y[test_index])
    print(f"Pliegue {fold_idx+1} - Distribución: {fold_distribution}")
    print(f"Pliegue {fold_idx+1} - Tamaño: {len(test_index)} ejemplos\n")

# Crear directorio para guardar los archivos CSV si no existe
output_dir = "mnist_splits"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Guardar cada pliegue en formato CSV
for i in range(5):
    X_fold = X.iloc[folds[i]]
    y_fold = y.iloc[folds[i]]
    
    # Combinar características y etiqueta en un solo DataFrame
    fold_df = X_fold.copy()
    fold_df['label'] = y_fold
    
    # Guardar en CSV
    csv_filename = os.path.join(output_dir, f'mnist_fold_{i+1}.csv')
    fold_df.to_csv(csv_filename, index=False)
    print(f"Guardando pliegue {i+1} en {csv_filename}: {X_fold.shape[0]} ejemplos")

print("\nVerificación final de balance:")
total = len(y)
print(f"Total de ejemplos en MNIST: {total}")

for i in range(5):
    fold_size = len(folds[i])
    print(f"Pliegue {i+1}: {fold_size} ejemplos ({fold_size/total*100:.2f}%)")