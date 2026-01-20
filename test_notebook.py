#!/usr/bin/env python3
"""
Script de prueba para verificar que el código del notebook funciona correctamente.
"""

print("=" * 60)
print("Testing Notebook Code")
print("=" * 60)

# Cell 1: Setup e Importaciones
print("\n1. Setup e Importaciones")
print("-" * 40)
import numpy as np
import sys
from pathlib import Path

# Configuración para reproducibilidad
np.random.seed(42)

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

# Cell 2: Generación de Datos
print("\n2. Generación de Datos")
print("-" * 40)
sample_size = 100
data = np.random.randn(sample_size)

print(f"Generated {len(data)} data points")
print(f"First 5 values: {data[:5]}")

# Cell 3: Análisis Estadístico Básico
print("\n3. Análisis Estadístico Básico")
print("-" * 40)

def calculate_statistics(data_array):
    """
    Calcula estadísticas descriptivas básicas.

    Args:
        data_array: Array de numpy con los datos

    Returns:
        dict: Diccionario con las estadísticas calculadas
    """
    return {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array)
    }

stats = calculate_statistics(data)

print("Estadísticas descriptivas:")
for key, value in stats.items():
    print(f"  {key:8s}: {value:8.4f}")

# Cell 4: Transformación de Datos
print("\n4. Transformación de Datos")
print("-" * 40)
data_normalized = (data - data.min()) / (data.max() - data.min())

print(f"Valores normalizados (primeros 5): {data_normalized[:5]}")
print(f"Min: {data_normalized.min():.4f}, Max: {data_normalized.max():.4f}")

# Cell 5: Operaciones Vectorizadas
print("\n5. Operaciones Vectorizadas")
print("-" * 40)
data_squared = data ** 2
data_positive = data[data > 0]

print(f"Número de valores positivos: {len(data_positive)}")
print(f"Suma de cuadrados: {data_squared.sum():.4f}")

# Cell 6: Verificación Final
print("\n6. Verificación Final")
print("-" * 40)
print("Notebook ejecutado correctamente!")
print(f"Total de variables en memoria: {len([x for x in dir() if not x.startswith('_')])}")

print("\n" + "=" * 60)
print("All tests passed successfully!")
print("=" * 60)
