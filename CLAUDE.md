# Mejores Prácticas para Programar en Jupyter Notebooks

## Estructura y Organización

### 1. Orden Lógico de las Celdas
- Coloca las importaciones al inicio del notebook
- Organiza el código en un flujo descendente: configuración → carga de datos → procesamiento → análisis → visualización
- Cada celda debe tener un propósito claro y atómico

### 2. Uso de Celdas Markdown
- Documenta cada sección importante con celdas Markdown
- Utiliza encabezados jerárquicos (# ## ###) para estructurar el notebook
- Incluye explicaciones del "por qué" no solo del "qué"
- Añade enlaces a documentación relevante cuando sea apropiado

## Código Limpio y Mantenible

### 3. Tamaño de las Celdas
- Mantén las celdas pequeñas y enfocadas (5-15 líneas idealmente)
- Evita celdas con más de 50 líneas de código
- Si una celda es muy larga, considera dividirla en funciones reutilizables

### 4. Nombres Descriptivos
- Usa nombres de variables claros y descriptivos
- Evita nombres de una sola letra excepto en bucles simples o convenciones matemáticas
- Prefiere `user_data` sobre `ud` o `data1`

### 5. Funciones y Modularidad
- Extrae lógica repetitiva en funciones
- Define funciones complejas en módulos externos (.py) e impórtalas
- Las funciones deben hacer una cosa y hacerla bien

## Estado y Reproducibilidad

### 6. Ejecución Secuencial
- El notebook debe poder ejecutarse de arriba hacia abajo sin errores
- Evita dependencias entre celdas no secuenciales
- Reinicia el kernel y ejecuta todo el notebook regularmente para verificar reproducibilidad

### 7. Manejo de Estado
- Evita modificar variables globales en múltiples lugares
- No dependas del orden de ejecución no lineal
- Si necesitas re-ejecutar celdas, asegúrate de que sean idempotentes

### 8. Seeds Aleatorias
- Establece seeds para reproducibilidad cuando uses operaciones aleatorias
```python
import numpy as np
import random
np.random.seed(42)
random.seed(42)
```

## Datos y Archivos

### 9. Rutas Relativas
- Usa rutas relativas al directorio del proyecto
- Utiliza `pathlib.Path` para manejo de rutas multiplataforma
```python
from pathlib import Path
data_path = Path(__file__).parent / "data" / "dataset.csv"
```

### 10. No Incluir Datos Grandes
- No versiones datos grandes en el notebook
- Carga datos desde archivos externos
- Documenta dónde obtener los datos si no están incluidos

## Visualización

### 11. Gráficos Claros
- Siempre incluye títulos, etiquetas de ejes y leyendas
- Usa tamaños de figura apropiados
```python
plt.figure(figsize=(10, 6))
plt.title("Título Descriptivo")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
```

### 12. Limita el Output
- Usa `head()`, `tail()`, o `sample()` al mostrar DataFrames
- Limita el número de gráficos generados en bucles
- Considera usar `%%capture` para suprimir outputs innecesarios

## Control de Versiones

### 13. Limpia los Outputs
- Limpia todos los outputs antes de hacer commit (opcional pero recomendado)
- Considera usar herramientas como `nbstripout`
- Esto reduce el tamaño del archivo y evita conflictos de merge

### 14. .gitignore Apropiado
```gitignore
.ipynb_checkpoints/
__pycache__/
.venv/
*.pyc
```

## Performance y Debugging

### 15. Monitoreo de Memoria
- Ten cuidado con DataFrames muy grandes
- Libera memoria cuando sea necesario con `del variable`
- Usa `%whos` para ver variables en memoria

### 16. Time Profiling
```python
%%time
# Código a medir (una ejecución)

%%timeit
# Código a medir (múltiples ejecuciones)
```

### 17. Debugging
- Usa `%pdb` para debugging automático
- Inserta `breakpoint()` o usa `%debug` después de un error
- Utiliza `print()` estratégicamente o logging apropiado

## Seguridad

### 18. No Incluir Credenciales
- Nunca incluyas API keys, passwords o tokens en el notebook
- Usa variables de entorno o archivos de configuración
```python
import os
api_key = os.getenv("API_KEY")
```

### 19. Sanitiza Outputs
- Revisa que los outputs no expongan información sensible
- Ten cuidado al compartir notebooks que accedan a datos privados

## Colaboración

### 20. README y Documentación
- Incluye un README.md explicando:
  - Propósito del notebook
  - Dependencias y cómo instalarlas
  - Cómo ejecutar el notebook
  - Estructura del proyecto

### 21. Requirements
- Mantén actualizado el archivo de dependencias
- Con uv, el `pyproject.toml` gestiona las dependencias automáticamente

## Magic Commands Útiles

```python
%matplotlib inline        # Gráficos inline
%load_ext autoreload     # Auto-recarga de módulos
%autoreload 2            # Recarga módulos antes de ejecutar
%config InlineBackend.figure_format = 'retina'  # Gráficos alta resolución
```

## Ejemplo de Estructura Ideal

```
# Título del Notebook

## 1. Introducción
[Markdown: Descripción del objetivo]

## 2. Setup
[Código: Importaciones]
[Código: Configuración]

## 3. Carga de Datos
[Markdown: Descripción de los datos]
[Código: Carga]
[Código: Exploración inicial]

## 4. Análisis Exploratorio
[Markdown: Pregunta a responder]
[Código: Análisis]
[Markdown: Interpretación]

## 5. Modelado (si aplica)
[Código: Preparación]
[Código: Entrenamiento]
[Código: Evaluación]

## 6. Conclusiones
[Markdown: Resultados y próximos pasos]
```

## Recursos Adicionales

- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/)
- [Google Colab Best Practices](https://colab.research.google.com/)
- PEP 8 para estilo de código Python
