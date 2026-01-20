# Jupyter Claude

> Proyecto de Jupyter Notebooks configurado con uv y Python 3.12, diseñado para proyectos de Machine Learning end-to-end con mejores prácticas integradas.

**Este README es para ti, el desarrollador**. Si estás trabajando con Claude Code en este proyecto, ten en cuenta que Claude tiene su propia guía técnica en `CLAUDE.md` que usa como referencia interna.

## ¿Por qué este proyecto?

Este repositorio es más que un simple setup de Jupyter Notebooks. Es un marco de trabajo completo para desarrollar proyectos de **Machine Learning de principio a fin**, desde la exploración inicial de datos hasta el deployment de modelos en producción.

## Características

- **Python 3.12** - La última versión estable
- **uv** - Gestión de dependencias ultrarrápida
- **Jupyter Notebook/Lab** - Entorno completamente configurado
- **Estructura de proyecto ML** - Organización profesional para proyectos end-to-end
- **Guía completa de mejores prácticas** - En `CLAUDE.md` para Claude y este README para ti
- **Ejemplos prácticos** - Código listo para usar y adaptar

## Requisitos

- [uv](https://github.com/astral-sh/uv) instalado en tu sistema

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/TU_USUARIO/jupyter_claude.git
cd jupyter_claude
```

2. Las dependencias se instalarán automáticamente cuando ejecutes cualquier comando con `uv run`:
```bash
uv sync
```

## Uso

### Iniciar Jupyter Notebook

```bash
uv run jupyter notebook
```

### Iniciar JupyterLab

```bash
uv run jupyter lab
```

### Ejecutar el notebook de ejemplo

```bash
# Como script Python
uv run python test_notebook.py

# O abre ejemplo.ipynb en Jupyter
uv run jupyter notebook ejemplo.ipynb
```

## Estructura del Proyecto

```
jupyter_claude/
├── CLAUDE.md              # Guía de mejores prácticas para Jupyter Notebooks
├── README.md              # Este archivo
├── ejemplo.ipynb          # Notebook de ejemplo
├── test_notebook.py       # Script de prueba del notebook
├── pyproject.toml         # Configuración del proyecto y dependencias
├── uv.lock                # Lock file de dependencias
└── .gitignore             # Archivos ignorados por git
```

---

## 🚀 Proyectos de Machine Learning End-to-End

Este proyecto está optimizado para desarrollar proyectos completos de ML siguiendo las mejores prácticas de la industria.

### El Pipeline de ML

Un proyecto de Machine Learning profesional sigue este flujo:

```
1. Definición del Problema
   ↓
2. Recopilación de Datos
   ↓
3. Análisis Exploratorio (EDA)
   ↓
4. Preparación de Datos
   ↓
5. Feature Engineering
   ↓
6. Modelado
   ↓
7. Evaluación
   ↓
8. Deployment
   ↓
9. Monitoreo
```

### Estructura Recomendada para Proyectos ML

Cuando empieces un nuevo proyecto de ML, organiza tu directorio así:

```
mi_proyecto_ml/
├── data/
│   ├── raw/              # Datos originales (¡nunca modificar!)
│   ├── interim/          # Datos en proceso de transformación
│   ├── processed/        # Datos finales listos para modelar
│   └── external/         # Datos de fuentes externas
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Análisis Exploratorio
│   ├── 02_preprocessing.ipynb          # Limpieza y Preparación
│   ├── 03_feature_engineering.ipynb    # Creación de Features
│   ├── 04_modeling.ipynb               # Entrenamiento de Modelos
│   └── 05_evaluation.ipynb             # Evaluación y Comparación
│
├── src/                  # Código de producción (no notebooks)
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/
│       └── visualize.py
│
├── models/               # Modelos entrenados (.pkl, .h5, etc.)
├── reports/              # Análisis generados
│   └── figures/          # Gráficos y visualizaciones
├── config/               # Archivos de configuración
├── tests/                # Tests unitarios
├── pyproject.toml        # Dependencias (gestionadas por uv)
└── README.md             # Documentación del proyecto
```

### Fases del Proyecto ML

#### Fase 1: Definición del Problema

**Lo primero es lo primero**: ¿Qué problema estás resolviendo?

- ¿Es un problema de clasificación o regresión?
- ¿Qué métricas usarás para medir el éxito?
- ¿Cuál es el baseline (rendimiento sin ML)?
- ¿Qué constraints tienes? (latencia, recursos, interpretabilidad)

**Tip**: Documenta esto en un notebook `00_problem_definition.ipynb` antes de empezar.

#### Fase 2: Análisis Exploratorio de Datos (EDA)

Esta es la fase más importante. Aquí descubres:

- Forma y tamaño de tus datos
- Tipos de variables
- Valores faltantes y su patrón
- Distribución de las variables
- Outliers y anomalías
- Correlaciones entre variables
- Desbalanceo de clases (en clasificación)

**Checklist rápido de EDA**:
- [ ] Verificar dimensiones del dataset
- [ ] Revisar tipos de datos
- [ ] Analizar valores faltantes
- [ ] Detectar duplicados
- [ ] Visualizar distribuciones
- [ ] Calcular correlaciones
- [ ] Identificar outliers

#### Fase 3: Preparación de Datos

Los datos nunca vienen perfectos. Necesitarás:

**Manejo de valores faltantes**:
- Imputación por media/mediana (variables numéricas)
- Imputación por moda (variables categóricas)
- KNN Imputation (más sofisticado)
- Eliminación (si es < 5% de los datos)

**Tratamiento de outliers**:
- Detección por IQR o Z-score
- Capping (limitar a percentiles)
- Transformación logarítmica
- Eliminación justificada

**Encoding de variables categóricas**:
- One-Hot Encoding (para variables nominales)
- Label Encoding (para variables ordinales)
- Target Encoding (cuando hay muchas categorías)
- Frequency Encoding

**Escalado de features**:
- StandardScaler: para distribuciones normales
- MinMaxScaler: para rangos específicos [0,1]
- RobustScaler: cuando hay outliers

#### Fase 4: Feature Engineering

Aquí es donde diferencias un proyecto básico de uno profesional:

**Creación de features**:
- Features polinomiales (x², x³)
- Interacciones entre variables (x₁ × x₂)
- Features de datetime (año, mes, día de semana)
- Agregaciones por grupos (media por categoría)
- Domain-specific features (basadas en conocimiento del negocio)

**Selección de features**:
- Correlación con el target
- Importancia por Random Forest
- Recursive Feature Elimination (RFE)
- SelectKBest con diferentes scores

**Por qué importa**: Más features ≠ mejor modelo. Features irrelevantes añaden ruido y overfitting.

#### Fase 5: Modelado

**Empieza simple, luego complejiza**:

1. **Baseline Model**: Siempre empieza con un modelo simple (DummyClassifier, media, etc.)
2. **Modelos Lineales**: Logistic/Linear Regression
3. **Árboles**: Decision Trees, Random Forest
4. **Boosting**: XGBoost, LightGBM, CatBoost
5. **Deep Learning**: Solo si tienes suficientes datos y recursos

**División de datos**:
```python
# Train-Validation-Test split
Train (70%): Entrenar el modelo
Validation (15%): Tuning de hiperparámetros
Test (15%): Evaluación final (tocar solo una vez)
```

**Hyperparameter Tuning**:
- Grid Search: exhaustivo pero lento
- Random Search: más rápido, buena exploración
- Optuna/Bayesian Optimization: lo más eficiente

**Cross-Validation**:
- K-Fold: para datasets balanceados
- Stratified K-Fold: para clasificación desbalanceada
- Time Series Split: para datos temporales

#### Fase 6: Evaluación

**Para Clasificación**:
- Accuracy: solo si las clases están balanceadas
- Precision/Recall: cuando el costo de FP y FN difiere
- F1-Score: balance entre precision y recall
- ROC-AUC: rendimiento general del clasificador
- Confusion Matrix: entender qué se está confundiendo

**Para Regresión**:
- MAE (Mean Absolute Error): interpretable, en mismas unidades
- RMSE (Root Mean Squared Error): penaliza errores grandes
- R²: qué % de varianza explica el modelo
- MAPE: error porcentual (cuidado con divisiones por 0)

**Interpretabilidad**:
- SHAP values: impacto de cada feature
- LIME: explicaciones locales
- Feature importance: qué features son más importantes

#### Fase 7: Deployment

Tu modelo necesita servir predicciones en el mundo real:

**Serialización**:
```python
import joblib
joblib.dump(model, 'model.pkl')  # Guardar
model = joblib.load('model.pkl')  # Cargar
```

**API de Predicción** (con FastAPI):
```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": prediction[0]}
```

**Dockerización**:
```dockerfile
FROM python:3.12
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

#### Fase 8: Monitoreo

El trabajo no termina en deployment:

- **Data Drift**: ¿Han cambiado las distribuciones de entrada?
- **Concept Drift**: ¿Ha cambiado la relación X→y?
- **Performance Monitoring**: ¿El modelo sigue siendo preciso?
- **Reentrenamiento**: Estrategia para actualizar el modelo

### Herramientas Esenciales por Fase

| Fase | Librerías Clave |
|------|----------------|
| EDA | pandas, numpy, matplotlib, seaborn, plotly |
| Preprocessing | scikit-learn, pandas |
| Feature Engineering | feature-engine, category_encoders |
| Modelado | scikit-learn, xgboost, lightgbm, catboost |
| Deep Learning | tensorflow, pytorch, keras |
| Interpretabilidad | shap, lime, eli5 |
| Tracking | mlflow, wandb |
| Deployment | fastapi, docker, kubernetes |
| Monitoreo | evidently, whylabs |

### Checklist de Proyecto Completo

**Antes de considerar tu proyecto "terminado"**, asegúrate de haber:

- [ ] Definido claramente el problema y las métricas
- [ ] Realizado EDA exhaustivo
- [ ] Manejado valores faltantes y outliers
- [ ] Creado y seleccionado features relevantes
- [ ] Probado múltiples modelos
- [ ] Realizado cross-validation
- [ ] Evaluado con conjunto de test independiente
- [ ] Comparado contra baseline
- [ ] Analizado interpretabilidad
- [ ] Serializado el modelo y pipeline completo
- [ ] Documentado todo el proceso
- [ ] Creado API de predicción (si aplica)
- [ ] Configurado monitoreo (para producción)

---

## Mejores Prácticas para Jupyter Notebooks

Consulta el archivo `CLAUDE.md` para una guía completa técnica. Aquí un resumen ejecutivo:

### Organización
- Importaciones siempre al inicio
- Flujo de arriba hacia abajo
- Una idea por celda
- Usa Markdown para documentar

### Reproducibilidad
- Establece seeds aleatorias (`np.random.seed(42)`)
- El notebook debe poder ejecutarse de arriba hacia abajo sin errores
- Reinicia el kernel regularmente para verificar

### Código Limpio
- Nombres de variables descriptivos
- Funciones para lógica repetitiva
- Celdas de máximo 50 líneas
- Extrae código complejo a módulos .py

### Visualización
- Siempre títulos, etiquetas y leyendas
- Usa `head()`, `tail()` para DataFrames grandes
- Configura tamaños de figura apropiados

### Control de Versiones
- Añade notebooks a git
- Considera limpiar outputs antes de commit
- Usa `.gitignore` apropiado (incluido en este proyecto)

## Dependencias Principales

- `jupyter` - Jupyter Notebook
- `notebook` - Interfaz web de Jupyter
- `ipykernel` - Kernel de IPython
- `numpy` - Computación numérica

Para ver todas las dependencias, consulta `pyproject.toml`.

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## Autor

Creado con Claude Code
