# Guía para Claude: Mejores Prácticas en Jupyter Notebooks y Machine Learning

> **NOTA PARA CLAUDE**: Este documento es tu referencia técnica para trabajar con Jupyter Notebooks y proyectos de Machine Learning. Contiene lineamientos estructurados, patrones de código y mejores prácticas que debes seguir cuando asistas a usuarios en este proyecto. El README.md es para lectura humana; este archivo es para ti.

---

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

---

# Fundamentos de Proyectos de Machine Learning End-to-End

## Visión General del Pipeline ML

Un proyecto de Machine Learning end-to-end sigue estas fases principales:

1. **Definición del Problema** → 2. **Recopilación de Datos** → 3. **EDA** → 4. **Preparación de Datos** → 5. **Feature Engineering** → 6. **Modelado** → 7. **Evaluación** → 8. **Deployment** → 9. **Monitoreo**

## 1. Definición del Problema

### Aspectos Clave a Documentar

```python
"""
PROBLEM DEFINITION TEMPLATE

Business Objective: [Descripción clara del objetivo de negocio]
ML Task Type: [Classification/Regression/Clustering/Ranking/etc.]
Success Metrics: [Métrica(s) que definen el éxito]
Baseline Performance: [Rendimiento actual sin ML o benchmark simple]
Constraints: [Latencia, recursos computacionales, interpretabilidad]
"""
```

### Clasificación de Problemas ML

- **Supervisado**: Classification (binary/multiclass), Regression
- **No Supervisado**: Clustering, Dimensionality Reduction, Anomaly Detection
- **Semi-supervisado**: Pocos datos etiquetados + muchos sin etiquetar
- **Reinforcement Learning**: Optimización de acciones secuenciales

### Métricas por Tipo de Problema

**Clasificación**:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix
- Log Loss, Brier Score

**Regresión**:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error), RMSE
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

**Clustering**:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

## 2. Recopilación y Carga de Datos

### Estructura de Directorios Recomendada

```
project_name/
├── data/
│   ├── raw/              # Datos originales (nunca modificar)
│   ├── interim/          # Datos intermedios transformados
│   ├── processed/        # Datos finales para modelado
│   └── external/         # Datos de fuentes externas
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── src/
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
├── models/               # Modelos entrenados serializados
├── reports/
│   └── figures/         # Gráficos generados
├── config/
│   └── config.yaml      # Configuración del proyecto
├── tests/
├── requirements.txt     # o pyproject.toml
└── README.md
```

### Código de Carga de Datos

```python
from pathlib import Path
import pandas as pd

# Usar rutas absolutas basadas en la estructura del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Carga datos raw sin modificaciones.

    Args:
        filename: Nombre del archivo en data/raw/

    Returns:
        DataFrame con los datos cargados
    """
    filepath = DATA_RAW / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    # Registrar metadata
    print(f"Cargando: {filepath}")
    print(f"Tamaño: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    return pd.read_csv(filepath)
```

## 3. Análisis Exploratorio de Datos (EDA)

### Checklist de EDA Sistemático

#### 3.1 Exploración Inicial

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
%config InlineBackend.figure_format = 'retina'

def initial_exploration(df: pd.DataFrame) -> None:
    """Exploración inicial sistemática del dataset."""

    print("=" * 60)
    print("EXPLORACIÓN INICIAL DEL DATASET")
    print("=" * 60)

    # 1. Forma del dataset
    print(f"\n📊 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # 2. Tipos de datos
    print("\n📋 Tipos de datos:")
    print(df.dtypes.value_counts())

    # 3. Memoria utilizada
    print(f"\n💾 Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 4. Valores faltantes
    print("\n❓ Valores faltantes:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing': missing[missing > 0],
        'Percent': missing_pct[missing > 0]
    }).sort_values('Percent', ascending=False)
    print(missing_df)

    # 5. Duplicados
    n_duplicates = df.duplicated().sum()
    print(f"\n🔄 Filas duplicadas: {n_duplicates} ({100*n_duplicates/len(df):.2f}%)")

    # 6. Vista previa
    print("\n👀 Primeras filas:")
    display(df.head())

    # 7. Estadísticas descriptivas
    print("\n📈 Estadísticas descriptivas (numéricas):")
    display(df.describe())

    print("\n📈 Estadísticas descriptivas (categóricas):")
    display(df.describe(include=['object', 'category']))
```

#### 3.2 Análisis de Variables Numéricas

```python
def analyze_numerical_features(df: pd.DataFrame, target: str = None) -> None:
    """Análisis completo de features numéricas."""

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in numerical_cols:
        numerical_cols.remove(target)

    for col in numerical_cols:
        print(f"\n{'='*60}\nAnálisis: {col}\n{'='*60}")

        # Estadísticas
        print(f"Min: {df[col].min():.2f}")
        print(f"Max: {df[col].max():.2f}")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Median: {df[col].median():.2f}")
        print(f"Std: {df[col].std():.2f}")
        print(f"Skewness: {df[col].skew():.2f}")
        print(f"Kurtosis: {df[col].kurtosis():.2f}")

        # Visualización
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Histograma
        df[col].hist(bins=50, ax=axes[0], edgecolor='black')
        axes[0].set_title(f'Distribución de {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frecuencia')

        # Boxplot
        df.boxplot(column=col, ax=axes[1])
        axes[1].set_title(f'Boxplot de {col}')

        # Q-Q plot
        from scipy import stats
        stats.probplot(df[col].dropna(), dist="norm", plot=axes[2])
        axes[2].set_title(f'Q-Q Plot de {col}')

        plt.tight_layout()
        plt.show()

        # Detección de outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"\n⚠️  Outliers (IQR method): {len(outliers)} ({100*len(outliers)/len(df):.2f}%)")
```

#### 3.3 Análisis de Variables Categóricas

```python
def analyze_categorical_features(df: pd.DataFrame, max_categories: int = 20) -> None:
    """Análisis de features categóricas."""

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        n_unique = df[col].nunique()
        print(f"\n{'='*60}\nAnálisis: {col}\n{'='*60}")
        print(f"Valores únicos: {n_unique}")

        if n_unique <= max_categories:
            # Value counts
            value_counts = df[col].value_counts()
            print(f"\nDistribución de valores:")
            print(value_counts)

            # Visualización
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts.plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_title(f'Distribución de {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frecuencia')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️  Demasiadas categorías ({n_unique}). Mostrando top 10:")
            print(df[col].value_counts().head(10))
```

#### 3.4 Análisis de Correlaciones

```python
def analyze_correlations(df: pd.DataFrame, target: str = None, threshold: float = 0.5) -> None:
    """Análisis de correlaciones entre variables."""

    # Matriz de correlación
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    plt.show()

    # Correlaciones fuertes
    if target:
        print(f"\n📊 Correlaciones con {target}:")
        target_corr = corr_matrix[target].sort_values(ascending=False)
        print(target_corr)

        # Visualizar correlaciones fuertes
        strong_corr = target_corr[abs(target_corr) > threshold].drop(target)
        if len(strong_corr) > 0:
            print(f"\n✅ Features con correlación > {threshold}:")
            print(strong_corr)

    # Multicolinealidad
    print(f"\n⚠️  Pares de features con alta correlación (>{threshold}):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"{feat1} ↔ {feat2}: {corr_val:.3f}")
```

## 4. Preparación de Datos

### 4.1 Manejo de Valores Faltantes

```python
from sklearn.impute import SimpleImputer, KNNImputer

class MissingValueHandler:
    """Estrategias para manejar valores faltantes."""

    @staticmethod
    def analyze_missing_pattern(df: pd.DataFrame) -> pd.DataFrame:
        """Analiza patrones de valores faltantes."""
        missing_df = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum(),
            'missing_pct': 100 * df.isnull().sum() / len(df),
            'dtype': df.dtypes
        })
        return missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)

    @staticmethod
    def impute_numerical(df: pd.DataFrame, columns: list, strategy: str = 'median') -> pd.DataFrame:
        """
        Imputa valores numéricos faltantes.

        Args:
            strategy: 'mean', 'median', 'most_frequent', 'constant', 'knn'
        """
        df_copy = df.copy()

        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=strategy)

        df_copy[columns] = imputer.fit_transform(df_copy[columns])
        return df_copy

    @staticmethod
    def impute_categorical(df: pd.DataFrame, columns: list, strategy: str = 'most_frequent') -> pd.DataFrame:
        """Imputa valores categóricos faltantes."""
        df_copy = df.copy()

        if strategy == 'mode':
            for col in columns:
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            for col in columns:
                df_copy[col].fillna('MISSING', inplace=True)
        else:
            imputer = SimpleImputer(strategy=strategy)
            df_copy[columns] = imputer.fit_transform(df_copy[columns])

        return df_copy
```

### 4.2 Detección y Manejo de Outliers

```python
class OutlierHandler:
    """Métodos para detectar y manejar outliers."""

    @staticmethod
    def detect_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.Series:
        """Detecta outliers usando el método IQR."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        return (df[column] < lower_bound) | (df[column] > upper_bound)

    @staticmethod
    def detect_zscore(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.Series:
        """Detecta outliers usando Z-score."""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold

    @staticmethod
    def cap_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
        """Capea outliers a percentiles."""
        df_copy = df.copy()

        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'percentile':
            lower_bound = df[column].quantile(0.01)
            upper_bound = df[column].quantile(0.99)

        df_copy[column] = df_copy[column].clip(lower_bound, upper_bound)
        return df_copy
```

### 4.3 Encoding de Variables Categóricas

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

class CategoricalEncoder:
    """Estrategias de encoding para variables categóricas."""

    @staticmethod
    def label_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Label encoding para variables ordinales."""
        df_copy = df.copy()
        le = LabelEncoder()

        for col in columns:
            df_copy[f'{col}_encoded'] = le.fit_transform(df_copy[col])

        return df_copy

    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: list, drop_first: bool = True) -> pd.DataFrame:
        """One-hot encoding para variables nominales."""
        return pd.get_dummies(df, columns=columns, drop_first=drop_first)

    @staticmethod
    def target_encode(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
        """Target encoding (mean encoding)."""
        df_copy = df.copy()

        # Calcular media del target por categoría
        target_means = df.groupby(column)[target].mean()
        df_copy[f'{column}_target_enc'] = df_copy[column].map(target_means)

        return df_copy

    @staticmethod
    def frequency_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Frequency encoding."""
        df_copy = df.copy()

        freq = df[column].value_counts(normalize=True)
        df_copy[f'{column}_freq_enc'] = df_copy[column].map(freq)

        return df_copy
```

### 4.4 Escalado de Features

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class FeatureScaler:
    """Diferentes métodos de escalado."""

    @staticmethod
    def standard_scale(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """StandardScaler: (x - μ) / σ"""
        df_copy = df.copy()
        scaler = StandardScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy, scaler

    @staticmethod
    def minmax_scale(df: pd.DataFrame, columns: list, feature_range=(0, 1)) -> pd.DataFrame:
        """MinMaxScaler: (x - min) / (max - min)"""
        df_copy = df.copy()
        scaler = MinMaxScaler(feature_range=feature_range)
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy, scaler

    @staticmethod
    def robust_scale(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """RobustScaler: Usa mediana y IQR, robusto a outliers"""
        df_copy = df.copy()
        scaler = RobustScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy, scaler
```

## 5. Feature Engineering

### 5.1 Creación de Features

```python
class FeatureEngineer:
    """Técnicas de feature engineering."""

    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
        """Crea features polinomiales."""
        from sklearn.preprocessing import PolynomialFeatures

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])

        feature_names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

        return pd.concat([df, poly_df], axis=1)

    @staticmethod
    def create_interaction_features(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
        """Crea features de interacción entre dos variables."""
        df_copy = df.copy()
        df_copy[f'{col1}_x_{col2}'] = df_copy[col1] * df_copy[col2]
        df_copy[f'{col1}_div_{col2}'] = df_copy[col1] / (df_copy[col2] + 1e-10)
        return df_copy

    @staticmethod
    def create_datetime_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Extrae features de columnas datetime."""
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])

        df_copy[f'{date_column}_year'] = df_copy[date_column].dt.year
        df_copy[f'{date_column}_month'] = df_copy[date_column].dt.month
        df_copy[f'{date_column}_day'] = df_copy[date_column].dt.day
        df_copy[f'{date_column}_dayofweek'] = df_copy[date_column].dt.dayofweek
        df_copy[f'{date_column}_quarter'] = df_copy[date_column].dt.quarter
        df_copy[f'{date_column}_is_weekend'] = df_copy[date_column].dt.dayofweek.isin([5, 6]).astype(int)

        return df_copy

    @staticmethod
    def create_aggregation_features(df: pd.DataFrame, group_col: str, agg_col: str, agg_funcs: list) -> pd.DataFrame:
        """Crea features agregadas por grupos."""
        df_copy = df.copy()

        for func in agg_funcs:
            agg_name = f'{agg_col}_{func}_by_{group_col}'
            df_copy[agg_name] = df_copy.groupby(group_col)[agg_col].transform(func)

        return df_copy
```

### 5.2 Selección de Features

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    """Métodos de selección de features."""

    @staticmethod
    def select_by_correlation(df: pd.DataFrame, target: str, threshold: float = 0.1) -> list:
        """Selecciona features por correlación con el target."""
        numerical_df = df.select_dtypes(include=[np.number])
        correlations = numerical_df.corr()[target].abs().sort_values(ascending=False)
        selected = correlations[correlations > threshold].index.tolist()
        selected.remove(target)
        return selected

    @staticmethod
    def select_by_variance(df: pd.DataFrame, threshold: float = 0.01) -> list:
        """Elimina features con baja varianza."""
        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)
        selected_columns = df.columns[selector.get_support()].tolist()
        return selected_columns

    @staticmethod
    def select_k_best(X, y, k: int = 10, score_func=f_classif) -> tuple:
        """Selecciona k mejores features."""
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)

        return X_selected, selected_features, scores

    @staticmethod
    def select_by_rfe(X, y, n_features: int = 10) -> list:
        """Recursive Feature Elimination."""
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X, y)

        selected_features = X.columns[selector.support_].tolist()
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')

        return selected_features, feature_ranking

    @staticmethod
    def select_by_importance(X, y, threshold: float = 0.01) -> tuple:
        """Selección basada en feature importance."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        selected = feature_importance[feature_importance['importance'] > threshold]['feature'].tolist()

        return selected, feature_importance
```

## 6. Modelado

### 6.1 Train-Test Split y Validación

```python
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit

class DataSplitter:
    """Estrategias de división de datos."""

    @staticmethod
    def simple_split(X, y, test_size: float = 0.2, random_state: int = 42):
        """Train-test split simple."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    @staticmethod
    def train_val_test_split(X, y, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42):
        """División en train, validation y test."""
        # Primero separar test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Luego separar train y validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def create_cv_splits(n_splits: int = 5, shuffle: bool = True, stratified: bool = True):
        """Crea splits para cross-validation."""
        if stratified:
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
        else:
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)

    @staticmethod
    def time_series_split(n_splits: int = 5):
        """Time series cross-validation."""
        return TimeSeriesSplit(n_splits=n_splits)
```

### 6.2 Baseline Models

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

class BaselineModels:
    """Modelos baseline para establecer rendimiento mínimo."""

    @staticmethod
    def dummy_classifier_baseline(X_train, X_test, y_train, y_test, strategy='most_frequent'):
        """Baseline usando DummyClassifier."""
        dummy = DummyClassifier(strategy=strategy, random_state=42)
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Dummy Classifier ({strategy}) Accuracy: {acc:.4f}")
        return dummy, acc

    @staticmethod
    def simple_logistic_regression(X_train, X_test, y_train, y_test):
        """Baseline con Logistic Regression simple."""
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Logistic Regression Accuracy: {acc:.4f}")
        return lr, acc

    @staticmethod
    def dummy_regressor_baseline(X_train, X_test, y_train, y_test, strategy='mean'):
        """Baseline usando DummyRegressor."""
        dummy = DummyRegressor(strategy=strategy)
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Dummy Regressor ({strategy}) RMSE: {rmse:.4f}")
        return dummy, rmse
```

### 6.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

class HyperparameterTuner:
    """Métodos para tuning de hiperparámetros."""

    @staticmethod
    def grid_search(model, param_grid, X_train, y_train, cv=5, scoring='accuracy'):
        """Grid search exhaustivo."""
        grid = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1, return_train_score=True
        )
        grid.fit(X_train, y_train)

        print(f"Best parameters: {grid.best_params_}")
        print(f"Best {scoring}: {grid.best_score_:.4f}")

        return grid.best_estimator_, grid.best_params_, grid.cv_results_

    @staticmethod
    def random_search(model, param_distributions, X_train, y_train, n_iter=100, cv=5, scoring='accuracy'):
        """Random search para exploración más amplia."""
        random = RandomizedSearchCV(
            model, param_distributions, n_iter=n_iter, cv=cv,
            scoring=scoring, n_jobs=-1, verbose=1, random_state=42,
            return_train_score=True
        )
        random.fit(X_train, y_train)

        print(f"Best parameters: {random.best_params_}")
        print(f"Best {scoring}: {random.best_score_:.4f}")

        return random.best_estimator_, random.best_params_, random.cv_results_
```

## 7. Evaluación de Modelos

### 7.1 Métricas de Clasificación

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationEvaluator:
    """Evaluación completa de modelos de clasificación."""

    @staticmethod
    def evaluate_classification(y_true, y_pred, y_pred_proba=None, labels=None):
        """Evaluación completa de clasificación."""

        print("=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=labels))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # ROC-AUC (solo si hay probabilidades)
        if y_pred_proba is not None:
            if len(np.unique(y_true)) == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True)
                plt.show()

                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])

                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.grid(True)
                plt.show()
```

### 7.2 Métricas de Regresión

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RegressionEvaluator:
    """Evaluación de modelos de regresión."""

    @staticmethod
    def evaluate_regression(y_true, y_pred, model_name='Model'):
        """Evaluación completa de regresión."""

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print("=" * 60)
        print(f"REGRESSION METRICS - {model_name}")
        print("=" * 60)
        print(f"MAE:   {mae:.4f}")
        print(f"MSE:   {mse:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"R²:    {r2:.4f}")
        print(f"MAPE:  {mape:.2f}%")

        # Residual Plot
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Predicted vs Actual
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Predicted vs Actual')
        axes[0].grid(True)

        # Residual Plot
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
```

### 7.3 Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate

class CrossValidator:
    """Cross-validation completo."""

    @staticmethod
    def perform_cv(model, X, y, cv=5, scoring='accuracy'):
        """Cross-validation simple."""
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        print(f"CV {scoring} scores: {scores}")
        print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores

    @staticmethod
    def perform_cv_multiple_metrics(model, X, y, cv=5, scoring_metrics=None):
        """Cross-validation con múltiples métricas."""
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring_metrics,
            return_train_score=True, n_jobs=-1
        )

        results_df = pd.DataFrame(cv_results)
        print("\nCross-Validation Results:")
        print(results_df.describe())

        return results_df
```

## 8. Serialización y Deployment

### 8.1 Guardado de Modelos

```python
import joblib
import pickle
from pathlib import Path

class ModelSerializer:
    """Guardar y cargar modelos."""

    @staticmethod
    def save_model(model, filepath: str, method: str = 'joblib'):
        """Guarda un modelo entrenado."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if method == 'joblib':
            joblib.dump(model, filepath)
        elif method == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)

        print(f"✅ Modelo guardado en: {filepath}")

    @staticmethod
    def load_model(filepath: str, method: str = 'joblib'):
        """Carga un modelo guardado."""
        if method == 'joblib':
            model = joblib.load(filepath)
        elif method == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)

        print(f"✅ Modelo cargado desde: {filepath}")
        return model

    @staticmethod
    def save_pipeline(pipeline, scaler, encoder, model, base_dir: str):
        """Guarda pipeline completo."""
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(scaler, base_path / 'scaler.pkl')
        joblib.dump(encoder, base_path / 'encoder.pkl')
        joblib.dump(model, base_path / 'model.pkl')

        print(f"✅ Pipeline guardado en: {base_path}")
```

### 8.2 API de Predicción

```python
# Ejemplo de API simple con FastAPI

"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo al inicio
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    prediction: float
    probability: float = None

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    # Preprocesar
    features = np.array(input_data.features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predecir
    prediction = model.predict(features_scaled)[0]

    # Probabilidad (si aplica)
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features_scaled)[0].max()
    else:
        probability = None

    return PredictionOutput(prediction=float(prediction), probability=probability)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
"""
```

## 9. Mejores Prácticas de MLOps

### 9.1 Versionado de Datos y Modelos

- **DVC (Data Version Control)**: Para versionar datasets
- **MLflow**: Para tracking de experimentos y versionado de modelos
- **Weights & Biases**: Tracking avanzado de experimentos

```python
# Ejemplo básico de tracking con MLflow
"""
import mlflow
import mlflow.sklearn

# Iniciar experimento
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Entrenar modelo
    model.fit(X_train, y_train)

    # Evaluar
    accuracy = model.score(X_test, y_test)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")
"""
```

### 9.2 Monitoreo en Producción

```python
class ModelMonitor:
    """Monitoreo de modelos en producción."""

    @staticmethod
    def detect_data_drift(reference_data, current_data, threshold=0.05):
        """Detecta drift en las distribuciones de datos."""
        from scipy.stats import ks_2samp

        drift_detected = {}

        for column in reference_data.columns:
            if reference_data[column].dtype in [np.float64, np.int64]:
                statistic, p_value = ks_2samp(
                    reference_data[column],
                    current_data[column]
                )
                drift_detected[column] = p_value < threshold

        return drift_detected

    @staticmethod
    def monitor_prediction_distribution(predictions, expected_mean, threshold=0.1):
        """Monitorea si la distribución de predicciones cambia."""
        current_mean = np.mean(predictions)
        drift = abs(current_mean - expected_mean) / expected_mean

        if drift > threshold:
            print(f"⚠️  Drift detectado: {drift:.2%}")
            return True
        return False
```

## 10. Checklist de Proyecto ML End-to-End

### Fase 1: Definición y Planificación
- [ ] Problema de negocio claramente definido
- [ ] Tipo de tarea ML identificado
- [ ] Métricas de éxito establecidas
- [ ] Baseline definido
- [ ] Constraints y requisitos documentados

### Fase 2: Datos
- [ ] Datos recolectados y almacenados en data/raw/
- [ ] Diccionario de datos creado
- [ ] EDA completo realizado
- [ ] Calidad de datos evaluada
- [ ] Valores faltantes analizados

### Fase 3: Preparación
- [ ] Estrategia de imputación definida
- [ ] Outliers manejados
- [ ] Variables categóricas encoded
- [ ] Features escaladas apropiadamente
- [ ] Train/val/test splits creados

### Fase 4: Feature Engineering
- [ ] Features creadas basadas en domain knowledge
- [ ] Feature selection realizada
- [ ] Multicolinealidad verificada
- [ ] Feature importance analizada

### Fase 5: Modelado
- [ ] Baseline model implementado
- [ ] Múltiples algoritmos probados
- [ ] Hyperparameter tuning realizado
- [ ] Cross-validation implementado
- [ ] Overfitting/underfitting verificado

### Fase 6: Evaluación
- [ ] Métricas apropiadas calculadas
- [ ] Confusion matrix/residual plots creados
- [ ] Errores analizados
- [ ] Modelo comparado con baseline
- [ ] Interpretabilidad evaluada (SHAP, LIME)

### Fase 7: Deployment
- [ ] Modelo serializado
- [ ] Pipeline de predicción creado
- [ ] API implementada (si aplica)
- [ ] Tests unitarios escritos
- [ ] Documentación creada

### Fase 8: Monitoreo
- [ ] Métricas de monitoreo definidas
- [ ] Data drift detection implementado
- [ ] Logging configurado
- [ ] Plan de reentrenamiento establecido

## Recursos y Herramientas Clave

### Librerías Esenciales
- **Data**: pandas, numpy, polars
- **Visualización**: matplotlib, seaborn, plotly
- **ML**: scikit-learn, xgboost, lightgbm, catboost
- **Deep Learning**: tensorflow, pytorch, keras
- **Feature Engineering**: feature-engine, category_encoders
- **Interpretability**: shap, lime, eli5
- **Tracking**: mlflow, wandb
- **Testing**: pytest, great_expectations

### Estructura de Código Modular

```python
# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# src/utils.py
import logging

def setup_logging(log_file='project.log'):
    """Configura logging del proyecto."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
```
