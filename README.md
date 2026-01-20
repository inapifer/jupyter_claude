# Jupyter Claude

Proyecto de Jupyter Notebooks configurado con uv y Python 3.12, que incluye mejores prácticas para desarrollo con notebooks.

## Características

- Python 3.12
- Gestión de dependencias con [uv](https://github.com/astral-sh/uv)
- Entorno de Jupyter Notebook/Lab completamente configurado
- Notebook de ejemplo con análisis de datos usando NumPy
- Guía completa de mejores prácticas en `CLAUDE.md`

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

## Mejores Prácticas

Consulta el archivo `CLAUDE.md` para una guía completa de mejores prácticas al trabajar con Jupyter Notebooks, incluyendo:

- Estructura y organización de notebooks
- Código limpio y mantenible
- Estado y reproducibilidad
- Manejo de datos y archivos
- Visualización
- Control de versiones
- Performance y debugging
- Seguridad
- Colaboración

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
