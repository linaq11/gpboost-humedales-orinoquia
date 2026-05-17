# Examen 2 — Reproducción y extensión de GPBoost

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Open Notebook 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/linaq11/examen-2-gpboost/blob/main/notebooks/01_reproduccion_simulacion.ipynb)

Configuración rápida para Santiago / Colab: ver [COLAB_SETUP.md](COLAB_SETUP.md).

Reproducción y extensión del ejemplo del artículo *Tree-Boosting for Spatial Data* de Fabio Sigrist (TDS Archive, 2021) y aplicación del algoritmo GPBoost a la clasificación de humedales en la Orinoquía colombiana.

**Curso:** Percepción Remota, Maestría en Geomática, Universidad Nacional de Colombia
**Autores:** Lina María Quintero Fonseca, Santiago Rodríguez Basabe
**Fecha de entrega:** 20 de mayo de 2026

---

## Resumen del trabajo

El algoritmo GPBoost (Sigrist, 2020) combina tree-boosting con modelos de proceso gaussiano para datos espaciales mediante el modelo `y = F(X) + b(s) + e`, donde `F(X)` es un ensemble de árboles (no-lineal sobre covariables) y `b(s)` es un proceso gaussiano (estructura espacial explícita). El presente trabajo se organiza en dos fases:

**Fase 1 — Reproducción del ejemplo simulado.** Se replica el ejemplo del artículo Sigrist 2021 con n=200 observaciones simuladas según `y = F(x₁) + b(s) + e` (covarianza exponencial, σ²_GP=0.35, ρ=0.1). Se entrena GPBoost con los hiperparámetros del artículo, se generan predicciones sobre rejilla 50×50, se interpretan los SHAP values, se sintoniza por grid search + CV 4-fold, y se compara contra Linear GP, L2Boost y Random Forest.

**Fase 2 — Extensión: manejo de dependencia espacial.** Se profundiza en el componente espacial del modelo mediante: (i) comparación de funciones de covarianza alternativas (exponential, Matérn 1.5, Matérn 2.5, gaussian); (ii) validación cruzada espacial por bloques geográficos frente a CV aleatoria; (iii) análisis del variograma empírico de residuales como diagnóstico de la captura del componente espacial; (iv) aplicación a datos reales de clasificación de humedales en la cuenca del río Bita (Orinoquía colombiana).

## Resultados principales

Sobre la simulación del artículo:

| Método | RMSE |
|---|---|
| **GPBoost (boosting + GP)** | **0.6073** |
| L2Boost (boosting + coords) | 0.6500 |
| Random Forest + coords | 0.6905 |
| Linear GP (universal kriging) | 1.0915 |

El orden cualitativo reproduce el del artículo original: GPBoost supera al resto en RMSE, el Linear GP queda muy por debajo por su asunción de linealidad violada por la función mean discontinua $F(x_1)$.

## Estructura del repositorio

```
examen-2-gpboost/
├── notebooks/
│   ├── 01_reproduccion_simulacion.ipynb       ← Fase 1 (ejecuta en ~16 s)
│   ├── 02_normalizacion_y_tuning.ipynb        ← Normalización 0-1 + gap CV/test
│   ├── 03_covarianzas_alternativas.ipynb      ← Exp / Matérn 1.5 / Matérn 2.5 / Gauss
│   ├── 04_cv_espacial_vs_aleatoria.ipynb      ← CV aleatoria vs por bloques 2×2
│   ├── 05_variograma_residuales.ipynb         ← Diagnóstico OOS k=5
│   └── 06_gpboost_humedales_bita.ipynb        ← Plantilla (espera datos Santiago)
├── data/
│   ├── simulado_train_seed1.csv               ← Datos simulados Sigrist 2021
│   ├── tabla_comparativa_metodos.csv          ← Tabla principal de resultados
│   └── bita_humedales_*.csv                   ← Datos de humedales Bita (Fase 2)
├── modelos/
│   ├── gpboost_basico.json                    ← Modelo GPBoost básico
│   ├── l2boost_basico.json                    ← L2Boost comparativo
│   └── randomforest_basico.pkl                ← RF tuneado
├── figuras/                                    ← PNGs de todas las figuras del reporte
├── reporte/
│   ├── reporte.md                             ← Reporte técnico fuente
│   └── reporte.docx                           ← Reporte final en .docx
├── README.md
├── LICENSE                                     ← MIT
├── CITATION.cff
├── requirements.txt                            ← Dependencias con versiones exactas
└── .gitignore
```

## Reproducibilidad

### Requisitos

```bash
python -m venv venv
source venv/bin/activate     # o .\venv\Scripts\activate en Windows
pip install -r requirements.txt
```

### Ejecución

Los notebooks deben ejecutarse en orden numérico. La semilla aleatoria está fijada en `np.random.seed(1)` para garantizar reproducibilidad.

```bash
jupyter notebook notebooks/
```

Cada notebook es autocontenido y carga los datos derivados que necesita desde `data/`. Los modelos entrenados se cargan/guardan automáticamente en `modelos/`.

## Datos

- **Datos simulados (Fase 1):** generados según las ecuaciones del Apéndice del artículo Sigrist 2021. Semilla `1`. Reproducibles ejecutando el notebook 01.
- **Datos de humedales Bita (Fase 2):** subset preparado por Santiago Rodríguez Basabe a partir de imágenes Sentinel-2, Sentinel-1, ALOS PALSAR, Copernicus DEM y GEDI sobre la cuenca del río Bita (Orinoquía colombiana). Los datos crudos no se redistribuyen en este repositorio por su tamaño; los CSV procesados con features y etiquetas sí están incluidos en `data/`.

## Cómo citar este trabajo

```
Quintero Fonseca, L. M. & Rodríguez Basabe, S. (2026). Reproducción y extensión del
algoritmo GPBoost para datos espaciales: aplicación a la clasificación de
humedales en la Orinoquía colombiana. Examen 2, curso de Percepción Remota,
Maestría en Geomática, Universidad Nacional de Colombia.
```

## Licencias y atribuciones

- El código de este repositorio se distribuye bajo licencia MIT (ver `LICENSE`).
- La biblioteca [GPBoost](https://github.com/fabsig/GPBoost) (Fabio Sigrist) se utiliza bajo su licencia Apache 2.0.
- El ejemplo simulado se reproduce a partir del artículo *Tree-Boosting for Spatial Data* publicado por Fabio Sigrist en TDS Archive Medium (marzo 2021), citado como referencia metodológica.
- Texto, figuras y documentación bajo Creative Commons Atribución 4.0 Internacional (CC BY 4.0).

## Referencias

- Sigrist, F. (2020). *Gaussian Process Boosting*. arXiv:2004.02653.
- Sigrist, F. (2021). *Tree-Boosting for Spatial Data*. TDS Archive Medium, marzo 2021.
- Repositorio oficial GPBoost: https://github.com/fabsig/GPBoost
- Meyer, H. & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods in Ecology and Evolution*, 12(9), 1620–1633.

## Contacto

Lina María Quintero Fonseca — lquinterof@unal.edu.co
Santiago Rodríguez Basabe — sanrodriguezba@unal.edu.co
