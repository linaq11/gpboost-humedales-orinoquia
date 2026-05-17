"""
build_split_notebooks.py
========================
Genera la suite limpia de 6 notebooks de la entrega del Examen 2 GPBoost.

Cada notebook es autocontenido:
- Celda Colab init (detecta IN_COLAB e instala dependencias)
- Carga sus datos desde data/ (CSVs derivados)
- Reutiliza modelos ya entrenados desde modelos/
- Genera y guarda sus figuras en figuras/

Salida:
- notebooks/01_reproduccion_simulacion.ipynb
- notebooks/02_normalizacion_y_tuning.ipynb
- notebooks/03_covarianzas_alternativas.ipynb
- notebooks/04_cv_espacial_vs_aleatoria.ipynb
- notebooks/05_variograma_residuales.ipynb
- notebooks/06_gpboost_humedales_bita.ipynb (plantilla; bloqueado por datos)
"""
import os
import nbformat as nbf

REPO = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(REPO, 'notebooks')
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Bloques reutilizables
# ---------------------------------------------------------------------------

COLAB_MD = (
    "## Compatibilidad con Google Colab\n\n"
    "La siguiente celda detecta si el notebook se ejecuta en Colab y, en ese caso, "
    "instala `gpboost` y `shap` y clona el repositorio para tener acceso a los datos "
    "derivados (`data/`) y a los modelos entrenados (`modelos/`). En Jupyter local "
    "no hace nada."
)

COLAB_CODE = (
    "import sys, os\n"
    "IN_COLAB = 'google.colab' in sys.modules\n"
    "if IN_COLAB:\n"
    "    print('Entorno Colab — instalando dependencias...')\n"
    "    !pip -q install gpboost shap\n"
    "    if not os.path.isdir('/content/repo'):\n"
    "        !git clone -q https://github.com/linaq11/examen-2-gpboost.git /content/repo\n"
    "    os.chdir('/content/repo')\n"
    "    print('CWD:', os.getcwd())\n"
    "else:\n"
    "    # En local: asume que el notebook está en repo/notebooks/\n"
    "    here = os.path.dirname(os.path.abspath('__file__')) if '__file__' in globals() else os.getcwd()\n"
    "    if os.path.basename(here) == 'notebooks':\n"
    "        os.chdir(os.path.dirname(here))\n"
    "    print('CWD:', os.getcwd())"
)

IMPORTS_COMMON = (
    "import json, os\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import gpboost as gpb\n"
    "\n"
    "np.random.seed(1)\n"
    "DATA = 'data'\n"
    "MOD  = 'modelos'\n"
    "FIG  = 'figuras'\n"
    "os.makedirs(FIG, exist_ok=True)"
)


def md(text):     return nbf.v4.new_markdown_cell(text)
def code(text):   return nbf.v4.new_code_cell(text)


def colab_block():
    return [md(COLAB_MD), code(COLAB_CODE)]


def write_nb(filename, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}
    }
    path = os.path.join(OUT, filename)
    nbf.write(nb, path)
    print(f'  ✓ {path}')


# ===========================================================================
# Notebook 01 — Reproducción simulación Sigrist 2021
# ===========================================================================

def build_nb01():
    cells = []
    cells.append(md(
        "# Notebook 01 — Reproducción del ejemplo de Sigrist (2021)\n\n"
        "Replicamos el ejemplo simulado del artículo *Tree-Boosting for Spatial Data* "
        "(TDS Archive, 2021): $n=200$ observaciones generadas por $y = F(x_1) + b(s) + e$ "
        "con covarianza exponencial ($\\sigma^2_{GP}=0{,}35$, $\\rho=0{,}1$) y nugget "
        "$\\sigma^2=0{,}1$. Entrenamos GPBoost, predecimos sobre rejilla 50×50, generamos "
        "la figura 1 del artículo y comparamos contra Linear GP (universal kriging), "
        "L2Boost y Random Forest."
    ))
    cells += colab_block()

    cells.append(md("## 1. Setup"))
    cells.append(code(IMPORTS_COMMON + "\nfrom sklearn.ensemble import RandomForestRegressor"))

    cells.append(md(
        "## 2. Carga de los datos simulados (semilla=1, reproducibles)\n\n"
        "Los datos se generaron una sola vez con la semilla `np.random.seed(1)` siguiendo "
        "el apéndice del artículo y se persistieron en `data/simulado_train_seed1.csv`."
    ))
    cells.append(code(
        "df_train = pd.read_csv(os.path.join(DATA, 'simulado_train_seed1.csv'))\n"
        "X      = df_train[['X1','X2']].values\n"
        "y      = df_train['y'].values\n"
        "coords = df_train[['s_1','s_2']].values\n"
        "n      = len(y)\n"
        "print(f'n={n}, X.shape={X.shape}, coords.shape={coords.shape}')\n"
        "df_train.head()"
    ))

    cells.append(md(
        "## 3. Visualización de los datos simulados\n\n"
        "Tres paneles: la función verdadera $F(x_1)$, el campo $b(s)$ y las observaciones "
        "$y$ proyectadas sobre el plano espacial."
    ))
    cells.append(code(
        "def f1d(x):\n"
        "    x = np.asarray(x, dtype=float).copy()\n"
        "    x[np.abs(x - 0.5) < 1e-12] = 0.5 + 1e-10\n"
        "    return np.sin(3*np.pi*x) + (1 + 3*np.maximum(np.zeros_like(x), x-0.5)/(x-0.5)) - 3\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n"
        "xx = np.linspace(0, 1, 200)\n"
        "axes[0].plot(xx, f1d(xx), 'k-', lw=2)\n"
        "axes[0].set_xlabel('x_1'); axes[0].set_ylabel('F(x_1)')\n"
        "axes[0].set_title('a) Función no-lineal F(x_1)'); axes[0].grid(alpha=0.3)\n"
        "\n"
        "sc = axes[1].scatter(coords[:,0], coords[:,1], c=df_train['b'].values,\n"
        "                     cmap='RdBu_r', s=30); plt.colorbar(sc, ax=axes[1])\n"
        "axes[1].set_xlabel('s_1'); axes[1].set_ylabel('s_2')\n"
        "axes[1].set_title('b) Campo b(s) verdadero')\n"
        "\n"
        "sc = axes[2].scatter(coords[:,0], coords[:,1], c=y, cmap='viridis', s=30)\n"
        "plt.colorbar(sc, ax=axes[2])\n"
        "axes[2].set_xlabel('s_1'); axes[2].set_ylabel('s_2')\n"
        "axes[2].set_title('c) Observaciones y')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(FIG, 'fig1_datos_simulados.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "## 4. Entrenamiento GPBoost\n\n"
        "Hiperparámetros del artículo: `learning_rate=0.01`, `max_depth=3`, "
        "`min_data_in_leaf=10`, `num_leaves=1024`, $M=247$ iteraciones, covarianza exponencial."
    ))
    cells.append(code(
        "# Reentrenamos el modelo (es rápido, ~3 s) para que el notebook sea autocontenido.\n"
        "gp_model = gpb.GPModel(gp_coords=coords, cov_function='exponential')\n"
        "params = {'objective':'regression_l2','learning_rate':0.01,'max_depth':3,\n"
        "          'min_data_in_leaf':10,'num_leaves':1024,'verbose':-1}\n"
        "ds = gpb.Dataset(X, y)\n"
        "bst = gpb.train(params=params, train_set=ds, gp_model=gp_model,\n"
        "                num_boost_round=247)\n"
        "bst.save_model(os.path.join(MOD, 'gpboost_basico.json'))\n"
        "print('Modelo guardado en', os.path.join(MOD, 'gpboost_basico.json'))"
    ))

    cells.append(md(
        "## 5. Predicción sobre rejilla 50×50 y figura 1 del artículo"
    ))
    cells.append(code(
        "nx = 50\n"
        "s1x = np.linspace(0,1,nx); s2x = np.linspace(0,1,nx)\n"
        "S1, S2 = np.meshgrid(s1x, s2x)\n"
        "coords_test = np.column_stack([S1.flatten(), S2.flatten()])\n"
        "X_test = np.column_stack([np.linspace(0,1,nx*nx), np.linspace(0,1,nx*nx)])\n"
        "\n"
        "pred = bst.predict(data=X_test, gp_coords_pred=coords_test,\n"
        "                   pred_latent=True, predict_var=True)\n"
        "F_pred = np.array(pred['fixed_effect']).reshape(nx, nx)\n"
        "b_pred = np.array(pred['random_effect_mean']).reshape(nx, nx)\n"
        "b_var  = np.array(pred['random_effect_cov']).reshape(nx, nx)\n"
        "\n"
        "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
        "ax = axes[0,0]; im = ax.imshow(b_pred, origin='lower', extent=[0,1,0,1], cmap='RdBu_r')\n"
        "plt.colorbar(im, ax=ax); ax.set_title('GP predicho b̂(s)')\n"
        "ax = axes[0,1]; im = ax.imshow(np.sqrt(b_var), origin='lower', extent=[0,1,0,1], cmap='magma')\n"
        "plt.colorbar(im, ax=ax); ax.set_title('SD posterior del GP')\n"
        "ax = axes[1,0]\n"
        "xx = np.linspace(0,1,200)\n"
        "X_curve = np.column_stack([xx, np.full_like(xx, 0.5)])\n"
        "F_curve = np.array(bst.predict(data=X_curve, pred_latent=True, ignore_gp_model=True)).flatten()\n"
        "ax.plot(xx, f1d(xx), 'k-', lw=2, label='F verdadero')\n"
        "ax.plot(xx, F_curve, 'r--', lw=2, label='F̂ GPBoost')\n"
        "ax.legend(); ax.set_xlabel('x_1'); ax.set_title('F̂(x_1) vs verdadero')\n"
        "ax = axes[1,1]\n"
        "sc = ax.scatter(coords[:,0], coords[:,1], c=df_train['b'].values, cmap='RdBu_r', s=30)\n"
        "plt.colorbar(sc, ax=ax); ax.set_title('b(s) verdadero (referencia)')\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(FIG, 'fig2_predicciones_gpboost.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "## 6. Comparativa con métodos alternativos\n\n"
        "Sobre los mismos 200 puntos de entrenamiento y un test grid 50×50 reproducible "
        "comparamos: Linear GP (universal kriging con $F(x)$ lineal), L2Boost (tree-boosting "
        "con coordenadas como features adicionales) y Random Forest con coordenadas."
    ))
    cells.append(code(
        "# Para evaluar RMSE necesitamos un test set con los mismos parámetros pero independiente.\n"
        "# Usamos el conjunto comparativo ya derivado:\n"
        "tabla = pd.read_csv(os.path.join(DATA, 'tabla_comparativa_metodos.csv'))\n"
        "tabla"
    ))
    cells.append(md(
        "### Lectura\n\n"
        "GPBoost obtiene el menor RMSE de los cuatro, ordenado igual que en el artículo "
        "original. El Linear GP queda muy por detrás porque su componente fijo lineal "
        "no puede capturar la discontinuidad de $F(x_1)$. L2Boost y Random Forest se "
        "acercan pero no superan a GPBoost: introducir las coordenadas como features "
        "no es equivalente a modelar explícitamente el proceso espacial."
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
        "ax.barh(tabla['Método'], tabla['RMSE'], color=['#2E86AB','#A23B72','#F18F01','#C73E1D'])\n"
        "for i, v in enumerate(tabla['RMSE']):\n"
        "    ax.text(v+0.01, i, f'{v:.4f}', va='center')\n"
        "ax.set_xlabel('RMSE sobre test grid 50×50')\n"
        "ax.set_title('Comparativa de métodos sobre la simulación de Sigrist 2021')\n"
        "ax.invert_yaxis()\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(FIG, 'fig3_comparativa_metodos.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "## 7. Interpretación con SHAP\n\n"
        "Valor SHAP del booster (componente fijo). Solo $X_1$ tiene señal por construcción; "
        "$X_2$ debería tener importancia cercana a cero."
    ))
    cells.append(code(
        "import shap\n"
        "explainer = shap.TreeExplainer(bst)\n"
        "shap_values = explainer.shap_values(X)\n"
        "shap.summary_plot(shap_values, X, feature_names=['X1','X2'], show=False)\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(FIG, 'fig4_shap_values.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "---\n\n"
        "**Conclusión Fase 1.** El ejemplo de Sigrist 2021 se reproduce de forma "
        "satisfactoria: GPBoost recupera la forma de $F(x_1)$ y el campo $b(s)$, SHAP "
        "atribuye casi toda la importancia a $X_1$, y la tabla comparativa preserva el "
        "orden cualitativo del artículo. Los notebooks 02–05 profundizan en la dependencia "
        "espacial."
    ))

    write_nb('01_reproduccion_simulacion.ipynb', cells)


# ===========================================================================
# Notebook 02 — Normalización + tuning + gap CV/test
# ===========================================================================

def build_nb02():
    cells = []
    cells.append(md(
        "# Notebook 02 — Normalización 0-1 y tuning con grid search\n\n"
        "Aplicamos `MinMaxScaler` 0-1 a los features y exploramos un grid de hiperparámetros "
        "con validación cruzada 3-fold. El hallazgo metodológico clave es el **gap CV/test**: "
        "el mejor CV resulta peor en test que los hiperparámetros defaults del artículo, lo "
        "que indica que la CV aleatoria sobreajusta a la estructura espacial. Esto motiva el "
        "Notebook 04 (CV espacial)."
    ))
    cells += colab_block()
    cells.append(code(IMPORTS_COMMON + "\nfrom sklearn.preprocessing import MinMaxScaler\nfrom sklearn.model_selection import KFold"))

    cells.append(md("## 1. Datos y normalización 0-1"))
    cells.append(code(
        "df = pd.read_csv(os.path.join(DATA, 'simulado_train_seed1.csv'))\n"
        "X      = df[['X1','X2']].values\n"
        "y      = df['y'].values\n"
        "coords = df[['s_1','s_2']].values\n"
        "\n"
        "scaler_X = MinMaxScaler().fit(X)\n"
        "scaler_c = MinMaxScaler().fit(coords)\n"
        "X_n      = scaler_X.transform(X)\n"
        "coords_n = scaler_c.transform(coords)\n"
        "\n"
        "with open(os.path.join(MOD, 'scalers_minmax.json'), 'w') as f:\n"
        "    json.dump({'X_min': scaler_X.data_min_.tolist(),\n"
        "               'X_max': scaler_X.data_max_.tolist(),\n"
        "               'coords_min': scaler_c.data_min_.tolist(),\n"
        "               'coords_max': scaler_c.data_max_.tolist()}, f, indent=2)\n"
        "print('Scalers persistidos. Rango X normalizado:', X_n.min(), '–', X_n.max())"
    ))

    cells.append(md(
        "## 2. Grid search + CV 3-fold\n\n"
        "Grid pequeño (36 combinaciones) sobre `learning_rate`, `max_depth`, `min_data_in_leaf` "
        "y `num_boost_round`. Cada combinación entrena un GPBoost completo con early stopping."
    ))
    cells.append(code(
        "grid = {\n"
        "    'learning_rate': [0.005, 0.01, 0.05],\n"
        "    'max_depth':     [3, 5, 7],\n"
        "    'min_data_in_leaf': [5, 10, 20],\n"
        "    'num_boost_round': [247, 500]\n"
        "}\n"
        "from itertools import product\n"
        "combos = list(product(*grid.values()))\n"
        "print(f'Combinaciones: {len(combos)}  (~38 s estimados con CV 3-fold)')"
    ))
    cells.append(code(
        "# El grid search ya se ejecutó y los resultados están en data/tuning_resultados.json\n"
        "# Aquí solo cargamos y mostramos la conclusión.\n"
        "with open(os.path.join(DATA, 'tuning_resultados.json')) as f:\n"
        "    tuning = json.load(f)\n"
        "print(json.dumps(tuning, indent=2))"
    ))

    cells.append(md(
        "## 3. Hallazgo: gap CV/test\n\n"
        "El JSON anterior resume tres grids progresivamente más exhaustivos (8, 12 y 36 "
        "combinaciones). En los tres, la combinación con mejor RMSE de CV resulta *peor* "
        "en el test grid 50×50 que los defaults del artículo:\n\n"
        "| Grid | mejor-CV RMSE | test RMSE de ese params | defaults test RMSE |\n"
        "|---|---|---|---|\n"
        "| 8 combinaciones | 0.32 | 0.6816 | **0.6073** |\n"
        "| 12 combinaciones | 0.36 | 0.6706 | **0.6073** |\n"
        "| 36 combinaciones | 0.43 | 0.6619 | **0.6073** |\n\n"
        "Es decir, gap CV vs test $\\approx 0{,}30$ unidades de RMSE. La causa es que los "
        "folds aleatorios comparten vecinos espaciales con el entrenamiento, así que el "
        "modelo aprende patrones que se replican en folds pero no en el grid completo. "
        "**Decisión:** se conservan los defaults de Sigrist 2021 como modelo final."
    ))
    cells.append(code(
        "# Defaults del artículo (modelo final), tal cual el Notebook 01:\n"
        "gp = gpb.GPModel(gp_coords=coords, cov_function='exponential')\n"
        "params_default = {'objective':'regression_l2','learning_rate':0.01,'max_depth':3,\n"
        "                  'min_data_in_leaf':10,'num_leaves':1024,'verbose':-1}\n"
        "ds = gpb.Dataset(X, y)\n"
        "bst_def = gpb.train(params=params_default, train_set=ds, gp_model=gp,\n"
        "                    num_boost_round=247)\n"
        "print('Modelo final (defaults Sigrist 2021) reentrenado y disponible en memoria.')"
    ))

    cells.append(md(
        "### Lectura\n\n"
        "La validación cruzada aleatoria selecciona combinaciones que *sobreajustan* a los "
        "folds aleatorios porque cada fold contiene puntos espacialmente cercanos a su "
        "entrenamiento. El test grid 50×50 (que cubre todo el dominio sin agruparse alrededor "
        "de los puntos de entrenamiento) penaliza esos hiperparámetros. La consecuencia "
        "práctica es: **para datos espaciales, validar con CV aleatoria es engañoso**. El "
        "Notebook 04 hace explícita esta crítica con CV por bloques."
    ))

    write_nb('02_normalizacion_y_tuning.ipynb', cells)


# ===========================================================================
# Notebook 03 — Covarianzas alternativas
# ===========================================================================

def build_nb03():
    cells = []
    cells.append(md(
        "# Notebook 03 — Covarianzas alternativas para el GP\n\n"
        "Comparamos cuatro funciones de covarianza para el componente $b(s)$ de GPBoost:\n"
        "exponential, Matérn $\\nu=1.5$, Matérn $\\nu=2.5$ y gaussiana. La hipótesis es "
        "que a mayor *smoothness* de la covarianza, peor desempeño en datos con discontinuidades "
        "como los simulados."
    ))
    cells += colab_block()
    cells.append(code(IMPORTS_COMMON))

    cells.append(md("## 1. Datos"))
    cells.append(code(
        "df = pd.read_csv(os.path.join(DATA, 'simulado_train_seed1.csv'))\n"
        "X      = df[['X1','X2']].values\n"
        "y      = df['y'].values\n"
        "coords = df[['s_1','s_2']].values\n"
        "tabla = pd.read_csv(os.path.join(DATA, 'tabla_covarianzas.csv'))\n"
        "tabla"
    ))

    cells.append(md(
        "## 2. Resultados\n\n"
        "RMSE sobre el test grid 50×50 ordenado por suavidad creciente de la covarianza:"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
        "colors = ['#2E86AB','#5DADE2','#F39C12','#E74C3C']\n"
        "ax.bar(tabla['cov_function'], tabla['RMSE'], color=colors)\n"
        "for i, v in enumerate(tabla['RMSE']):\n"
        "    ax.text(i, v+0.005, f'{v:.4f}', ha='center')\n"
        "ax.set_ylabel('RMSE test grid 50×50')\n"
        "ax.set_title('Comparativa de funciones de covarianza')\n"
        "plt.xticks(rotation=15)\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(FIG, 'fig6_covarianzas_RMSE.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "### Lectura\n\n"
        "El RMSE crece de forma monótona con la suavidad de la covarianza. La exponencial "
        "($\\nu=0.5$, no diferenciable) gana, seguida por Matérn $\\nu=1.5$, Matérn $\\nu=2.5$ "
        "y al final la gaussiana (infinitamente suave). Esto es consistente con la naturaleza "
        "no suave de la simulación, que incluye una discontinuidad en $x_1=0.5$. El criterio "
        "general: **a más rugosidad en los datos, covarianza menos suave**. La elección por "
        "defecto del artículo (exponencial) es la correcta para este caso."
    ))

    cells.append(md("## 3. Sintaxis de GPBoost para Matérn\n\n"
                    "Recordatorio para el reporte: GPBoost expone Matérn con dos argumentos:"))
    cells.append(code(
        "# Sintaxis correcta para Matérn (nu se pasa como cov_fct_shape, no en el nombre)\n"
        "# Ejemplo (no ejecutamos aquí, solo documentación):\n"
        "# gp_matern_15 = gpb.GPModel(gp_coords=coords, cov_function='matern', cov_fct_shape=1.5)\n"
        "# gp_matern_25 = gpb.GPModel(gp_coords=coords, cov_function='matern', cov_fct_shape=2.5)\n"
        "print('Sintaxis documentada en cov_function=\"matern\", cov_fct_shape=ν')"
    ))

    write_nb('03_covarianzas_alternativas.ipynb', cells)


# ===========================================================================
# Notebook 04 — CV espacial vs aleatoria
# ===========================================================================

def build_nb04():
    cells = []
    cells.append(md(
        "# Notebook 04 — Validación cruzada espacial vs aleatoria\n\n"
        "Comparamos dos esquemas de CV: aleatoria $k=4$ y espacial por bloques 2×2. "
        "La hipótesis es que la CV aleatoria *subestima* el RMSE al contaminar cada fold "
        "con vecinos espaciales del entrenamiento, mientras que la espacial entrega una "
        "estimación más honesta del desempeño en zonas no vistas. Esto se alinea con la "
        "*Area of Applicability* de Meyer & Pebesma (2021)."
    ))
    cells += colab_block()
    cells.append(code(IMPORTS_COMMON + "\nfrom sklearn.model_selection import KFold"))

    cells.append(md("## 1. Datos y tabla de resultados"))
    cells.append(code(
        "df = pd.read_csv(os.path.join(DATA, 'simulado_train_seed1.csv'))\n"
        "X      = df[['X1','X2']].values\n"
        "y      = df['y'].values\n"
        "coords = df[['s_1','s_2']].values\n"
        "\n"
        "tabla = pd.read_csv(os.path.join(DATA, 'tabla_cv_espacial.csv'))\n"
        "tabla"
    ))

    cells.append(md(
        "## 2. Visualización del gap CV/test\n\n"
        "El RMSE del test grid 50×50 es 0.6073 (referencia). La CV aleatoria queda por "
        "debajo (optimista) y la CV espacial queda por encima (conservadora):"
    ))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "x = np.arange(len(tabla))\n"
        "ax.bar(x, tabla['RMSE_medio'], yerr=tabla['RMSE_sd'], capsize=8,\n"
        "       color=['#2E86AB','#A23B72'], alpha=0.85)\n"
        "ax.axhline(0.6073, color='red', ls='--', lw=2, label='RMSE test grid (referencia)')\n"
        "ax.set_xticks(x); ax.set_xticklabels(tabla['esquema'])\n"
        "ax.set_ylabel('RMSE k-fold')\n"
        "ax.set_title('CV aleatoria vs CV espacial — gap respecto al test')\n"
        "ax.legend()\n"
        "for i, (m, s, gap) in enumerate(zip(tabla['RMSE_medio'], tabla['RMSE_sd'], tabla['gap_vs_test_pct'])):\n"
        "    ax.text(i, m+s+0.03, f'gap={gap:+.1f}%', ha='center')\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(FIG, 'fig7_cv_espacial_vs_aleatoria.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "### Lectura\n\n"
        "La CV aleatoria entrega RMSE $\\approx 0{,}58$ ($-4\\%$ respecto al test), es decir, "
        "le miente al usuario sugiriendo mejor desempeño del real. La CV espacial entrega "
        "$\\approx 0{,}70$ ($+15{,}5\\%$ respecto al test), conservadora pero honesta porque "
        "evalúa explícitamente la capacidad de extrapolar a zonas espacialmente independientes. "
        "Para reportar desempeño en un Digital Twin de monitoreo ambiental — donde la pregunta "
        "es predecir en zonas no muestreadas — la CV espacial es el estimador correcto."
    ))

    write_nb('04_cv_espacial_vs_aleatoria.ipynb', cells)


# ===========================================================================
# Notebook 05 — Variograma de residuales
# ===========================================================================

def build_nb05():
    cells = []
    cells.append(md(
        "# Notebook 05 — Variograma empírico de residuales\n\n"
        "Diagnóstico de captura del componente espacial $b(s)$. Si GPBoost está correctamente "
        "especificado, los residuales out-of-sample tras restar la suma fija + el GP deben "
        "carecer de estructura espacial (variograma plano cercano al nugget). Comparamos "
        "contra L2Boost (coords como features sin GP) y contra $y - \\hat F(X)$ (qué queda "
        "cuando no se modela $b(s)$)."
    ))
    cells += colab_block()
    cells.append(code(IMPORTS_COMMON + "\nfrom scipy.spatial.distance import pdist, squareform"))

    cells.append(md("## 1. Datos y tabla de variograma precomputada"))
    cells.append(code(
        "df = pd.read_csv(os.path.join(DATA, 'simulado_train_seed1.csv'))\n"
        "X      = df[['X1','X2']].values\n"
        "y      = df['y'].values\n"
        "coords = df[['s_1','s_2']].values\n"
        "\n"
        "df_v = pd.read_csv(os.path.join(DATA, 'variograma_residuales.csv'))\n"
        "df_v"
    ))

    cells.append(md("## 2. Figura: variograma de $y$ vs residuales"))
    cells.append(code(
        "centers = df_v['distancia'].values\n"
        "gamma_y      = df_v['gamma_y'].values\n"
        "gamma_Fonly  = df_v['gamma_resid_F_only'].values\n"
        "gamma_gpb    = df_v['gamma_resid_gpboost'].values\n"
        "gamma_l2     = df_v['gamma_resid_l2boost'].values\n"
        "gamma_theory = df_v['gamma_teorico_exp'].values\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
        "ax = axes[0]\n"
        "ax.plot(centers, gamma_y, 'o-', label='y_train', color='black', lw=2)\n"
        "ax.plot(centers, gamma_Fonly, 's-', label='y − F̂(X) (sin GP)', color='darkgreen', lw=2)\n"
        "ax.plot(centers, gamma_theory, '--', label='Teórico exp (σ²=0.35, ρ=0.1)', color='gray')\n"
        "ax.axhline(0.45, color='red', ls=':', alpha=0.5, label='Sill teórico ≈ 0.45')\n"
        "ax.set_xlabel('Distancia |s_i − s_j|'); ax.set_ylabel('γ(h)')\n"
        "ax.set_title('a) y_train y residuales solo-F'); ax.legend(fontsize=8); ax.grid(alpha=0.3)\n"
        "\n"
        "ax = axes[1]\n"
        "ax.plot(centers, gamma_gpb, 'o-', color='steelblue', lw=2,\n"
        "        label=f'GPBoost (γ̄={gamma_gpb.mean():.3f})')\n"
        "ax.plot(centers, gamma_l2,  's--', color='orange',    lw=1.5,\n"
        "        label=f'L2Boost (γ̄={gamma_l2.mean():.3f})')\n"
        "ax.axhline(0.1, color='red', ls=':', alpha=0.5, label='Nugget teórico σ²=0.10')\n"
        "ax.set_xlabel('Distancia |s_i − s_j|'); ax.set_ylabel('γ(h) residuales')\n"
        "ax.set_title('b) Residuales OOS (k=5)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.savefig(os.path.join(FIG, 'fig8_variograma_residuales.png'), dpi=120, bbox_inches='tight')\n"
        "plt.show()"
    ))

    cells.append(md(
        "### Lectura\n\n"
        "Tres lecturas concurrentes:\n\n"
        "1. **$y$ tiene estructura espacial clara** ($\\bar\\gamma=2{,}94$), con incremento "
        "monótono hasta el sill teórico.\n"
        "2. **Quitar solo $\\hat F(X)$** deja residuales con $\\bar\\gamma=0{,}44$, que coinciden "
        "con el sill teórico $\\sigma^2_{GP}=0{,}45$ — evidencia de que el booster aisla $F(x_1)$ "
        "y deja al GP la dependencia espacial pura.\n"
        "3. **GPBoost completo** reduce $\\bar\\gamma$ a $0{,}36$ (87.9 % de reducción), por "
        "debajo de L2Boost ($0{,}44$, 85.0 %). El residuo restante respecto al nugget "
        "$\\sigma^2=0{,}10$ se debe a la incertidumbre de estimación en $k$-fold y a leves "
        "no-estacionariedades; la diferencia entre GPBoost y L2Boost ratifica que tratar las "
        "coordenadas como features no equivale a modelar el proceso espacial."
    ))

    cells.append(md(
        "## 3. Stats resumen"
    ))
    cells.append(code(
        "with open(os.path.join(DATA, 'variograma_stats.json')) as f:\n"
        "    stats = json.load(f)\n"
        "print(json.dumps(stats, indent=2))"
    ))

    write_nb('05_variograma_residuales.ipynb', cells)


# ===========================================================================
# Notebook 06 — Aplicación a humedales del Bita (plantilla)
# ===========================================================================

def build_nb06():
    cells = []
    cells.append(md(
        "# Notebook 06 — Aplicación a humedales del río Bita (Orinoquía)\n\n"
        "**Estado:** plantilla. Bloqueado hasta que Santiago Rodríguez entregue los CSV "
        "con muestras de entrenamiento (Sentinel-2 + Sentinel-1 + ALOS PALSAR + Copernicus "
        "DEM + GEDI) sobre la cuenca del río Bita.\n\n"
        "Cuando lleguen los datos, este notebook ajustará GPBoost a la clasificación de "
        "humedales con coordenadas como GP y los demás bandas como features fijos. Se "
        "reportará:\n"
        "- Tabla comparativa GPBoost vs Random Forest vs L2Boost en el mismo split.\n"
        "- Mapas de probabilidad por clase y varianza posterior.\n"
        "- Validación cruzada espacial por bloques geográficos.\n"
        "- Variograma de residuales para diagnosticar captura de dependencia espacial."
    ))
    cells += colab_block()
    cells.append(code(IMPORTS_COMMON))

    cells.append(md("## 1. Carga (placeholder)"))
    cells.append(code(
        "# Cuando estén los datos:\n"
        "# df_bita = pd.read_csv(os.path.join(DATA, 'bita_humedales_train.csv'))\n"
        "# X      = df_bita[features].values\n"
        "# y      = df_bita['clase'].values\n"
        "# coords = df_bita[['lon','lat']].values\n"
        "print('Datos del Bita pendientes de Santiago — notebook bloqueado.')"
    ))

    cells.append(md(
        "## 2. Plan de entrenamiento (pendiente)\n\n"
        "```python\n"
        "gp = gpb.GPModel(gp_coords=coords, cov_function='exponential')\n"
        "params = {'objective': 'multiclass', 'num_class': 4,\n"
        "          'learning_rate': 0.01, 'max_depth': 5,\n"
        "          'min_data_in_leaf': 20, 'verbose': -1}\n"
        "bst = gpb.train(params=params, train_set=gpb.Dataset(X, y),\n"
        "                gp_model=gp, num_boost_round=500)\n"
        "```"
    ))

    write_nb('06_gpboost_humedales_bita.ipynb', cells)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generando notebooks split en:', OUT)
    build_nb01()
    build_nb02()
    build_nb03()
    build_nb04()
    build_nb05()
    build_nb06()
    print('\nListo.')
