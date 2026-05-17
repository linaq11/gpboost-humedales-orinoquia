# Extracción de features de teledetección en GEE — Bita

Este directorio contiene el script `Bita_Extract_Features_GEE.js` que extrae bandas Sentinel-2, Sentinel-1, ALOS PALSAR-2 y Copernicus DEM en los 479 puntos de referencia del Bita preparados en `data/bita_clean/puntos_bita_479_para_GEE.csv`.

---

## Paso a paso

### 1. Subir los puntos como asset de tabla en GEE

1. Abre [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2. En la barra lateral izquierda, ve a la pestaña **Assets**.
3. Clic en **NEW → Table upload → CSV file**.
4. Selecciona el archivo `data/bita_clean/puntos_bita_479_para_GEE.csv`.
5. En *Advanced options*:
   - **X column**: `lon`
   - **Y column**: `lat`
   - **CRS**: `EPSG:4326`
6. Asset ID: déjalo como `puntos_bita_479` dentro de tu carpeta `users/linaq_Ejercicios_clase1/`.
7. Clic en **Upload** y espera a que termine la ingesta (1–2 min).

### 2. Pegar el script en GEE

1. En el Code Editor, ve a **Scripts → Owner → tu cuenta**.
2. Clic en **NEW → Script**, ponle nombre `Bita_Extract_Features`.
3. Copia y pega el contenido de `Bita_Extract_Features_GEE.js`.

### 3. Ajustar la ruta del asset

En la línea 14 del script, verifica que la variable `PUNTOS_ASSET` apunte a la ruta correcta. Si subiste el asset bajo otro path, edítala. Por ejemplo:

```javascript
var PUNTOS_ASSET = 'projects/basic-buttress-338101/assets/puntos_bita_479';
```

### 4. Correr el script

1. Clic en **Run** (botón gris arriba del editor).
2. Espera unos segundos. En el panel de la derecha aparecerá:
   - El número de puntos cargados (debe ser 479).
   - La lista de bandas extraídas (~26 features).
   - Una muestra de la primera fila.

### 5. Lanzar la exportación

1. Ve a la pestaña **Tasks** (arriba a la derecha).
2. Verás una tarea llamada `Bita_features_479_2022` en estado **UNSUBMITTED**.
3. Clic en **RUN** al lado de la tarea.
4. En el diálogo, deja todo por defecto (Drive folder, etc.) y confirma.
5. La exportación tarda 2–5 min. Cuando termine, encontrarás `Bita_features_479_2022.csv` en tu Google Drive.

### 6. Descargar el CSV y colocarlo en el repo

Descarga el CSV de Drive y guárdalo en:

```
Examen_2_GPBoost/data/bita_with_features.csv
```

---

## Bandas extraídas (~26 features)

**Sentinel-2 dry y wet** (10 bandas × 2 temporadas = 20):
- B2 (azul), B3 (verde), B4 (rojo), B8 (NIR), B11, B12 (SWIR)
- NDVI, NDWI, MNDWI, EVI

**Sentinel-1 dry y wet** (4 bandas × 2 temporadas = 8):
- VV, VH, VH−VV, VH/VV

**ALOS PALSAR-2 banda L (yearly)** (4):
- HH (dB), HV (dB), HV/HH, HV−HH

**Copernicus DEM** (3):
- Elevación, pendiente, TWI (proxy)

Total: ~35 features.

---

## Notas

- **Año por defecto**: 2022. Si necesitas otro año, edita `var ANIO = 2022;`. Recomendado mantener un año consistente con la fecha de levantamiento de campo de los puntos refe2.
- **Filtro de nubes Sentinel-2**: descarta escenas con > 30 % nubes. Para AOI cerca del piedemonte llanero esto deja suficientes imágenes en temporada seca; en wet puede ser más escaso.
- **Speckle filter Sentinel-1**: focal median 3×3 (30 m). Si quieres más suavizado, usa `focalMedian(50, ...)` o reemplaza por Lee.
- **PALSAR-2 yearly**: el mosaic L2.2 de JAXA es global y anual, no hay separación dry/wet.
- **GEDI L4A AGBD**: NO incluido porque su cobertura es discontinua y muchos puntos quedarían NaN. Si Santiago lo necesita para Notebook 06, lo añadimos como capa opcional con manejo explícito de nulos.

---

## Siguiente paso

Una vez tengas `bita_with_features.csv` en `data/`, corro:

1. Notebook 06 — GPBoost clasificador sobre Bita
2. Notebook 07 — CV espacial Bita → Ariporo  
3. Notebook 08 — GPBoost vs RF/SVM/kNN
