# Plan de trabajo — Examen 2 GPBoost

Para coordinar con Santiago. Resume el estado actual, las fases y la división de trabajo.

---

## Hola Santiago,

Te resumo cómo va el examen 2 (GPBoost) y dónde entras tú. Lo dividí en dos fases.

### Fase 1 — Reproducción y mejoras sobre el ejemplo simulado del artículo (HECHO)

Repliqué el ejemplo de *Sigrist 2021 — Tree-Boosting for Spatial Data* con su simulación de 200 puntos y agregué cuatro mejoras metodológicas. Quedó en cinco notebooks corriendo end-to-end:

- **01 — Reproducción.** Simulación, entrenamiento GPBoost con los hiperparámetros del paper, comparativa contra Linear GP, L2Boost y Random Forest. RMSE 0,6073 — gana GPBoost igual que en el artículo.
- **02 — Normalización 0-1 + tuning.** Grid search con validación cruzada, hallazgo: el mejor de CV es peor en test (gap de 0,30 unidades). Lo dejamos como hallazgo metodológico — la CV aleatoria miente cuando hay autocorrelación espacial.
- **03 — Covarianzas alternativas.** Comparativa entre exponencial, Matérn 1.5, Matérn 2.5 y gaussiana. El RMSE crece monótonamente con la suavidad, la exponencial gana.
- **04 — CV espacial vs aleatoria.** Demostración explícita del problema del notebook 02: CV aleatoria k=4 subestima en 4 %, CV espacial por bloques 2×2 sobreestima en 15 %. La espacial es la honesta para extrapolación.
- **05 — Variograma de residuales.** Diagnóstico geoestadístico. GPBoost reduce 88 % de la estructura espacial vs 85 % de L2Boost. Confirma que modelar el GP funciona mejor que meter coordenadas como features.

Cada notebook abre en Colab con un clic. Repo: `https://github.com/linaq11/examen-2-gpboost`.

### Fase 2 — Aplicación a humedales del Bita (TU TURNO)

Esto es donde necesito tus datos. La idea es montar tres notebooks adicionales para llevar GPBoost a tu área de trabajo:

- **Notebook 06 — GPBoost clasificador sobre Bita.** Lead tuyo (definición de clases, balance de muestras, interpretación ecológica), yo te apoyo con la implementación GPBoost multiclase.
- **Notebook 07 — CV espacial Bita → extrapolación Ariporo.** Tuyo. Aplicamos la metodología de validación espacial por bloques del notebook 04 a tu área.
- **Notebook 08 — GPBoost vs RF / SVM / kNN sobre Bita.** Conjunto. Tabla comparativa final con interpretación ecológica.

### Lo que necesito de tu lado para arrancar la Fase 2

1. **Un CSV de entrenamiento** con columnas:
   - `lon`, `lat` (WGS84)
   - `clase` (entero o string)
   - Features Sentinel-2 (B2, B3, B4, B8, NDVI, NDWI, MNDWI, EVI)
   - Features Sentinel-1 (VV, VH, VH/VV, idealmente medias de seca y húmeda)
   - Features ALOS PALSAR (HH, HV, HV/HH banda L)
   - Features Copernicus DEM (elevación, pendiente, TWI)
   - Features GEDI si las tienes (RH98, RH50, FHD)
   - `año`

   Si no tienes todo, con 4–6 features arranco. Lo demás lo vamos sumando.

2. **El esquema de clases.** Qué representa cada valor de `clase` (humedal estacional, permanente, bosque ripario, sabana, etc.).

3. **El AOI del Bita** en GeoJSON o Shapefile.

4. *Opcional*: split train/test si ya lo tienes. Si no, lo armo con CV espacial.

5. *Opcional*: README corto con resolución de cada feature, temporada de adquisición y conteo de muestras por clase.

### Fechas

- Entrega del examen: **20 de mayo**.
- Idealmente recibo tus datos esta semana para alcanzar a montar al menos el Notebook 06 antes de la entrega.
- Si los datos llegan después del 20, dejamos los notebooks 06–08 documentados como trabajo futuro y los terminamos para la tesis.

### Cómo te conectas al repo

1. Acepta la invitación al repo `linaq11/examen-2-gpboost` que te llega al correo (o me dices cuál usar).
2. Cualquier notebook lo abres en Colab con la URL:
   `https://colab.research.google.com/github/linaq11/examen-2-gpboost/blob/main/notebooks/<archivo>.ipynb`
3. La primera celda detecta Colab y te instala todo. Ya no tienes que pelear con la instalación local.

Cualquier duda me dices.

— Lina
