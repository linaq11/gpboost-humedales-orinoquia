// =====================================================================
// Bita_Extract_Features_GEE_v2.js
// Versión robusta: AOI hardcoded + diagnósticos + fallbacks.
// Si v1 daba "no bands" o "empty geometry", usa esta versión.
// AUTORES: Lina Quintero, Santiago Rodríguez — Mayo 2026
// =====================================================================

// ----------------------------------------------------------------------
// 0. CONFIGURACIÓN
// ----------------------------------------------------------------------
var PUNTOS_ASSET = 'users/linaq_Ejercicios_clase1/puntos_bita_479';
var ANIO         = 2021;   // 2022 sin cobertura S1; 2021 tiene ~236 escenas
var EXPORT_PREFIX = 'Bita_features_479_' + ANIO;

// AOI HARDCODED: bbox real de los 479 puntos según el CSV preparado
// (lon: -70.579 a -67.518, lat: 3.112 a 6.213) + 0.05° de margen.
var aoi = ee.Geometry.Rectangle([-70.65, 3.05, -67.45, 6.30]);

Map.centerObject(aoi, 7);
Map.addLayer(aoi, {color:'yellow'}, 'AOI hardcoded');

// ----------------------------------------------------------------------
// 1. Puntos — cargar y forzar geometría desde lon/lat
// ----------------------------------------------------------------------
var puntos_raw = ee.FeatureCollection(PUNTOS_ASSET);
print('Asset: filas leídas =', puntos_raw.size());

var puntos = puntos_raw.map(function(f) {
  var lon = ee.Number.parse(f.get('lon'));
  var lat = ee.Number.parse(f.get('lat'));
  return f.setGeometry(ee.Geometry.Point([lon, lat]));
});
print('Primer punto con geometría reconstruida:', puntos.first());
Map.addLayer(puntos, {color:'red'}, 'Puntos 479');

// ----------------------------------------------------------------------
// 2. Sentinel-2 — composite anual completo (más robusto que dry/wet)
// ----------------------------------------------------------------------
function maskS2(img) {
  var scl = img.select('SCL');
  var mask = scl.gte(4).and(scl.lte(7));
  return img.updateMask(mask).divide(10000);
}

var s2_year = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(ANIO + '-01-01', ANIO + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2);
print('S2 año completo — imágenes:', s2_year.size());

var s2_med = s2_year.median().clip(aoi)
  .select(['B2','B3','B4','B8','B11','B12']);

var ndvi  = s2_med.normalizedDifference(['B8','B4']).rename('NDVI');
var ndwi  = s2_med.normalizedDifference(['B3','B8']).rename('NDWI');
var mndwi = s2_med.normalizedDifference(['B3','B11']).rename('MNDWI');
var evi = s2_med.expression(
  '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
  {NIR: s2_med.select('B8'), RED: s2_med.select('B4'), BLUE: s2_med.select('B2')}
).rename('EVI');
var s2_stack = s2_med.addBands([ndvi, ndwi, mndwi, evi]);

// ----------------------------------------------------------------------
// 3. Sentinel-1 — composite anual completo
// ----------------------------------------------------------------------
var s1_year = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate(ANIO + '-01-01', ANIO + '-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
print('S1 año completo — imágenes:', s1_year.size());

var s1_med = s1_year.select(['VV','VH']).median().clip(aoi)
  .focalMedian(30, 'square', 'meters');
var vh_minus_vv = s1_med.select('VH').subtract(s1_med.select('VV')).rename('VHmVV');
var vh_div_vv   = s1_med.select('VH').divide(s1_med.select('VV')).rename('VHdVV');
var s1_stack = s1_med.addBands([vh_minus_vv, vh_div_vv]);

// ----------------------------------------------------------------------
// 4. ALOS PALSAR-2 — yearly mosaic (banda L)
// ----------------------------------------------------------------------
var palsar = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
  .filterDate(ANIO + '-01-01', ANIO + '-12-31')
  .first();
print('PALSAR yearly mosaic:', palsar);

var hh = palsar.select('HH').pow(2).log10().multiply(10).subtract(83).rename('HH_L');
var hv = palsar.select('HV').pow(2).log10().multiply(10).subtract(83).rename('HV_L');
var palsar_stack = hh.addBands(hv).clip(aoi);

// ----------------------------------------------------------------------
// 5. DEM Copernicus
// ----------------------------------------------------------------------
var dem   = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .mosaic().select('DEM').rename('elev').clip(aoi);
var slope = ee.Terrain.slope(dem).rename('slope');
var dem_stack = dem.addBands(slope);

// ----------------------------------------------------------------------
// 6. Apilar y extraer
// ----------------------------------------------------------------------
var stack = s2_stack.addBands(s1_stack).addBands(palsar_stack).addBands(dem_stack);
print('Total bandas a extraer:', stack.bandNames());

var samples = stack.reduceRegions({
  collection: puntos,
  reducer: ee.Reducer.mean(),
  scale: 10,
  tileScale: 4
});
print('Muestra de la primera fila:', samples.first());

// ----------------------------------------------------------------------
// 7. Exportar
// ----------------------------------------------------------------------
Export.table.toDrive({
  collection: samples,
  description: EXPORT_PREFIX,
  fileNamePrefix: EXPORT_PREFIX,
  fileFormat: 'CSV'
});

// Visualización (opcional)
Map.addLayer(s2_med.select(['B4','B3','B2']), {min:0, max:0.3}, 'S2 RGB', false);
Map.addLayer(s1_med.select('VV'), {min:-20, max:0}, 'S1 VV', false);
Map.addLayer(palsar.select('HV'), {min:0, max:10000}, 'PALSAR HV (DN)', false);
