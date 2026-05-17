// =====================================================================
// Bita_Extract_Features_GEE_v3_no_S1.js
// Versión sin Sentinel-1 (sobre Vichada hay huecos de cobertura en 2022).
// Features: S2 (10) + PALSAR L (2) + DEM (2) = 14 features.
// AUTORES: Lina Quintero, Santiago Rodríguez — Mayo 2026
// =====================================================================

var PUNTOS_ASSET = 'users/linaq_Ejercicios_clase1/puntos_bita_479';
var ANIO         = 2022;
var EXPORT_PREFIX = 'Bita_features_479_' + ANIO + '_noS1';

// AOI hardcoded del bbox de los 479 puntos + margen
var aoi = ee.Geometry.Rectangle([-70.65, 3.05, -67.45, 6.30]);
Map.centerObject(aoi, 7);
Map.addLayer(aoi, {color:'yellow'}, 'AOI');

// ----- Puntos -----
var puntos_raw = ee.FeatureCollection(PUNTOS_ASSET);
print('Asset filas:', puntos_raw.size());
var puntos = puntos_raw.map(function(f) {
  var lon = ee.Number.parse(f.get('lon'));
  var lat = ee.Number.parse(f.get('lat'));
  return f.setGeometry(ee.Geometry.Point([lon, lat]));
});
Map.addLayer(puntos, {color:'red'}, 'Puntos 479');

// ----- Sentinel-2 -----
function maskS2(img) {
  var scl = img.select('SCL');
  return img.updateMask(scl.gte(4).and(scl.lte(7))).divide(10000);
}
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(ANIO + '-01-01', ANIO + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2);
print('S2 imágenes:', s2.size());

var s2_med = s2.median().clip(aoi).select(['B2','B3','B4','B8','B11','B12']);
var ndvi  = s2_med.normalizedDifference(['B8','B4']).rename('NDVI');
var ndwi  = s2_med.normalizedDifference(['B3','B8']).rename('NDWI');
var mndwi = s2_med.normalizedDifference(['B3','B11']).rename('MNDWI');
var evi = s2_med.expression(
  '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
  {NIR: s2_med.select('B8'), RED: s2_med.select('B4'), BLUE: s2_med.select('B2')}
).rename('EVI');
var s2_stack = s2_med.addBands([ndvi, ndwi, mndwi, evi]);

// ----- ALOS PALSAR-2 -----
var palsar = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
  .filterDate(ANIO + '-01-01', ANIO + '-12-31').first();
var hh = palsar.select('HH').pow(2).log10().multiply(10).subtract(83).rename('HH_L');
var hv = palsar.select('HV').pow(2).log10().multiply(10).subtract(83).rename('HV_L');
var palsar_stack = hh.addBands(hv).clip(aoi);

// ----- DEM -----
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').mosaic()
  .select('DEM').rename('elev').clip(aoi);
var slope = ee.Terrain.slope(dem).rename('slope');
var dem_stack = dem.addBands(slope);

// ----- Apilar y extraer -----
var stack = s2_stack.addBands(palsar_stack).addBands(dem_stack);
print('Bandas finales:', stack.bandNames());

var samples = stack.reduceRegions({
  collection: puntos,
  reducer: ee.Reducer.mean(),
  scale: 10,
  tileScale: 4
});
print('Primera fila extraída:', samples.first());

// ----- Exportar -----
Export.table.toDrive({
  collection: samples,
  description: EXPORT_PREFIX,
  fileNamePrefix: EXPORT_PREFIX,
  fileFormat: 'CSV'
});

// Visualizaciones opcionales
Map.addLayer(s2_med.select(['B4','B3','B2']), {min:0, max:0.3}, 'S2 RGB', false);
Map.addLayer(hv, {min:-25, max:-5}, 'PALSAR HV', false);
Map.addLayer(dem, {min:50, max:300, palette:['blue','green','yellow','red']}, 'Elev', false);
