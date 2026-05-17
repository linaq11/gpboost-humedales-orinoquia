// =====================================================================
// Bita_Extract_Features_GEE.js
// Extrae bandas Sentinel-2, Sentinel-1, ALOS PALSAR-2, Copernicus DEM
// y derivados en los 479 puntos de referencia del Bita
// (112 campo refe2 + 300 IDEAM Orinoco + 67 no-humedal manual).
//
// Salida: un CSV exportado a Drive listo para alimentar el Notebook 06
// (GPBoost clasificador sobre Bita).
//
// AUTORES: Lina Quintero, Santiago Rodríguez
// FECHA:   Mayo 2026
// =====================================================================

// ----------------------------------------------------------------------
// 0. CONFIGURACIÓN — editar antes de correr
// ----------------------------------------------------------------------
var PUNTOS_ASSET = 'users/linaq_Ejercicios_clase1/puntos_bita_479';
//   ↑ después de subir el CSV `puntos_bita_479_para_GEE.csv` como
//     Table asset en GEE, reemplaza esta ruta por la que te dé GEE.

var ANIO            = 2022;                              // año central
var BUFFER_PIX_M    = 30;                                // radio de extracción
var DRY_START       = ANIO + '-12-01';                   // temporada seca
var DRY_END         = (ANIO + 1) + '-03-31';
var WET_START       = ANIO + '-06-01';                   // temporada húmeda
var WET_END         = ANIO + '-09-30';
var CLOUD_PCT_MAX   = 30;                                // Sentinel-2 cobertura nubes
var EXPORT_PREFIX   = 'Bita_features_479_' + ANIO;

// ----------------------------------------------------------------------
// 1. Puntos y AOI
// ----------------------------------------------------------------------
var puntos_raw = ee.FeatureCollection(PUNTOS_ASSET);
print('Asset cargado, filas:', puntos_raw.size());
print('Propiedades del primer registro:', puntos_raw.first());

// Forzar geometría desde columnas lon/lat. Esto resuelve el caso en que
// el upload del CSV no asoció X/Y como geometry y todas las features
// quedaron sin punto (error típico: "Image.clip ... empty geometry").
var puntos = puntos_raw.map(function(f) {
  var lon = ee.Number.parse(f.get('lon'));
  var lat = ee.Number.parse(f.get('lat'));
  return f.setGeometry(ee.Geometry.Point([lon, lat]));
});
print('Puntos con geometría reconstruida — primero:', puntos.first());

var aoi = puntos.geometry().bounds().buffer(5000);       // bbox + 5 km
Map.centerObject(aoi, 8);
Map.addLayer(aoi, {color:'yellow'}, 'AOI');
Map.addLayer(puntos, {color:'red'}, 'Puntos referencia');

// ----------------------------------------------------------------------
// 2. Sentinel-2 SR — composites dry y wet
// ----------------------------------------------------------------------
function maskS2(img) {
  // SCL: 4=vegetation, 5=bare, 6=water, 7=cloud_low_prob, 8/9/10/11=clouds
  var scl = img.select('SCL');
  var mask = scl.gte(4).and(scl.lte(7));
  return img.updateMask(mask).divide(10000);
}

function s2composite(startDate, endDate, suffix) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_PCT_MAX))
    .map(maskS2);
  print('S2 ' + suffix + ' — imágenes:', col.size());
  var med = col.median().clip(aoi);
  var ndvi  = med.normalizedDifference(['B8','B4']).rename('NDVI');
  var ndwi  = med.normalizedDifference(['B3','B8']).rename('NDWI');
  var mndwi = med.normalizedDifference(['B3','B11']).rename('MNDWI');
  var evi   = med.expression(
    '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
    {NIR: med.select('B8'), RED: med.select('B4'), BLUE: med.select('B2')}
  ).rename('EVI');
  var out = med.select(['B2','B3','B4','B8','B11','B12'])
    .addBands([ndvi, ndwi, mndwi, evi]);
  // Renombrar con sufijo dry/wet
  var newNames = out.bandNames().map(function(n) {
    return ee.String(n).cat('_S2_').cat(suffix);
  });
  return out.rename(newNames);
}

var s2_dry = s2composite(DRY_START, DRY_END, 'dry');
var s2_wet = s2composite(WET_START, WET_END, 'wet');

// ----------------------------------------------------------------------
// 3. Sentinel-1 — composites VV/VH dry y wet
// ----------------------------------------------------------------------
// S1: si la temporada está vacía, expandir progresivamente la ventana
// hasta encontrar al menos 4 escenas dual-pol.
function s1collection(startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
}

function s1composite(startDate, endDate, suffix) {
  var col = s1collection(startDate, endDate);
  print('S1 ' + suffix + ' — imágenes en ventana original:', col.size());

  // Fallback automático: si la ventana tiene <4 escenas, expandir a todo el año
  var year = startDate.substring(0, 4);
  var yearStart = year + '-01-01';
  var yearEnd   = year + '-12-31';
  var colYear = s1collection(yearStart, yearEnd);
  var colSafe = ee.ImageCollection(
    ee.Algorithms.If(col.size().gte(4), col, colYear)
  );
  print('S1 ' + suffix + ' — imágenes utilizadas (fallback si <4):', colSafe.size());

  var med = colSafe.median().clip(aoi);
  // Speckle filter focal median 3x3
  med = med.focalMedian(30, 'square', 'meters');
  var vh = med.select('VH');
  var vv = med.select('VV');
  var diff  = vh.subtract(vv).rename('VHmVV');
  var ratio = vh.divide(vv).rename('VHdVV');
  var out = med.select(['VV','VH']).addBands([diff, ratio]);
  var newNames = out.bandNames().map(function(n) {
    return ee.String(n).cat('_S1_').cat(suffix);
  });
  return out.rename(newNames);
}

var s1_dry = s1composite(DRY_START, DRY_END, 'dry');
var s1_wet = s1composite(WET_START, WET_END, 'wet');

// ----------------------------------------------------------------------
// 4. ALOS PALSAR-2 — yearly mosaic (banda L)
// ----------------------------------------------------------------------
var palsar = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH')
  .filterDate(ANIO + '-01-01', ANIO + '-12-31')
  .first();
// DN -> gamma0 (dB): 10*log10(DN^2) - 83
var hh_db = palsar.select('HH').pow(2).log10().multiply(10).subtract(83).rename('HH_L_db');
var hv_db = palsar.select('HV').pow(2).log10().multiply(10).subtract(83).rename('HV_L_db');
var ratio_L = hv_db.divide(hh_db).rename('HVdHH_L');
var diff_L  = hv_db.subtract(hh_db).rename('HVmHH_L');
var palsar_out = hh_db.addBands([hv_db, ratio_L, diff_L]).clip(aoi);

// ----------------------------------------------------------------------
// 5. Copernicus DEM (GLO-30) — elevación, pendiente, TWI
// ----------------------------------------------------------------------
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .mosaic().select('DEM').rename('elev').clip(aoi);
var slope = ee.Terrain.slope(dem).rename('slope');
// TWI = ln(a / tan(slope)) — proxy hidrológico
var flowAcc = dem.focalMean(90, 'circle', 'meters').rename('flowAcc_proxy');
var twi = flowAcc.divide(slope.tan().max(0.001)).log().rename('TWI');
var dem_out = dem.addBands([slope, twi]);

// ----------------------------------------------------------------------
// 6. Apilar todo
// ----------------------------------------------------------------------
var stack = s2_dry
  .addBands(s2_wet)
  .addBands(s1_dry)
  .addBands(s1_wet)
  .addBands(palsar_out)
  .addBands(dem_out);

print('Bandas extraídas:', stack.bandNames());
print('Número de features por punto:', stack.bandNames().size());

// ----------------------------------------------------------------------
// 7. Extraer en cada punto y exportar
// ----------------------------------------------------------------------
var samples = stack.reduceRegions({
  collection: puntos,
  reducer: ee.Reducer.mean(),
  scale: 10,
  tileScale: 4
});

print('Muestra de la primera fila extraída:', samples.first());

Export.table.toDrive({
  collection: samples,
  description: EXPORT_PREFIX,
  fileNamePrefix: EXPORT_PREFIX,
  fileFormat: 'CSV',
  selectors: null   // dejar null para incluir todas las columnas
});

// ----------------------------------------------------------------------
// 8. Capas opcionales para inspección visual
// ----------------------------------------------------------------------
Map.addLayer(s2_wet.select(['B4_S2_wet','B3_S2_wet','B2_S2_wet']),
             {min:0, max:0.3}, 'S2 wet RGB', false);
Map.addLayer(s1_wet.select('VV_S1_wet'), {min:-20, max:0}, 'S1 VV wet', false);
Map.addLayer(palsar_out.select('HV_L_db'), {min:-25, max:-5}, 'PALSAR HV-L', false);
Map.addLayer(dem, {min:50, max:300, palette:['blue','green','yellow','red']}, 'Elevación', false);
