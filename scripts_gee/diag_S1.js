// Diagnóstico mínimo de Sentinel-1 sobre el AOI del Bita.
// Pega esto en un script GEE nuevo y dale Run.
// Pega la salida de la consola para identificar la causa exacta.

var aoi = ee.Geometry.Rectangle([-70.65, 3.05, -67.45, 6.30]);
Map.centerObject(aoi, 7);
Map.addLayer(aoi, {color:'yellow'}, 'AOI');

// 1. Colección base (sin filtros de polarización)
var s1_base = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate('2022-01-01', '2022-12-31');
print('1. S1 base (sin filtros polariz):', s1_base.size());

// 2. Solo IW
var s1_iw = s1_base.filter(ee.Filter.eq('instrumentMode', 'IW'));
print('2. S1 IW:', s1_iw.size());

// 3. IW + VV
var s1_iw_vv = s1_iw.filter(
  ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'));
print('3. S1 IW + VV:', s1_iw_vv.size());

// 4. IW + VV + VH (lo más restrictivo)
var s1_iw_vvvh = s1_iw_vv.filter(
  ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
print('4. S1 IW + VV + VH (dual-pol):', s1_iw_vvvh.size());

// 5. Primera imagen — comprobar bandas
print('5. Bandas primera imagen IW+dual-pol:',
  s1_iw_vvvh.first().bandNames());

// 6. Bandas tras median()
var med = s1_iw_vvvh.select(['VV','VH']).median();
print('6. Bandas tras median():', med.bandNames());

// 7. Bandas tras median().clip(aoi)
var med_clip = med.clip(aoi);
print('7. Bandas tras median().clip():', med_clip.bandNames());
