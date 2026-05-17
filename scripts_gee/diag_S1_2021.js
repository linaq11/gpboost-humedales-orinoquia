// Diagnóstico Sentinel-1 sobre el Bita en 2021 (Sentinel-1B aún activo).

var aoi = ee.Geometry.Rectangle([-70.65, 3.05, -67.45, 6.30]);
Map.centerObject(aoi, 7);
Map.addLayer(aoi, {color:'yellow'}, 'AOI');

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate('2021-01-01', '2021-12-31');
print('S1 base 2021:', s1.size());

var s1_dual = s1
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
print('S1 IW + VV + VH 2021:', s1_dual.size());

print('Primera imagen 2021 — bandas:', s1_dual.first().bandNames());

var med = s1_dual.select(['VV','VH']).median().clip(aoi);
print('Median bandas:', med.bandNames());
Map.addLayer(med.select('VV'), {min:-20, max:0}, 'VV median 2021');
