# vnwoodknot Audit

- Images: 1515
- Annotations: 1021
- Classes: 3
- Empty images: 500
- Small annotations: 0
- Small annotation ratio: 0.0000
- Splits: {'test': 229, 'train': 1060, 'validation': 226}

## Small-Defect Rule
- Enabled: True
- Combine mode: `any`
- Min area ratio: 0.01
- Min width px: 16.0
- Min height px: 16.0

## Class Distribution
```
class_name  class_id  box_count  image_count  small_box_count  small_box_ratio  small_image_count
 live_knot         0        519          519                0              0.0                  0
 dead_knot         1        502          496                0              0.0                  0
 knot_free         2          0            0                0              0.0                  0
```

## Image Size Distribution
```
 width  height  image_count
  1500    1500         1515
```

## Bounding Box Distribution
```
          metric  count          min          mean           std           p25           p50           p75           p95          max
 bbox_width_norm   1021     0.104667      0.336161      0.119021      0.252666      0.322666      0.400667      0.548000 9.126680e-01
bbox_height_norm   1021     0.072667      0.331689      0.115867      0.246000      0.317333      0.392666      0.544000 9.346670e-01
  bbox_area_norm   1021     0.012281      0.119985      0.079489      0.065693      0.100292      0.154499      0.271192 5.540160e-01
   bbox_width_px   1021   157.000500    504.241936    178.530970    378.999000    483.999000    601.000500    822.000000 1.369002e+03
  bbox_height_px   1021   109.000500    497.533782    173.800997    369.000000    475.999500    588.999000    816.000000 1.402001e+03
    bbox_area_px   1021 27632.250000 269966.930950 178849.822389 147809.250000 225657.000000 347622.750000 610182.000000 1.246536e+06
```

## Validation Summary
```
                  issue  count
      background_sample    500
          clipped_boxes     15
orphan_annotation_files      0
```

- Manifest: `data/processed/vnwoodknot_manifest.jsonl`
- Summary JSON: `outputs/tables/vnwoodknot_summary.json`
- Size Figure: `outputs/figures/vnwoodknot_object_size_distribution.png`
