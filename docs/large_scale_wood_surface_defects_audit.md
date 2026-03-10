# large_scale_wood_surface_defects Audit

- Images: 20276
- Annotations: 43974
- Classes: 10
- Empty images: 1992
- Small annotations: 33400
- Small annotation ratio: 0.7595

## Small-Defect Rule
- Enabled: True
- Combine mode: `any`
- Min area ratio: 0.01
- Min width px: 16.0
- Min height px: 16.0

## Class Distribution
```
     class_name  class_id  box_count  image_count  small_box_count  small_box_ratio  small_image_count
      live_knot         5      21224        11912            17366         0.818225              10286
      dead_knot         2      11985         8350            10482         0.874593               7432
          resin         9       3455         2624             2877         0.832706               2254
knot_with_crack         4       2276         1835              446         0.195958                353
          crack         1       2169         1578             1238         0.570770                905
         marrow         6       1181         1061              331         0.280271                296
      quartzity         8       1075          847              221         0.205581                156
   knot_missing         3        503          478              422         0.838966                406
     blue_stain         0         96           77                7         0.072917                  6
      overgrown         7         10            6               10         1.000000                  6
```

## Image Size Distribution
```
 width  height  image_count
  2800    1024        20107
  2799    1024            9
  2782    1024            7
  2793    1024            7
  2757    1024            5
  2774    1024            5
  2790    1024            5
  2767    1024            4
  2776    1024            4
  2781    1024            4
  2786    1024            4
  2791    1024            4
  2798    1024            4
  2751    1024            3
  2756    1024            3
  2762    1024            3
  2764    1024            3
  2766    1024            3
  2771    1024            3
  2773    1024            3
  2775    1024            3
  2777    1024            3
  2780    1024            3
  2784    1024            3
  2785    1024            3
  2788    1024            3
  2796    1024            3
  2797    1024            3
  2726    1024            2
  2745    1024            2
  2746    1024            2
  2747    1024            2
  2749    1024            2
  2754    1024            2
  2758    1024            2
  2765    1024            2
  2768    1024            2
  2770    1024            2
  2772    1024            2
  2783    1024            2
  2787    1024            2
  2792    1024            2
  2794    1024            2
  2795    1024            2
  2708    1024            1
  2709    1024            1
  2713    1024            1
  2714    1024            1
  2717    1024            1
  2721    1024            1
  2723    1024            1
  2724    1024            1
  2725    1024            1
  2727    1024            1
  2728    1024            1
  2729    1024            1
  2730    1024            1
  2731    1024            1
  2732    1024            1
  2734    1024            1
  2737    1024            1
  2738    1024            1
  2739    1024            1
  2740    1024            1
  2741    1024            1
  2742    1024            1
  2750    1024            1
  2753    1024            1
  2755    1024            1
  2759    1024            1
  2760    1024            1
  2769    1024            1
  2778    1024            1
  2779    1024            1
```

## Bounding Box Distribution
```
          metric  count      min         mean          std         p25         p50          p75           p95           max
 bbox_width_norm  43974 0.000357     0.057941     0.048451    0.027143    0.041071     0.071429      0.154286      0.502143
bbox_height_norm  43974 0.000976     0.135290     0.181067    0.056640    0.075195     0.131836      0.490234      1.000000
  bbox_area_norm  43974 0.000000     0.008857     0.014200    0.001770    0.003361     0.009551      0.035714      0.335616
   bbox_width_px  43974 0.999600   162.217523   135.637109   76.000400  114.998800   200.001200    432.000800   1406.000400
  bbox_height_px  43974 0.999424   138.536952   185.412126   57.999360   76.999680   135.000064    501.999616   1024.000000
    bbox_area_px  43974 0.000000 25391.502322 40707.567121 5074.944000 9636.659200 27376.742400 102399.180800 962278.195200
```

## Validation Summary
```
                    issue  count
    empty_annotation_file   1992
  orphan_annotation_files      0
orphan_semantic_map_files      0
```

- Manifest: `data/processed/large_scale_wood_surface_defects_manifest.jsonl`
- Summary JSON: `outputs/tables/large_scale_wood_surface_defects_summary.json`
- Size Figure: `outputs/figures/large_scale_wood_surface_defects_object_size_distribution.png`
