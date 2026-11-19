# Generated Tables for AI403 Technical Report (All 15 MVTec AD Categories)

## Table 4A & Table 7: Multi-Seed Stability (Mean ± Seed SD on Bottle, 3 Seeds) & Computational Cost

| Model       | Image AUROC (Mean ± Seed SD)   | F1-Score (Mean ± Seed SD)   | Pixel AUROC (Mean ± Seed SD)   | Pixel IoU (Mean ± Seed SD)   | Params   |   Checkpoint Size (MB) |   Train Time (s) |   Inf Time (ms) |
|:------------|:-------------------------------|:----------------------------|:-------------------------------|:-----------------------------|:---------|-----------------------:|-----------------:|----------------:|
| BASELINE    | 0.6894 ± 0.0151                | 0.3196 ± 0.0823             | 0.5869 ± 0.0038                | 0.0511 ± 0.0018              | N/A      |                   0.59 |             9.72 |           61.32 |
| AUTOENCODER | 0.8971 ± 0.0057                | 0.8493 ± 0.0069             | 0.8011 ± 0.0113                | 0.1578 ± 0.0210              | 2430915  |                   9.3  |            95.5  |           35.5  |
| PATCHCORE   | 1.0000 ± 0.0000                | 0.9922 ± 0.0134             | 0.9869 ± 0.0002                | 0.2286 ± 0.0049              | 24862528 |                   5.86 |            18.29 |           61.41 |
| EFFICIENTAD | 0.9849 ± 0.0016                | 0.9676 ± 0.0223             | 0.9768 ± 0.0011                | 0.3794 ± 0.0059              | 3315264  |                  12.68 |           259.11 |           62.04 |

## Table 4B: Overall Benchmark Performance Across All 15 MVTec AD Categories (Seed 42) (Mean ± Category SD)

| Model       |   Categories Evaluated | Image AUROC (Mean ± Category SD)   |   Mean Precision |   Mean Recall | F1-Score (Mean ± Category SD)   | Pixel AUROC (Mean ± Category SD)   | Pixel IoU (Mean ± Category SD)   |
|:------------|-----------------------:|:-----------------------------------|-----------------:|--------------:|:--------------------------------|:-----------------------------------|:---------------------------------|
| BASELINE    |                     15 | 0.6776 ± 0.1222                    |           0.9127 |        0.2733 | 0.3942 ± 0.1940                 | 0.6796 ± 0.1976                    | 0.0804 ± 0.1107                  |
| AUTOENCODER |                     15 | 0.7757 ± 0.1806                    |           0.9036 |        0.4344 | 0.5267 ± 0.2927                 | 0.8220 ± 0.1224                    | 0.1016 ± 0.0657                  |
| PATCHCORE   |                     15 | 0.9648 ± 0.0504                    |           0.9503 |        0.9267 | 0.9327 ± 0.0814                 | 0.9730 ± 0.0207                    | 0.1668 ± 0.1332                  |
| EFFICIENTAD |                     15 | 0.8509 ± 0.1148                    |           0.928  |        0.6365 | 0.7185 ± 0.2073                 | 0.9314 ± 0.0626                    | 0.1926 ± 0.1055                  |

## Per-Class Image AUROC Across All 15 Categories (Seed 42)

| category             |   baseline |   autoencoder |   patchcore |   efficientad |
|:---------------------|-----------:|--------------:|------------:|--------------:|
| bottle               |     0.704  |        0.9008 |      1      |        0.9865 |
| cable                |     0.4959 |        0.5708 |      0.9856 |        0.7701 |
| capsule              |     0.6462 |        0.6677 |      0.9465 |        0.8145 |
| carpet               |     0.6091 |        0.5951 |      0.9852 |        0.8832 |
| grid                 |     0.6608 |        0.9841 |      0.9557 |        0.5514 |
| hazelnut             |     0.7193 |        0.9721 |      1      |        0.9714 |
| leather              |     0.6332 |        0.8876 |      1      |        0.9783 |
| metal_nut            |     0.6491 |        0.3856 |      0.999  |        0.9692 |
| pill                 |     0.5783 |        0.7608 |      0.9514 |        0.82   |
| screw                |     0.5091 |        0.8576 |      0.8123 |        0.7383 |
| tile                 |     0.6079 |        0.7204 |      1      |        0.7886 |
| toothbrush           |     0.8917 |        0.9472 |      0.9028 |        0.9056 |
| transistor           |     0.755  |        0.6008 |      0.9862 |        0.8542 |
| wood                 |     0.9184 |        0.9667 |      0.9851 |        0.8956 |
| zipper               |     0.7862 |        0.8178 |      0.9617 |        0.8369 |
| MEAN (15 Categories) |     0.6776 |        0.7757 |      0.9648 |        0.8509 |

## Study 1: Hyperparameter Study (Learning Rate & Image Resolution)

| filename                                 |   image_auroc |   f1_score |   pixel_auroc |   pixel_iou |   train_time_sec |   inference_time_ms |
|:-----------------------------------------|--------------:|-----------:|--------------:|------------:|-----------------:|--------------------:|
| autoencoder_bottle_s42_sz128.json        |        0.9698 |     0.8224 |        0.8324 |      0.1895 |            79.39 |               34.47 |
| autoencoder_bottle_s42_sz256.json        |        0.9008 |     0.8468 |        0.8028 |      0.173  |            93.94 |               35.58 |
| autoencoder_bottle_s42_sz256_lr0005.json |        0.9008 |     0.8468 |        0.8028 |      0.173  |            93.94 |               35.58 |
| autoencoder_bottle_s42_sz256_lr001.json  |        0.8976 |     0.8113 |        0.789  |      0.1427 |            88.9  |               32.43 |
| patchcore_bottle_s42_sz256.json          |        1      |     1      |        0.9869 |      0.232  |            18.08 |               58.25 |
| patchcore_bottle_s42_sz320.json          |        1      |     0.9921 |        0.987  |      0.5572 |            20.37 |               75.58 |

## Study 2: Data Strategy Study (Without Augmentation vs With Augmentation)

| filename                              |   image_auroc |   f1_score |   pixel_auroc |   pixel_iou |   train_time_sec |   inference_time_ms |
|:--------------------------------------|--------------:|-----------:|--------------:|------------:|-----------------:|--------------------:|
| autoencoder_bottle_s42_sz256.json     |        0.9008 |     0.8468 |        0.8028 |      0.173  |            93.94 |               35.58 |
| autoencoder_bottle_s42_sz256_aug.json |        0.8889 |     0.8148 |        0.7381 |      0.1191 |           112.57 |               44.76 |

## Study 3: Pretrained vs Scratch Encoder (Autoencoder)

| filename                                     |   image_auroc |   f1_score |   pixel_auroc |   pixel_iou |   train_time_sec |   inference_time_ms |
|:---------------------------------------------|--------------:|-----------:|--------------:|------------:|-----------------:|--------------------:|
| autoencoder_bottle_s42_sz256.json            |        0.9008 |     0.8468 |        0.8028 |      0.173  |            93.94 |               35.58 |
| autoencoder_bottle_s42_sz256_pretrained.json |        0.9349 |     0.8649 |        0.816  |      0.1863 |           126.45 |               39.1  |

## Study 4 / Table 5: Ablation Study (EfficientAD With AE Branch vs Without AE Branch)

| filename                               |   image_auroc |   f1_score |   pixel_auroc |   pixel_iou |   train_time_sec |   inference_time_ms |
|:---------------------------------------|--------------:|-----------:|--------------:|------------:|-----------------:|--------------------:|
| efficientad_bottle_s42_sz256.json      |        0.9865 |     0.9841 |        0.9759 |      0.3766 |           255.13 |               60.84 |
| efficientad_bottle_s42_sz256_noAE.json |        0.9778 |     0.976  |        0.9759 |      0.3968 |           187.71 |               52.88 |

## All Raw Experiment Runs

| filename                                     | model       | category   |   seed |   image_size | pretrained   | no_ae   | augmentation   |   image_auroc |   f1_score |   pixel_auroc |   pixel_iou |
|:---------------------------------------------|:------------|:-----------|-------:|-------------:|:-------------|:--------|:---------------|--------------:|-----------:|--------------:|------------:|
| autoencoder_bottle_s123_sz256.json           | autoencoder | bottle     |    123 |          256 | False        | False   | False          |      0.890476 |  0.844037  |      0.789039 | 0.133923    |
| autoencoder_bottle_s2026_sz256.json          | autoencoder | bottle     |   2026 |          256 | False        | False   | False          |      0.9      |  0.857143  |      0.811516 | 0.16661     |
| autoencoder_bottle_s42_sz128.json            | autoencoder | bottle     |     42 |          128 | False        | False   | False          |      0.969841 |  0.82243   |      0.832435 | 0.189472    |
| autoencoder_bottle_s42_sz256.json            | autoencoder | bottle     |     42 |          256 | False        | False   | False          |      0.900794 |  0.846847  |      0.802806 | 0.172973    |
| autoencoder_bottle_s42_sz256_aug.json        | autoencoder | bottle     |     42 |          256 | False        | False   | True           |      0.888889 |  0.814815  |      0.738101 | 0.119089    |
| autoencoder_bottle_s42_sz256_lr0005.json     | autoencoder | bottle     |     42 |          256 | False        | False   | False          |      0.900794 |  0.846847  |      0.802806 | 0.172973    |
| autoencoder_bottle_s42_sz256_lr001.json      | autoencoder | bottle     |     42 |          256 | False        | False   | False          |      0.897619 |  0.811321  |      0.789019 | 0.142687    |
| autoencoder_bottle_s42_sz256_pretrained.json | autoencoder | bottle     |     42 |          256 | True         | False   | False          |      0.934921 |  0.864865  |      0.816013 | 0.186277    |
| autoencoder_cable_s123_sz256.json            | autoencoder | cable      |    123 |          256 | False        | False   | False          |      0.572339 |  0.0631579 |      0.571769 | 0.0437038   |
| autoencoder_cable_s2026_sz256.json           | autoencoder | cable      |   2026 |          256 | False        | False   | False          |      0.556034 |  0.0833333 |      0.668897 | 0.047979    |
| autoencoder_cable_s42_sz256.json             | autoencoder | cable      |     42 |          256 | False        | False   | False          |      0.57084  |  0.103093  |      0.724547 | 0.0548679   |
| autoencoder_capsule_s123_sz256.json          | autoencoder | capsule    |    123 |          256 | False        | False   | False          |      0.649382 |  0.428571  |      0.914698 | 0.0849457   |
| autoencoder_capsule_s2026_sz256.json         | autoencoder | capsule    |   2026 |          256 | False        | False   | False          |      0.66773  |  0.330827  |      0.931501 | 0.088318    |
| autoencoder_capsule_s42_sz256.json           | autoencoder | capsule    |     42 |          256 | False        | False   | False          |      0.66773  |  0.391304  |      0.901783 | 0.0849593   |
| autoencoder_carpet_s123_sz256.json           | autoencoder | carpet     |    123 |          256 | False        | False   | False          |      0.572632 |  0.644295  |      0.763454 | 0.0720157   |
| autoencoder_carpet_s2026_sz256.json          | autoencoder | carpet     |   2026 |          256 | False        | False   | False          |      0.524077 |  0.546763  |      0.746538 | 0.0543107   |
| autoencoder_carpet_s42_sz256.json            | autoencoder | carpet     |     42 |          256 | False        | False   | False          |      0.595104 |  0.591549  |      0.767723 | 0.0753522   |
| autoencoder_grid_s123_sz256.json             | autoencoder | grid       |    123 |          256 | False        | False   | False          |      0.963241 |  0.76087   |      0.903166 | 0.0681692   |
| autoencoder_grid_s2026_sz256.json            | autoencoder | grid       |   2026 |          256 | False        | False   | False          |      0.974937 |  0.733333  |      0.896136 | 0.0740977   |
| autoencoder_grid_s42_sz256.json              | autoencoder | grid       |     42 |          256 | False        | False   | False          |      0.984127 |  0.824742  |      0.887377 | 0.0699123   |
| autoencoder_hazelnut_s123_sz256.json         | autoencoder | hazelnut   |    123 |          256 | False        | False   | False          |      0.946071 |  0.813559  |      0.942978 | 0.254145    |
| autoencoder_hazelnut_s2026_sz256.json        | autoencoder | hazelnut   |   2026 |          256 | False        | False   | False          |      0.957857 |  0.803419  |      0.972722 | 0.274195    |
| autoencoder_hazelnut_s42_sz256.json          | autoencoder | hazelnut   |     42 |          256 | False        | False   | False          |      0.972143 |  0.793103  |      0.965248 | 0.253174    |
| autoencoder_leather_s123_sz256.json          | autoencoder | leather    |    123 |          256 | False        | False   | False          |      0.848166 |  0.84507   |      0.939403 | 0.018278    |
| autoencoder_leather_s2026_sz256.json         | autoencoder | leather    |   2026 |          256 | False        | False   | False          |      0.823709 |  0.834171  |      0.952045 | 0.0165315   |
| autoencoder_leather_s42_sz256.json           | autoencoder | leather    |     42 |          256 | False        | False   | False          |      0.887568 |  0.846512  |      0.951916 | 0.01857     |
| autoencoder_metal_nut_s123_sz256.json        | autoencoder | metal_nut  |    123 |          256 | False        | False   | False          |      0.473607 |  0.0421053 |      0.764978 | 0.0879001   |
| autoencoder_metal_nut_s2026_sz256.json       | autoencoder | metal_nut  |   2026 |          256 | False        | False   | False          |      0.759042 |  0.69863   |      0.805387 | 0.191988    |
| autoencoder_metal_nut_s42_sz256.json         | autoencoder | metal_nut  |     42 |          256 | False        | False   | False          |      0.38563  |  0.113208  |      0.778187 | 0.103967    |
| autoencoder_pill_s123_sz256.json             | autoencoder | pill       |    123 |          256 | False        | False   | False          |      0.763502 |  0.40678   |      0.90139  | 0.201583    |
| autoencoder_pill_s2026_sz256.json            | autoencoder | pill       |   2026 |          256 | False        | False   | False          |      0.738953 |  0.56      |      0.893912 | 0.196995    |
| autoencoder_pill_s42_sz256.json              | autoencoder | pill       |     42 |          256 | False        | False   | False          |      0.760775 |  0.369942  |      0.909713 | 0.205954    |
| autoencoder_screw_s123_sz256.json            | autoencoder | screw      |    123 |          256 | False        | False   | False          |      0.910842 |  0.0806452 |      0.951752 | 0.0322704   |
| autoencoder_screw_s2026_sz256.json           | autoencoder | screw      |   2026 |          256 | False        | False   | False          |      0.899775 |  0.096     |      0.950503 | 0.0303982   |
| autoencoder_screw_s42_sz256.json             | autoencoder | screw      |     42 |          256 | False        | False   | False          |      0.857553 |  0.0491803 |      0.949552 | 0.0287271   |
| autoencoder_tile_s123_sz256.json             | autoencoder | tile       |    123 |          256 | False        | False   | False          |      0.685786 |  0.352941  |      0.502842 | 0.0513735   |
| autoencoder_tile_s2026_sz256.json            | autoencoder | tile       |   2026 |          256 | False        | False   | False          |      0.614719 |  0.3       |      0.559542 | 0.0515069   |
| autoencoder_tile_s42_sz256.json              | autoencoder | tile       |     42 |          256 | False        | False   | False          |      0.720418 |  0.415094  |      0.520322 | 0.0597879   |
| autoencoder_toothbrush_s123_sz256.json       | autoencoder | toothbrush |    123 |          256 | False        | False   | False          |      0.955556 |  0.695652  |      0.923099 | 0.117675    |
| autoencoder_toothbrush_s2026_sz256.json      | autoencoder | toothbrush |   2026 |          256 | False        | False   | False          |      0.925    |  0.836364  |      0.918487 | 0.109582    |
| autoencoder_toothbrush_s42_sz256.json        | autoencoder | toothbrush |     42 |          256 | False        | False   | False          |      0.947222 |  0.872727  |      0.922497 | 0.109708    |
| autoencoder_transistor_s123_sz256.json       | autoencoder | transistor |    123 |          256 | False        | False   | False          |      0.61375  |  0.291667  |      0.696534 | 0.0906554   |
| autoencoder_transistor_s2026_sz256.json      | autoencoder | transistor |   2026 |          256 | False        | False   | False          |      0.5975   |  0.407407  |      0.656296 | 0.0879772   |
| autoencoder_transistor_s42_sz256.json        | autoencoder | transistor |     42 |          256 | False        | False   | False          |      0.600833 |  0.5       |      0.692726 | 0.0882285   |
| autoencoder_wood_s123_sz256.json             | autoencoder | wood       |    123 |          256 | False        | False   | False          |      0.967544 |  0.695652  |      0.760116 | 0.148144    |
| autoencoder_wood_s2026_sz256.json            | autoencoder | wood       |   2026 |          256 | False        | False   | False          |      0.965789 |  0.8       |      0.743463 | 0.131878    |
| autoencoder_wood_s42_sz256.json              | autoencoder | wood       |     42 |          256 | False        | False   | False          |      0.966667 |  0.77551   |      0.753301 | 0.13871     |
| autoencoder_zipper_s123_sz256.json           | autoencoder | zipper     |    123 |          256 | False        | False   | False          |      0.843225 |  0.78      |      0.82811  | 0.0644237   |
| autoencoder_zipper_s2026_sz256.json          | autoencoder | zipper     |   2026 |          256 | False        | False   | False          |      0.853729 |  0.813397  |      0.836147 | 0.0635108   |
| autoencoder_zipper_s42_sz256.json            | autoencoder | zipper     |     42 |          256 | False        | False   | False          |      0.817752 |  0.407895  |      0.802943 | 0.0595082   |
| baseline_bottle_s123_sz256.json              | baseline    | bottle     |    123 |          256 | False        | False   | False          |      0.690476 |  0.414634  |      0.590662 | 0.0527418   |
| baseline_bottle_s2026_sz256.json             | baseline    | bottle     |   2026 |          256 | False        | False   | False          |      0.67381  |  0.27027   |      0.58308  | 0.0492076   |
| baseline_bottle_s42_sz256.json               | baseline    | bottle     |     42 |          256 | False        | False   | False          |      0.703968 |  0.273973  |      0.586905 | 0.0514251   |
| baseline_cable_s123_sz256.json               | baseline    | cable      |    123 |          256 | False        | False   | False          |      0.509933 |  0.122449  |      0.548458 | 0.0308559   |
| baseline_cable_s2026_sz256.json              | baseline    | cable      |   2026 |          256 | False        | False   | False          |      0.501312 |  0.173077  |      0.54466  | 0.0294721   |
| baseline_cable_s42_sz256.json                | baseline    | cable      |     42 |          256 | False        | False   | False          |      0.495877 |  0.252252  |      0.545855 | 0.030349    |
| baseline_capsule_s123_sz256.json             | baseline    | capsule    |    123 |          256 | False        | False   | False          |      0.648983 |  0.355556  |      0.909493 | 0.079393    |
| baseline_capsule_s2026_sz256.json            | baseline    | capsule    |   2026 |          256 | False        | False   | False          |      0.647387 |  0.343284  |      0.910273 | 0.0865683   |
| baseline_capsule_s42_sz256.json              | baseline    | capsule    |     42 |          256 | False        | False   | False          |      0.646191 |  0.330827  |      0.909281 | 0.0795668   |
| baseline_carpet_s123_sz256.json              | baseline    | carpet     |    123 |          256 | False        | False   | False          |      0.608347 |  0.635762  |      0.774555 | 0.0452087   |
| baseline_carpet_s2026_sz256.json             | baseline    | carpet     |   2026 |          256 | False        | False   | False          |      0.611156 |  0.616438  |      0.775738 | 0.046286    |
| baseline_carpet_s42_sz256.json               | baseline    | carpet     |     42 |          256 | False        | False   | False          |      0.609149 |  0.653595  |      0.774847 | 0.0458811   |
| baseline_grid_s123_sz256.json                | baseline    | grid       |    123 |          256 | False        | False   | False          |      0.670844 |  0.323529  |      0.23377  | 0.000132954 |
| baseline_grid_s2026_sz256.json               | baseline    | grid       |   2026 |          256 | False        | False   | False          |      0.663325 |  0.0677966 |      0.230813 | 8.14784e-05 |
| baseline_grid_s42_sz256.json                 | baseline    | grid       |     42 |          256 | False        | False   | False          |      0.660819 |  0.131148  |      0.230751 | 0.000204709 |
| baseline_hazelnut_s123_sz256.json            | baseline    | hazelnut   |    123 |          256 | False        | False   | False          |      0.718571 |  0.261905  |      0.845809 | 0.0936193   |
| baseline_hazelnut_s2026_sz256.json           | baseline    | hazelnut   |   2026 |          256 | False        | False   | False          |      0.725714 |  0.261905  |      0.845666 | 0.0925411   |
| baseline_hazelnut_s42_sz256.json             | baseline    | hazelnut   |     42 |          256 | False        | False   | False          |      0.719286 |  0.261905  |      0.845845 | 0.0930226   |
| baseline_leather_s123_sz256.json             | baseline    | leather    |    123 |          256 | False        | False   | False          |      0.631793 |  0.533333  |      0.660258 | 0.025181    |
| baseline_leather_s2026_sz256.json            | baseline    | leather    |   2026 |          256 | False        | False   | False          |      0.631454 |  0.503817  |      0.659863 | 0.0271772   |
| baseline_leather_s42_sz256.json              | baseline    | leather    |     42 |          256 | False        | False   | False          |      0.633152 |  0.503817  |      0.66031  | 0.0247001   |
| baseline_metal_nut_s123_sz256.json           | baseline    | metal_nut  |    123 |          256 | False        | False   | False          |      0.63392  |  0.471545  |      0.897661 | 0.468192    |
| baseline_metal_nut_s2026_sz256.json          | baseline    | metal_nut  |   2026 |          256 | False        | False   | False          |      0.630499 |  0.527132  |      0.896958 | 0.460871    |
| baseline_metal_nut_s42_sz256.json            | baseline    | metal_nut  |     42 |          256 | False        | False   | False          |      0.649071 |  0.433333  |      0.8968   | 0.458225    |
| baseline_pill_s123_sz256.json                | baseline    | pill       |    123 |          256 | False        | False   | False          |      0.579378 |  0.365714  |      0.570273 | 0.0241163   |
| baseline_pill_s2026_sz256.json               | baseline    | pill       |   2026 |          256 | False        | False   | False          |      0.58347  |  0.413043  |      0.570284 | 0.0242407   |
| baseline_pill_s42_sz256.json                 | baseline    | pill       |     42 |          256 | False        | False   | False          |      0.578287 |  0.289157  |      0.571526 | 0.0247004   |
| baseline_screw_s123_sz256.json               | baseline    | screw      |    123 |          256 | False        | False   | False          |      0.522443 |  0.151515  |      0.867233 | 0.0127886   |
| baseline_screw_s2026_sz256.json              | baseline    | screw      |   2026 |          256 | False        | False   | False          |      0.515884 |  0.108527  |      0.8671   | 0.0129668   |
| baseline_screw_s42_sz256.json                | baseline    | screw      |     42 |          256 | False        | False   | False          |      0.509121 |  0.165414  |      0.86799  | 0.0131327   |
| baseline_tile_s123_sz256.json                | baseline    | tile       |    123 |          256 | False        | False   | False          |      0.604618 |  0.3       |      0.351756 | 0.0344098   |
| baseline_tile_s2026_sz256.json               | baseline    | tile       |   2026 |          256 | False        | False   | False          |      0.609668 |  0.361905  |      0.351464 | 0.0348907   |
| baseline_tile_s42_sz256.json                 | baseline    | tile       |     42 |          256 | False        | False   | False          |      0.607864 |  0.333333  |      0.351476 | 0.0346315   |
| baseline_toothbrush_s123_sz256.json          | baseline    | toothbrush |    123 |          256 | False        | False   | False          |      0.875    |  0.77551   |      0.769167 | 0.0593962   |
| baseline_toothbrush_s2026_sz256.json         | baseline    | toothbrush |   2026 |          256 | False        | False   | False          |      0.894444 |  0.823529  |      0.775326 | 0.0579602   |
| baseline_toothbrush_s42_sz256.json           | baseline    | toothbrush |     42 |          256 | False        | False   | False          |      0.891667 |  0.823529  |      0.768341 | 0.0563506   |
| baseline_transistor_s123_sz256.json          | baseline    | transistor |    123 |          256 | False        | False   | False          |      0.75625  |  0.36      |      0.65181  | 0.108129    |
| baseline_transistor_s2026_sz256.json         | baseline    | transistor |   2026 |          256 | False        | False   | False          |      0.750417 |  0.4       |      0.652806 | 0.106614    |
| baseline_transistor_s42_sz256.json           | baseline    | transistor |     42 |          256 | False        | False   | False          |      0.755    |  0.367347  |      0.653743 | 0.107602    |
| baseline_wood_s123_sz256.json                | baseline    | wood       |    123 |          256 | False        | False   | False          |      0.915789 |  0.651685  |      0.809013 | 0.145069    |
| baseline_wood_s2026_sz256.json               | baseline    | wood       |   2026 |          256 | False        | False   | False          |      0.910526 |  0.441558  |      0.813129 | 0.159739    |
| baseline_wood_s42_sz256.json                 | baseline    | wood       |     42 |          256 | False        | False   | False          |      0.918421 |  0.651685  |      0.808945 | 0.13269     |
| baseline_zipper_s123_sz256.json              | baseline    | zipper     |    123 |          256 | False        | False   | False          |      0.791623 |  0.542169  |      0.730961 | 0.0563321   |
| baseline_zipper_s2026_sz256.json             | baseline    | zipper     |   2026 |          256 | False        | False   | False          |      0.790047 |  0.477987  |      0.72619  | 0.0552203   |
| baseline_zipper_s42_sz256.json               | baseline    | zipper     |     42 |          256 | False        | False   | False          |      0.786239 |  0.441558  |      0.721653 | 0.0532464   |
| efficientad_bottle_s123_sz256.json           | efficientad | bottle     |    123 |          256 | False        | False   | False          |      0.983333 |  0.942149  |      0.977997 | 0.386151    |
| efficientad_bottle_s2026_sz256.json          | efficientad | bottle     |   2026 |          256 | False        | False   | False          |      0.984921 |  0.976378  |      0.976602 | 0.375362    |
| efficientad_bottle_s42_sz256.json            | efficientad | bottle     |     42 |          256 | False        | False   | False          |      0.986508 |  0.984127  |      0.975863 | 0.376626    |
| efficientad_bottle_s42_sz256_noAE.json       | efficientad | bottle     |     42 |          256 | False        | True    | False          |      0.977778 |  0.976     |      0.975886 | 0.39684     |
| efficientad_cable_s123_sz256.json            | efficientad | cable      |    123 |          256 | False        | False   | False          |      0.778861 |  0.413793  |      0.908478 | 0.173601    |
| efficientad_cable_s2026_sz256.json           | efficientad | cable      |   2026 |          256 | False        | False   | False          |      0.779423 |  0.396552  |      0.901338 | 0.161821    |
| efficientad_cable_s42_sz256.json             | efficientad | cable      |     42 |          256 | False        | False   | False          |      0.770052 |  0.52381   |      0.901246 | 0.153548    |
| efficientad_capsule_s123_sz256.json          | efficientad | capsule    |    123 |          256 | False        | False   | False          |      0.838452 |  0.641975  |      0.971016 | 0.105912    |
| efficientad_capsule_s2026_sz256.json         | efficientad | capsule    |   2026 |          256 | False        | False   | False          |      0.814918 |  0.562092  |      0.968446 | 0.112689    |
| efficientad_capsule_s42_sz256.json           | efficientad | capsule    |     42 |          256 | False        | False   | False          |      0.814519 |  0.543046  |      0.970194 | 0.109734    |
| efficientad_carpet_s123_sz256.json           | efficientad | carpet     |    123 |          256 | False        | False   | False          |      0.889246 |  0.851852  |      0.953714 | 0.189116    |
| efficientad_carpet_s2026_sz256.json          | efficientad | carpet     |   2026 |          256 | False        | False   | False          |      0.904093 |  0.901734  |      0.955461 | 0.186365    |
| efficientad_carpet_s42_sz256.json            | efficientad | carpet     |     42 |          256 | False        | False   | False          |      0.883226 |  0.832298  |      0.951905 | 0.188905    |
| efficientad_grid_s123_sz256.json             | efficientad | grid       |    123 |          256 | False        | False   | False          |      0.54553  |  0.294118  |      0.931044 | 0.0680396   |
| efficientad_grid_s2026_sz256.json            | efficientad | grid       |   2026 |          256 | False        | False   | False          |      0.517962 |  0.129032  |      0.924066 | 0.0644581   |
| efficientad_grid_s42_sz256.json              | efficientad | grid       |     42 |          256 | False        | False   | False          |      0.551378 |  0.242424  |      0.933067 | 0.0681084   |
| efficientad_hazelnut_s123_sz256.json         | efficientad | hazelnut   |    123 |          256 | False        | False   | False          |      0.973571 |  0.929577  |      0.977824 | 0.252856    |
| efficientad_hazelnut_s2026_sz256.json        | efficientad | hazelnut   |   2026 |          256 | False        | False   | False          |      0.9775   |  0.929577  |      0.976367 | 0.252288    |
| efficientad_hazelnut_s42_sz256.json          | efficientad | hazelnut   |     42 |          256 | False        | False   | False          |      0.971429 |  0.929577  |      0.977628 | 0.247115    |
| efficientad_leather_s123_sz256.json          | efficientad | leather    |    123 |          256 | False        | False   | False          |      0.97962  |  0.914573  |      0.985333 | 0.0789854   |
| efficientad_leather_s2026_sz256.json         | efficientad | leather    |   2026 |          256 | False        | False   | False          |      0.98981  |  0.962162  |      0.989839 | 0.0775946   |
| efficientad_leather_s42_sz256.json           | efficientad | leather    |     42 |          256 | False        | False   | False          |      0.978261 |  0.905473  |      0.985682 | 0.0816419   |
| efficientad_metal_nut_s123_sz256.json        | efficientad | metal_nut  |    123 |          256 | False        | False   | False          |      0.963832 |  0.924855  |      0.932602 | 0.411544    |
| efficientad_metal_nut_s2026_sz256.json       | efficientad | metal_nut  |   2026 |          256 | False        | False   | False          |      0.962854 |  0.956044  |      0.931671 | 0.421578    |
| efficientad_metal_nut_s42_sz256.json         | efficientad | metal_nut  |     42 |          256 | False        | False   | False          |      0.969208 |  0.943182  |      0.930971 | 0.411236    |
| efficientad_pill_s123_sz256.json             | efficientad | pill       |    123 |          256 | False        | False   | False          |      0.802782 |  0.598039  |      0.944801 | 0.268971    |
| efficientad_pill_s2026_sz256.json            | efficientad | pill       |   2026 |          256 | False        | False   | False          |      0.817512 |  0.631579  |      0.946118 | 0.278133    |
| efficientad_pill_s42_sz256.json              | efficientad | pill       |     42 |          256 | False        | False   | False          |      0.819967 |  0.672811  |      0.941212 | 0.26989     |
| efficientad_screw_s123_sz256.json            | efficientad | screw      |    123 |          256 | False        | False   | False          |      0.711006 |  0.573099  |      0.979921 | 0.039685    |
| efficientad_screw_s2026_sz256.json           | efficientad | screw      |   2026 |          256 | False        | False   | False          |      0.757327 |  0.613636  |      0.980784 | 0.0384476   |
| efficientad_screw_s42_sz256.json             | efficientad | screw      |     42 |          256 | False        | False   | False          |      0.738266 |  0.509317  |      0.979364 | 0.0384807   |
| efficientad_tile_s123_sz256.json             | efficientad | tile       |    123 |          256 | False        | False   | False          |      0.801948 |  0.4       |      0.843461 | 0.206264    |
| efficientad_tile_s2026_sz256.json            | efficientad | tile       |   2026 |          256 | False        | False   | False          |      0.779221 |  0.712121  |      0.836802 | 0.203636    |
| efficientad_tile_s42_sz256.json              | efficientad | tile       |     42 |          256 | False        | False   | False          |      0.7886   |  0.671875  |      0.842071 | 0.201502    |
| efficientad_toothbrush_s123_sz256.json       | efficientad | toothbrush |    123 |          256 | False        | False   | False          |      0.905556 |  0.847458  |      0.966555 | 0.154588    |
| efficientad_toothbrush_s2026_sz256.json      | efficientad | toothbrush |   2026 |          256 | False        | False   | False          |      0.913889 |  0.847458  |      0.961036 | 0.148381    |
| efficientad_toothbrush_s42_sz256.json        | efficientad | toothbrush |     42 |          256 | False        | False   | False          |      0.905556 |  0.833333  |      0.968713 | 0.154791    |
| efficientad_transistor_s123_sz256.json       | efficientad | transistor |    123 |          256 | False        | False   | False          |      0.846667 |  0.590164  |      0.751573 | 0.160493    |
| efficientad_transistor_s2026_sz256.json      | efficientad | transistor |   2026 |          256 | False        | False   | False          |      0.848333 |  0.685714  |      0.752027 | 0.156632    |
| efficientad_transistor_s42_sz256.json        | efficientad | transistor |     42 |          256 | False        | False   | False          |      0.854167 |  0.694444  |      0.753103 | 0.158423    |
| efficientad_wood_s123_sz256.json             | efficientad | wood       |    123 |          256 | False        | False   | False          |      0.877193 |  0.846847  |      0.902016 | 0.237582    |
| efficientad_wood_s2026_sz256.json            | efficientad | wood       |   2026 |          256 | False        | False   | False          |      0.871053 |  0.857143  |      0.90267  | 0.240851    |
| efficientad_wood_s42_sz256.json              | efficientad | wood       |     42 |          256 | False        | False   | False          |      0.895614 |  0.867257  |      0.902774 | 0.241835    |
| efficientad_zipper_s123_sz256.json           | efficientad | zipper     |    123 |          256 | False        | False   | False          |      0.836922 |  0.662983  |      0.955115 | 0.18943     |
| efficientad_zipper_s2026_sz256.json          | efficientad | zipper     |   2026 |          256 | False        | False   | False          |      0.8427   |  0.621469  |      0.960055 | 0.186454    |
| efficientad_zipper_s42_sz256.json            | efficientad | zipper     |     42 |          256 | False        | False   | False          |      0.836922 |  0.625     |      0.956609 | 0.186463    |
| patchcore_bottle_s123_sz256.json             | patchcore   | bottle     |    123 |          256 | False        | False   | False          |      1        |  1         |      0.986667 | 0.230716    |
| patchcore_bottle_s2026_sz256.json            | patchcore   | bottle     |   2026 |          256 | False        | False   | False          |      1        |  0.976744  |      0.986989 | 0.222932    |
| patchcore_bottle_s42_sz256.json              | patchcore   | bottle     |     42 |          256 | False        | False   | False          |      1        |  1         |      0.986912 | 0.232013    |
| patchcore_bottle_s42_sz320.json              | patchcore   | bottle     |     42 |          320 | False        | False   | False          |      1        |  0.992126  |      0.987012 | 0.557235    |
| patchcore_cable_s123_sz256.json              | patchcore   | cable      |    123 |          256 | False        | False   | False          |      0.982196 |  0.930481  |      0.978031 | 0.224617    |
| patchcore_cable_s2026_sz256.json             | patchcore   | cable      |   2026 |          256 | False        | False   | False          |      0.981447 |  0.926316  |      0.979535 | 0.230194    |
| patchcore_cable_s42_sz256.json               | patchcore   | cable      |     42 |          256 | False        | False   | False          |      0.98557  |  0.934783  |      0.978909 | 0.212682    |
| patchcore_capsule_s123_sz256.json            | patchcore   | capsule    |    123 |          256 | False        | False   | False          |      0.930594 |  0.92233   |      0.986614 | 0.0732287   |
| patchcore_capsule_s2026_sz256.json           | patchcore   | capsule    |   2026 |          256 | False        | False   | False          |      0.920622 |  0.742857  |      0.98625  | 0.0808647   |
| patchcore_capsule_s42_sz256.json             | patchcore   | capsule    |     42 |          256 | False        | False   | False          |      0.94655  |  0.823529  |      0.986435 | 0.0818156   |
| patchcore_carpet_s123_sz256.json             | patchcore   | carpet     |    123 |          256 | False        | False   | False          |      0.98435  |  0.945652  |      0.987698 | 0.062819    |
| patchcore_carpet_s2026_sz256.json            | patchcore   | carpet     |   2026 |          256 | False        | False   | False          |      0.984751 |  0.945652  |      0.98733  | 0.064957    |
| patchcore_carpet_s42_sz256.json              | patchcore   | carpet     |     42 |          256 | False        | False   | False          |      0.985152 |  0.945652  |      0.987362 | 0.0663003   |
| patchcore_grid_s123_sz256.json               | patchcore   | grid       |    123 |          256 | False        | False   | False          |      0.945698 |  0.896552  |      0.969718 | 0.0441139   |
| patchcore_grid_s2026_sz256.json              | patchcore   | grid       |   2026 |          256 | False        | False   | False          |      0.951546 |  0.902655  |      0.96711  | 0.051192    |
| patchcore_grid_s42_sz256.json                | patchcore   | grid       |     42 |          256 | False        | False   | False          |      0.955723 |  0.928571  |      0.965279 | 0.0369933   |
| patchcore_hazelnut_s123_sz256.json           | patchcore   | hazelnut   |    123 |          256 | False        | False   | False          |      1        |  1         |      0.987198 | 0.192424    |
| patchcore_hazelnut_s2026_sz256.json          | patchcore   | hazelnut   |   2026 |          256 | False        | False   | False          |      1        |  1         |      0.987624 | 0.196769    |
| patchcore_hazelnut_s42_sz256.json            | patchcore   | hazelnut   |     42 |          256 | False        | False   | False          |      1        |  0.992908  |      0.987132 | 0.197535    |
| patchcore_leather_s123_sz256.json            | patchcore   | leather    |    123 |          256 | False        | False   | False          |      1        |  0.994595  |      0.991881 | 0.0279747   |
| patchcore_leather_s2026_sz256.json           | patchcore   | leather    |   2026 |          256 | False        | False   | False          |      1        |  0.989247  |      0.991983 | 0.0285638   |
| patchcore_leather_s42_sz256.json             | patchcore   | leather    |     42 |          256 | False        | False   | False          |      1        |  0.963351  |      0.991915 | 0.02833     |
| patchcore_metal_nut_s123_sz256.json          | patchcore   | metal_nut  |    123 |          256 | False        | False   | False          |      0.999022 |  0.994652  |      0.982922 | 0.519734    |
| patchcore_metal_nut_s2026_sz256.json         | patchcore   | metal_nut  |   2026 |          256 | False        | False   | False          |      0.998045 |  0.989247  |      0.983535 | 0.517552    |
| patchcore_metal_nut_s42_sz256.json           | patchcore   | metal_nut  |     42 |          256 | False        | False   | False          |      0.999022 |  0.994652  |      0.983762 | 0.521292    |
| patchcore_pill_s123_sz256.json               | patchcore   | pill       |    123 |          256 | False        | False   | False          |      0.960447 |  0.945055  |      0.979447 | 0.262952    |
| patchcore_pill_s2026_sz256.json              | patchcore   | pill       |   2026 |          256 | False        | False   | False          |      0.951718 |  0.960289  |      0.979191 | 0.273988    |
| patchcore_pill_s42_sz256.json                | patchcore   | pill       |     42 |          256 | False        | False   | False          |      0.951446 |  0.945055  |      0.980081 | 0.273851    |
| patchcore_screw_s123_sz256.json              | patchcore   | screw      |    123 |          256 | False        | False   | False          |      0.795245 |  0.573099  |      0.975986 | 0.0316067   |
| patchcore_screw_s2026_sz256.json             | patchcore   | screw      |   2026 |          256 | False        | False   | False          |      0.773724 |  0.614525  |      0.972292 | 0.0307647   |
| patchcore_screw_s42_sz256.json               | patchcore   | screw      |     42 |          256 | False        | False   | False          |      0.812257 |  0.684492  |      0.97399  | 0.0303866   |
| patchcore_tile_s123_sz256.json               | patchcore   | tile       |    123 |          256 | False        | False   | False          |      1        |  0.994083  |      0.950751 | 0.246265    |
| patchcore_tile_s2026_sz256.json              | patchcore   | tile       |   2026 |          256 | False        | False   | False          |      1        |  0.994083  |      0.950976 | 0.245508    |
| patchcore_tile_s42_sz256.json                | patchcore   | tile       |     42 |          256 | False        | False   | False          |      1        |  0.994083  |      0.949631 | 0.249245    |
| patchcore_toothbrush_s123_sz256.json         | patchcore   | toothbrush |    123 |          256 | False        | False   | False          |      0.9      |  0.935484  |      0.98768  | 0.110612    |
| patchcore_toothbrush_s2026_sz256.json        | patchcore   | toothbrush |   2026 |          256 | False        | False   | False          |      0.897222 |  0.9375    |      0.987419 | 0.104851    |
| patchcore_toothbrush_s42_sz256.json          | patchcore   | toothbrush |     42 |          256 | False        | False   | False          |      0.902778 |  0.952381  |      0.987556 | 0.095023    |
| patchcore_transistor_s123_sz256.json         | patchcore   | transistor |    123 |          256 | False        | False   | False          |      0.99375  |  0.915663  |      0.939607 | 0.288542    |
| patchcore_transistor_s2026_sz256.json        | patchcore   | transistor |   2026 |          256 | False        | False   | False          |      0.994167 |  0.879121  |      0.941745 | 0.279275    |
| patchcore_transistor_s42_sz256.json          | patchcore   | transistor |     42 |          256 | False        | False   | False          |      0.98625  |  0.915663  |      0.929913 | 0.273928    |
| patchcore_wood_s123_sz256.json               | patchcore   | wood       |    123 |          256 | False        | False   | False          |      0.987719 |  0.951613  |      0.927844 | 0.112809    |
| patchcore_wood_s2026_sz256.json              | patchcore   | wood       |   2026 |          256 | False        | False   | False          |      0.985965 |  0.95935   |      0.92977  | 0.112505    |
| patchcore_wood_s42_sz256.json                | patchcore   | wood       |     42 |          256 | False        | False   | False          |      0.985088 |  0.944     |      0.928708 | 0.106814    |
| patchcore_zipper_s123_sz256.json             | patchcore   | zipper     |    123 |          256 | False        | False   | False          |      0.9677   |  0.959016  |      0.978452 | 0.0915022   |
| patchcore_zipper_s2026_sz256.json            | patchcore   | zipper     |   2026 |          256 | False        | False   | False          |      0.955882 |  0.970954  |      0.976876 | 0.0935193   |
| patchcore_zipper_s42_sz256.json              | patchcore   | zipper     |     42 |          256 | False        | False   | False          |      0.96166  |  0.970954  |      0.977957 | 0.0964118   |