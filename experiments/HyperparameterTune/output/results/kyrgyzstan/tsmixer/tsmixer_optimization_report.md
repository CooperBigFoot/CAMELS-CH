# Hyperparameter Optimization Report for TSMIXER - Kyrgyzstan

Date: 2025-03-27 17:17:47
Number of trials: 15
Best trial: #5

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 71 |
| hidden_size | 80 |
| dropout | 0.017194260557609198 |
| learning_rate | 0.000658628931758311 |
| num_mixing_layers | 5 |
| static_embedding_size | 15 |
| fusion_method | concat |

**Best validation loss**: 0.053448

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| basin_count | 61 |
| country | Kyrgyzstan |
| model_type | tsmixer |
| train_size | 206729 |
| val_size | 100955 |
| test_size | 101008 |
| best_epoch | 19 |
| checkpoint_path | experiments/HyperparameterTune/output/checkpoints/kyrgyzstan/tsmixer/trial_5/last.ckpt |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| input_length | 0.3316 |
| learning_rate | 0.2564 |
| hidden_size | 0.2195 |
| num_mixing_layers | 0.0941 |
| static_embedding_size | 0.0592 |
| dropout | 0.0283 |
| fusion_method | 0.0109 |

## Dataset Information

- **Country**: Kyrgyzstan
- **Number of basins**: 61
- **Training samples**: 206729
- **Validation samples**: 100955
- **Testing samples**: 101008