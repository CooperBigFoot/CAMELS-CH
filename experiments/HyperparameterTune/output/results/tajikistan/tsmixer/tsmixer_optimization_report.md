# Hyperparameter Optimization Report for TSMIXER - Tajikistan

Date: 2025-03-27 11:53:45
Number of trials: 15
Best trial: #8

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 124 |
| hidden_size | 84 |
| dropout | 0.07046211248738132 |
| learning_rate | 0.0004021554526690286 |
| num_mixing_layers | 3 |
| static_embedding_size | 20 |
| fusion_method | add |

**Best validation loss**: 0.057879

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| basin_count | 16 |
| country | Tajikistan |
| model_type | tsmixer |
| train_size | 52714 |
| val_size | 25289 |
| test_size | 25309 |
| best_epoch | 60 |
| checkpoint_path | experiments/HyperparameterTune/output/checkpoints/tajikistan/tsmixer/trial_8/last.ckpt |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| dropout | 0.3513 |
| hidden_size | 0.2124 |
| input_length | 0.1995 |
| learning_rate | 0.0896 |
| num_mixing_layers | 0.0733 |
| static_embedding_size | 0.0703 |
| fusion_method | 0.0037 |

## Dataset Information

- **Country**: Tajikistan
- **Number of basins**: 16
- **Training samples**: 52714
- **Validation samples**: 25289
- **Testing samples**: 25309