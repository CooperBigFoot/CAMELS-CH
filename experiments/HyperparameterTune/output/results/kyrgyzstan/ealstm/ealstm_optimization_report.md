# Hyperparameter Optimization Report for EALSTM - Kyrgyzstan

Date: 2025-03-28 12:48:16
Number of trials: 15
Best trial: #2

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 36 |
| learning_rate | 0.0008706020878304854 |
| num_layers | 3 |
| hidden_size | 79 |
| dropout | 0.09091248360355031 |

**Best validation loss**: 0.055493

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| basin_count | 61 |
| country | Kyrgyzstan |
| model_type | ealstm |
| train_size | 208829 |
| val_size | 103055 |
| test_size | 103108 |
| best_epoch | 19 |
| checkpoint_path | experiments/HyperparameterTune/output/checkpoints/kyrgyzstan/ealstm/trial_2/last.ckpt |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| dropout | 0.5525 |
| learning_rate | 0.2055 |
| hidden_size | 0.1333 |
| input_length | 0.0994 |
| num_layers | 0.0093 |

## Dataset Information

- **Country**: Kyrgyzstan
- **Number of basins**: 61
- **Training samples**: 208829
- **Validation samples**: 103055
- **Testing samples**: 103108