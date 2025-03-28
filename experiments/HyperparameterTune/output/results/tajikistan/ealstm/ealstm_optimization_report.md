# Hyperparameter Optimization Report for EALSTM - Tajikistan

Date: 2025-03-28 09:38:54
Number of trials: 15
Best trial: #8

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 71 |
| learning_rate | 9.7803370166594e-05 |
| num_layers | 1 |
| hidden_size | 236 |
| dropout | 0.12938999080000846 |

**Best validation loss**: 0.053769

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| basin_count | 16 |
| country | Tajikistan |
| model_type | ealstm |
| train_size | 53562 |
| val_size | 26137 |
| test_size | 26157 |
| best_epoch | 42 |
| checkpoint_path | experiments/HyperparameterTune/output/checkpoints/tajikistan/ealstm/trial_8/last.ckpt |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| learning_rate | 0.6435 |
| dropout | 0.1333 |
| input_length | 0.1272 |
| hidden_size | 0.0785 |
| num_layers | 0.0176 |

## Dataset Information

- **Country**: Tajikistan
- **Number of basins**: 16
- **Training samples**: 53562
- **Validation samples**: 26137
- **Testing samples**: 26157