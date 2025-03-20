# Hyperparameter Optimization Report for TSMIXER

Date: 2025-03-20 13:58:03
Number of trials: 10
Best trial: #9

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 31 |
| hidden_size | 111 |
| dropout | 0.35342867192380856 |
| learning_rate | 0.0002870875348195468 |
| num_mixing_layers | 12 |
| static_embedding_size | 6 |
| fusion_method | add |

**Best validation loss**: 0.060576

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| train_size | 307398 |
| val_size | 152149 |
| test_size | 152331 |
| best_epoch | 46 |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| num_mixing_layers | 0.3952 |
| input_length | 0.3228 |
| dropout | 0.0868 |
| hidden_size | 0.0767 |
| learning_rate | 0.0607 |
| fusion_method | 0.0469 |
| static_embedding_size | 0.0109 |

## Visualizations

- [Optimization History](./tsmixer_optimization_history.png)
- [Parameter Importance](./tsmixer_param_importances.png)
- [Parameter Contours](./tsmixer_param_contours.png)
- [Parameter Correlation](./tsmixer_param_correlation.png)