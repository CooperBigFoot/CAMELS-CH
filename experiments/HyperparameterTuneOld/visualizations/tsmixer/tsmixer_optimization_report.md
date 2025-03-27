# Hyperparameter Optimization Report for TSMIXER

Date: 2025-03-21 11:18:25
Number of trials: 10
Best trial: #7

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 59 |
| hidden_size | 51 |
| dropout | 0.022613644455269033 |
| learning_rate | 4.473636174621264e-05 |
| num_mixing_layers | 7 |
| static_embedding_size | 9 |
| fusion_method | add |

**Best validation loss**: 0.060139

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| train_size | 305270 |
| val_size | 150021 |
| test_size | 150203 |
| best_epoch | 50 |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| input_length | 0.3407 |
| dropout | 0.2400 |
| num_mixing_layers | 0.1513 |
| learning_rate | 0.1202 |
| hidden_size | 0.0655 |
| fusion_method | 0.0424 |
| static_embedding_size | 0.0399 |

## Visualizations

- [Optimization History](./tsmixer_optimization_history.png)
- [Parameter Importance](./tsmixer_param_importances.png)
- [Parameter Contours](./tsmixer_param_contours.png)
- [Parameter Correlation](./tsmixer_param_correlation.png)