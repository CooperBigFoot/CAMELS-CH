# Hyperparameter Optimization Report for EALSTM

Date: 2025-03-20 15:32:07
Number of trials: 10
Best trial: #1

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 36 |
| hidden_size | 250 |
| dropout | 0.41622132040021087 |
| learning_rate | 2.6587543983272695e-05 |
| num_layers | 1 |
| future_hidden_size | 73 |
| future_layers | 1 |
| bidirectional_fusion | concat |

**Best validation loss**: 0.057259

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| train_size | 307018 |
| val_size | 151769 |
| test_size | 151951 |
| best_epoch | 44 |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| input_length | 0.2424 |
| dropout | 0.2424 |
| future_hidden_size | 0.2121 |
| learning_rate | 0.1818 |
| hidden_size | 0.1212 |
| num_layers | 0.0000 |
| future_layers | 0.0000 |
| bidirectional_fusion | 0.0000 |

## Visualizations

- [Optimization History](./ealstm_optimization_history.png)
- [Parameter Importance](./ealstm_param_importances.png)
- [Parameter Contours](./ealstm_param_contours.png)
- [Parameter Correlation](./ealstm_param_correlation.png)