# Hyperparameter Optimization Report for EALSTM

Date: 2025-03-20 21:36:22
Number of trials: 10
Best trial: #1

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 82 |
| hidden_size | 45 |
| dropout | 0.4330880728874676 |
| learning_rate | 0.00015930522616241006 |
| num_layers | 3 |

**Best validation loss**: 0.053142

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| train_size | 303522 |
| val_size | 148273 |
| test_size | 148455 |
| best_epoch | 50 |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| hidden_size | 0.3360 |
| dropout | 0.2280 |
| input_length | 0.1763 |
| num_layers | 0.1489 |
| learning_rate | 0.1108 |

## Visualizations

- [Optimization History](./ealstm_optimization_history.png)
- [Parameter Importance](./ealstm_param_importances.png)
- [Parameter Contours](./ealstm_param_contours.png)
- [Parameter Correlation](./ealstm_param_correlation.png)