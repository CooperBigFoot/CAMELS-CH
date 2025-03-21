# Hyperparameter Optimization Report for TIDE

Date: 2025-03-21 15:10:18
Number of trials: 10
Best trial: #9

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 70 |
| hidden_size | 101 |
| dropout | 0.3803925243084487 |
| learning_rate | 0.0001326033192269654 |
| num_encoder_layers | 3 |
| num_decoder_layers | 2 |
| decoder_output_size | 21 |
| temporal_decoder_hidden_size | 36 |
| use_layer_norm | False |

**Best validation loss**: 0.053880

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| train_size | 304434 |
| val_size | 149185 |
| test_size | 149367 |
| best_epoch | 22 |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| learning_rate | 0.3620 |
| use_layer_norm | 0.2556 |
| hidden_size | 0.1009 |
| dropout | 0.0994 |
| input_length | 0.0746 |
| num_decoder_layers | 0.0487 |
| decoder_output_size | 0.0310 |
| num_encoder_layers | 0.0175 |
| temporal_decoder_hidden_size | 0.0103 |

## Visualizations

- [Optimization History](./tide_optimization_history.png)
- [Parameter Importance](./tide_param_importances.png)
- [Parameter Contours](./tide_param_contours.png)
- [Parameter Correlation](./tide_param_correlation.png)