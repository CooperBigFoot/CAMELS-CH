# Hyperparameter Optimization Report for TFT

Date: 2025-03-21 18:40:45
Number of trials: 10
Best trial: #4

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 55 |
| hidden_size | 128 |
| dropout | 0.3861223846483287 |
| learning_rate | 2.497073714505272e-05 |
| num_attention_heads | 1 |
| lstm_layers | 3 |
| variable_selection_method | dot_product |
| attn_dropout | 0.2313811040057837 |
| add_relative_index | False |
| use_revin | False |
| context_length_ratio | 0.811649063413779 |
| use_embedding_for_context | True |
| encoder_layers | 1 |

**Best validation loss**: 0.052637

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| train_size | 305574 |
| val_size | 150325 |
| test_size | 150507 |
| best_epoch | 50 |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| use_revin | 0.2574 |
| input_length | 0.1347 |
| hidden_size | 0.1234 |
| use_embedding_for_context | 0.1159 |
| context_length_ratio | 0.0813 |
| num_attention_heads | 0.0780 |
| learning_rate | 0.0747 |
| attn_dropout | 0.0626 |
| dropout | 0.0509 |
| add_relative_index | 0.0072 |
| encoder_layers | 0.0070 |
| variable_selection_method | 0.0062 |
| lstm_layers | 0.0006 |

## Visualizations

- [Optimization History](./tft_optimization_history.png)
- [Parameter Importance](./tft_param_importances.png)
- [Parameter Contours](./tft_param_contours.png)
- [Parameter Correlation](./tft_param_correlation.png)