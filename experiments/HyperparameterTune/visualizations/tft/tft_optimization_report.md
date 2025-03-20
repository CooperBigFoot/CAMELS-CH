# Hyperparameter Optimization Report for TFT

Date: 2025-03-20 18:58:44
Number of trials: 10
Best trial: #1

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 206 |
| hidden_size | 72 |
| dropout | 0.14561457009902096 |
| learning_rate | 0.00016738085788752134 |
| num_attention_heads | 2 |
| lstm_layers | 1 |
| variable_selection_method | dot_product |
| attn_dropout | 0.23555278841790406 |
| add_relative_index | False |
| use_revin | True |
| context_length_ratio | 0.8037724259507192 |
| use_embedding_for_context | True |
| encoder_layers | 3 |

**Best validation loss**: 0.025899

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| train_size | 294098 |
| val_size | 138849 |
| test_size | 139031 |
| best_epoch | 50 |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| add_relative_index | 0.2648 |
| context_length_ratio | 0.1764 |
| learning_rate | 0.1590 |
| num_attention_heads | 0.0924 |
| attn_dropout | 0.0850 |
| hidden_size | 0.0575 |
| encoder_layers | 0.0521 |
| use_embedding_for_context | 0.0379 |
| dropout | 0.0341 |
| input_length | 0.0325 |
| variable_selection_method | 0.0052 |
| lstm_layers | 0.0018 |
| use_revin | 0.0014 |

## Visualizations

- [Optimization History](./tft_optimization_history.png)
- [Parameter Importance](./tft_param_importances.png)
- [Parameter Contours](./tft_param_contours.png)
- [Parameter Correlation](./tft_param_correlation.png)