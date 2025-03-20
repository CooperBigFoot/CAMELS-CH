# Hyperparameter Optimization Report for TFT

Date: 2025-03-20 15:02:33
Number of trials: 10
Best trial: #0

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 155 |
| hidden_size | 124 |
| dropout | 0.36599697090570255 |
| learning_rate | 0.00015751320499779721 |
| num_attention_heads | 2 |
| lstm_layers | 1 |
| variable_selection_method | dot_product |
| attn_dropout | 0.18033450352296262 |
| add_relative_index | True |
| use_revin | True |
| context_length_ratio | 0.6061695553391381 |
| use_embedding_for_context | False |
| encoder_layers | 1 |

**Best validation loss**: 0.026287

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| train_size | 297974 |
| val_size | 142725 |
| test_size | 142907 |
| best_epoch | 47 |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| add_relative_index | 0.1634 |
| hidden_size | 0.1499 |
| attn_dropout | 0.1280 |
| use_embedding_for_context | 0.1123 |
| context_length_ratio | 0.0943 |
| use_revin | 0.0942 |
| learning_rate | 0.0918 |
| input_length | 0.0570 |
| num_attention_heads | 0.0531 |
| dropout | 0.0374 |
| lstm_layers | 0.0187 |
| variable_selection_method | 0.0000 |
| encoder_layers | 0.0000 |

## Visualizations

- [Optimization History](./tft_optimization_history.png)
- [Parameter Importance](./tft_param_importances.png)
- [Parameter Contours](./tft_param_contours.png)
- [Parameter Correlation](./tft_param_correlation.png)