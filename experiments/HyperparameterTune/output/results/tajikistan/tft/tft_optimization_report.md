# Hyperparameter Optimization Report for TFT - Tajikistan

Date: 2025-03-27 12:56:21
Number of trials: 15
Best trial: #13

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 190 |
| hidden_size | 48 |
| dropout | 0.09262986969204212 |
| learning_rate | 0.0002993620086989372 |
| num_attention_heads | 8 |
| lstm_layers | 3 |
| attn_dropout | 0.12346441473728025 |
| add_relative_index | False |
| use_revin | False |
| context_length_ratio | 0.5125053734835204 |
| use_embedding_for_context | True |
| encoder_layers | 1 |

**Best validation loss**: 0.052559

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| basin_count | 16 |
| country | Tajikistan |
| model_type | tft |
| train_size | 51658 |
| val_size | 24233 |
| test_size | 24253 |
| best_epoch | 62 |
| checkpoint_path | experiments/HyperparameterTune/output/checkpoints/tajikistan/tft/trial_13/last.ckpt |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| hidden_size | 0.2165 |
| context_length_ratio | 0.2018 |
| dropout | 0.1626 |
| attn_dropout | 0.1446 |
| input_length | 0.0902 |
| num_attention_heads | 0.0650 |
| learning_rate | 0.0561 |
| add_relative_index | 0.0207 |
| encoder_layers | 0.0171 |
| use_revin | 0.0139 |
| use_embedding_for_context | 0.0074 |
| lstm_layers | 0.0041 |

## Dataset Information

- **Country**: Tajikistan
- **Number of basins**: 16
- **Training samples**: 51658
- **Validation samples**: 24233
- **Testing samples**: 24253