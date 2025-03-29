# Hyperparameter Optimization Report for TFT - Kyrgyzstan

Date: 2025-03-29 16:01:06
Number of trials: 15
Best trial: #13

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 34 |
| learning_rate | 0.00015996162179839556 |
| hidden_size | 120 |
| dropout | 0.3690491710631163 |
| num_attention_heads | 3 |
| lstm_layers | 1 |
| attn_dropout | 0.12456198490898998 |
| add_relative_index | True |
| use_revin | False |
| context_length_ratio | 0.9286955650046743 |
| use_embedding_for_context | True |
| encoder_layers | 1 |

**Best validation loss**: 0.054412

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| basin_count | 61 |
| country | Kyrgyzstan |
| model_type | tft |
| train_size | 208949 |
| val_size | 103175 |
| test_size | 103228 |
| best_epoch | 23 |
| checkpoint_path | experiments/HyperparameterTune/output/checkpoints/kyrgyzstan/tft/trial_13/last.ckpt |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| add_relative_index | 0.2706 |
| dropout | 0.2379 |
| context_length_ratio | 0.1602 |
| hidden_size | 0.1241 |
| learning_rate | 0.1064 |
| num_attention_heads | 0.0486 |
| attn_dropout | 0.0314 |
| lstm_layers | 0.0124 |
| input_length | 0.0040 |
| use_revin | 0.0039 |
| encoder_layers | 0.0006 |
| use_embedding_for_context | 0.0000 |

## Dataset Information

- **Country**: Kyrgyzstan
- **Number of basins**: 61
- **Training samples**: 208949
- **Validation samples**: 103175
- **Testing samples**: 103228