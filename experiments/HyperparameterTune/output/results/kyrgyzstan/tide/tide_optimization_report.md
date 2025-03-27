# Hyperparameter Optimization Report for TIDE - Kyrgyzstan

Date: 2025-03-27 15:02:05
Number of trials: 15
Best trial: #14

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 34 |
| hidden_size | 110 |
| dropout | 0.4040330172235821 |
| learning_rate | 0.00029399848560567596 |
| num_encoder_layers | 2 |
| num_decoder_layers | 2 |
| decoder_output_size | 24 |
| temporal_decoder_hidden_size | 51 |
| use_layer_norm | False |

**Best validation loss**: 0.054531

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| basin_count | 61 |
| country | Kyrgyzstan |
| model_type | tide |
| train_size | 208949 |
| val_size | 103175 |
| test_size | 103228 |
| best_epoch | 29 |
| checkpoint_path | experiments/HyperparameterTune/output/checkpoints/kyrgyzstan/tide/trial_14/last.ckpt |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| use_layer_norm | 0.7779 |
| learning_rate | 0.0573 |
| dropout | 0.0573 |
| hidden_size | 0.0359 |
| num_decoder_layers | 0.0300 |
| input_length | 0.0221 |
| temporal_decoder_hidden_size | 0.0108 |
| num_encoder_layers | 0.0069 |
| decoder_output_size | 0.0018 |

## Dataset Information

- **Country**: Kyrgyzstan
- **Number of basins**: 61
- **Training samples**: 208949
- **Validation samples**: 103175
- **Testing samples**: 103228