# Hyperparameter Optimization Report for TIDE - Tajikistan

Date: 2025-03-27 11:10:28
Number of trials: 15
Best trial: #11

## Best Parameters

| Parameter | Value |
| --------- | ----- |
| input_length | 99 |
| hidden_size | 76 |
| dropout | 0.3078685206868833 |
| learning_rate | 0.00018159850484411317 |
| num_encoder_layers | 3 |
| num_decoder_layers | 2 |
| decoder_output_size | 22 |
| temporal_decoder_hidden_size | 32 |
| use_layer_norm | False |

**Best validation loss**: 0.061865

## Best Trial Details

| Attribute | Value |
| --------- | ----- |
| basin_count | 16 |
| country | Tajikistan |
| model_type | tide |
| train_size | 53114 |
| val_size | 25689 |
| test_size | 25709 |
| best_epoch | 57 |
| checkpoint_path | experiments/HyperparameterTune/output/checkpoints/tajikistan/tide/trial_11/last.ckpt |

## Parameter Importance

| Parameter | Importance |
| --------- | ---------- |
| learning_rate | 0.4210 |
| hidden_size | 0.2175 |
| use_layer_norm | 0.1445 |
| input_length | 0.0922 |
| num_decoder_layers | 0.0452 |
| decoder_output_size | 0.0308 |
| dropout | 0.0291 |
| num_encoder_layers | 0.0131 |
| temporal_decoder_hidden_size | 0.0065 |

## Dataset Information

- **Country**: Tajikistan
- **Number of basins**: 16
- **Training samples**: 53114
- **Validation samples**: 25689
- **Testing samples**: 25709