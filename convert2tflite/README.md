# CRNN to TFLite Conversion

This directory contains scripts to convert CRNN (Convolutional Recurrent Neural Network) models for text recognition to TFLite format.

## Overview

CRNN combines CNN features with RNN sequence modeling for optical character recognition (OCR). This implementation is based on the paper:

> "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" (2016) by Baoguang Shi et al.

**Key Features:**
- Converts CRNN models from [crnn-pytorch](https://github.com/GitYCC/crnn-pytorch) to TFLite
- Supports automatic pretrained model downloading from GitHub
- Includes quantization options for model compression
- Provides comprehensive verification and testing scripts
- Handles both grayscale and RGB input images

## Architecture

The CRNN architecture consists of:
1. **CNN Backbone**: 7 convolutional layers with pooling for feature extraction
2. **RNN Layers**: 2 bidirectional LSTM layers for sequence modeling  
3. **CTC Output**: Connectionist Temporal Classification for variable-length text prediction

**Model Specifications:**
- Input: `[batch, channels, height, width]` where height must be multiple of 16
- Default: `[1, 1, 32, 128]` (grayscale images 32x128)
- Output: `[sequence_length, batch, num_classes]` 
- Classes: 62 alphanumeric characters + 1 CTC blank token (63 total)
- Alphabet: `0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`

## Files

- `crnn.py`: CRNN model definition and utilities
- `convert_to_tflite.py`: Main conversion script
- `verify.py`: Verification script comparing PyTorch vs TFLite outputs  
- `test_inference.py`: Inference testing on text images
- `README.md`: This documentation

## Quick Start

### 1. Convert CRNN to TFLite

Basic conversion with default settings:

```bash
python convert_to_tflite.py
```

This will:
- Download the pretrained model from GitHub (~8MB)
- Convert to TFLite format (float32)
- Save to `/tmp/crnn_32x128_ch1_f32.tflite` 

### 2. Custom Conversion Options

```bash
# Convert with custom dimensions
python convert_to_tflite.py \
  --img_height=32 \
  --img_width=256 \
  --num_channels=1 \
  --output_path=/path/to/output/

# Convert RGB model with quantization
python convert_to_tflite.py \
  --num_channels=3 \
  --quantize=true \
  --output_name_prefix=crnn_rgb

# Use custom checkpoint
python convert_to_tflite.py \
  --checkpoint_path=/path/to/checkpoint.pt
```

### 3. Verify Conversion

Compare PyTorch and TFLite model outputs:

```bash
python verify.py \
  --tflite_path=/tmp/crnn_32x128_ch1_f32.tflite \
  --num_tests=10
```

### 4. Test Inference

Test on synthetic or real text images:

```bash
# Test with synthetic images
python test_inference.py \
  --tflite_path=/tmp/crnn_32x128_ch1_f32.tflite

# Test on real images
python test_inference.py \
  --tflite_path=/tmp/crnn_32x128_ch1_f32.tflite \
  --image_path=/path/to/text/images/ \
  --compare_pytorch=true
```

## Input Preprocessing

CRNN expects preprocessed images:

1. **Resize**: Scale to target dimensions (default: 128x32)
2. **Normalize**: Convert to range [-1, 1] using `(pixel/255 - 0.5) / 0.5`
3. **Format**: Grayscale (1 channel) or RGB (3 channels)

Example preprocessing:

```python
from crnn import crnn

# Load and preprocess image
input_tensor = crnn.preprocess_image(
    "text_image.jpg", 
    imgH=32, 
    imgW=128, 
    nc=1  # Grayscale
)

# Or create sample input
sample_input = crnn.get_sample_input(imgH=32, imgW=128, nc=1)
```

## Output Decoding

CRNN outputs logits for each time step. Use CTC decoding to get text:

```python
import torch
from crnn import crnn

# Assume 'output' is model output tensor [seq_len, batch, num_classes]
alphabet = crnn.create_alphabet()
predicted_text = crnn.decode_prediction(output, alphabet, method='greedy')
print(f"Predicted: {predicted_text}")
```

## Model Variants

### Default Configuration
- **Input**: 128x32 grayscale images
- **Parameters**: ~8.7M (CNN: 6.4M, RNN: 2.3M)
- **Model Size**: ~35MB (float32), ~9MB (quantized)
- **Use Case**: General text recognition

### Custom Configurations

```bash
# Wider images for longer text
python convert_to_tflite.py --img_width=256

# RGB images 
python convert_to_tflite.py --num_channels=3

# Larger LSTM hidden size
python convert_to_tflite.py --hidden_size=512
```

## Quantization

Two quantization approaches are supported:

### 1. Simple Dynamic Quantization (Recommended)
```bash
python convert_to_tflite.py --quantize=true
```

This applies dynamic quantization to LSTM and Linear layers, typically reducing model size by ~75% with minimal accuracy loss.

### 2. Advanced PT2E Quantization
For more advanced quantization, see the commented PT2E code in `convert_to_tflite.py`. Note that this requires specific PyTorch versions and may have compatibility issues.

## Performance Notes

- **Inference Speed**: ~2-5ms per image on CPU
- **Memory Usage**: ~50MB for float32 model, ~15MB for quantized
- **Accuracy**: 93%+ on Synth90k dataset (from original paper)
- **Text Length**: Works best with 4-10 character sequences

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're running from the correct directory
   ```bash
   cd ai_edge_torch/generative/examples/crnn
   python convert_to_tflite.py
   ```

2. **Height Not Multiple of 16**: 
   ```
   ValueError: Image height must be a multiple of 16
   ```
   Use heights like 16, 32, 48, 64, etc.

3. **Download Failures**: If GitHub download fails, manually download:
   ```bash
   mkdir -p ~/.cache/crnn_pytorch
   wget https://github.com/GitYCC/crnn-pytorch/raw/master/checkpoints/crnn_synth90k.pt \
        -O ~/.cache/crnn_pytorch/crnn_synth90k.pt
   ```

4. **Quantization Errors**: If quantization fails, disable it:
   ```bash
   python convert_to_tflite.py --quantize=false
   ```

### Dependencies

Required packages:
- `ai_edge_torch`
- `torch` 
- `tensorflow` (for TFLite interpreter)
- `numpy`
- `requests` (for model download)
- `PIL` (for image processing)
- `torchvision` (for transforms)
- `absl-py` (for flags)

Install with:
```bash
pip install torch torchvision tensorflow pillow requests absl-py
```

## Citation

If you use this CRNN conversion in your work, please cite:

```bibtex
@article{shi2016end,
  title={An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition},
  author={Shi, Baoguang and Bai, Xiang and Yao, Cong},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={39},
  number={11},
  pages={2298--2304},
  year={2016},
  publisher={IEEE}
}
```

Original PyTorch implementation:
```bibtex
@misc{crnn-pytorch,
  author = {GitYCC},
  title = {CRNN PyTorch Implementation},
  url = {https://github.com/GitYCC/crnn-pytorch},
  year = {2020}
}
```

## License

This code is licensed under the Apache License 2.0. The pretrained model weights are from the original crnn-pytorch repository which uses the MIT license. 