# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example of converting CRNN model to TFLite model."""

import os
import sys
import pathlib
from absl import app
from absl import flags

# Add current directory to Python path to handle relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from crnn import crnn
except ImportError:
    # Try absolute import if relative import fails
    from ai_edge_torch.generative.examples.crnn import crnn

import ai_edge_torch
import torch

# Define flags for CRNN conversion
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    'The path to the model checkpoint. If None, downloads pretrained model from GitHub.',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    '/tmp/',
    'The path to export the tflite model.',
)

_OUTPUT_NAME_PREFIX = flags.DEFINE_string(
    'output_name_prefix',
    'crnn',
    'The prefix of the output tflite model name.',
)

_IMG_HEIGHT = flags.DEFINE_integer(
    'img_height',
    32,
    'Input image height (must be multiple of 16).',
)

_IMG_WIDTH = flags.DEFINE_integer(
    'img_width',
    128,
    'Input image width.',
)

_NUM_CHANNELS = flags.DEFINE_integer(
    'num_channels',
    1,
    'Number of input channels (1 for grayscale, 3 for RGB).',
)

_HIDDEN_SIZE = flags.DEFINE_integer(
    'hidden_size',
    256,
    'Hidden size for LSTM layers.',
)

_QUANTIZE = flags.DEFINE_bool(
    'quantize',
    False,  # Start with False to avoid compatibility issues
    'Whether to apply quantization to the model.',
)


def apply_simple_quantization(pytorch_model, sample_input):
  """Applies simple quantization to the model without PT2E complications."""
  try:
    print("Applying simple quantization...")
    
    # Apply dynamic quantization directly to the model (avoid tracing)
    quantized_model = torch.quantization.quantize_dynamic(
        pytorch_model,
        {torch.nn.LSTM, torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized_model
    
  except Exception as e:
    print(f"Simple quantization failed: {e}")
    print("Trying without quantization...")
    return None


def main(_):
  checkpoint_path = _CHECKPOINT_PATH.value
  output_path = _OUTPUT_PATH.value
  output_name_prefix = _OUTPUT_NAME_PREFIX.value
  img_height = _IMG_HEIGHT.value
  img_width = _IMG_WIDTH.value
  num_channels = _NUM_CHANNELS.value
  hidden_size = _HIDDEN_SIZE.value
  quantize = _QUANTIZE.value
  
  # Validate image height
  if img_height % 16 != 0:
    raise ValueError(f"Image height ({img_height}) must be a multiple of 16")
  
  # Load the CRNN model
  print(f"Loading CRNN model...")
  print(f"Model config: {img_height}x{img_width}, {num_channels} channels, hidden_size={hidden_size}")
  
  pytorch_model = crnn.build_model(
      checkpoint_path=checkpoint_path,
      imgH=img_height,
      nc=num_channels,
      nh=hidden_size,
      download_if_missing=True
  )
  
  # Get sample input for the model
  sample_input = crnn.get_sample_input(
      imgH=img_height, 
      imgW=img_width, 
      nc=num_channels
  )
  
  # Apply quantization if requested
  quant_suffix = "f32"
  final_model = pytorch_model
  
  if quantize:
    quantized_model = apply_simple_quantization(pytorch_model, sample_input)
    if quantized_model is not None:
      final_model = quantized_model
      quant_suffix = "q8"
      print("✅ Simple quantization applied successfully!")
    else:
      print("⚠️ Quantization failed, using float32 model")
  
  print("Converting model to TFLite...")
  
  # Convert the model to TFLite
  try:
    edge_model = ai_edge_torch.convert(
        final_model, 
        (sample_input,)
    )
  except Exception as e:
    print(f"Conversion failed: {e}")
    print("Trying with scripted model...")
    
    # Try with torch.jit.script as fallback
    try:
      scripted_model = torch.jit.script(final_model)
      edge_model = ai_edge_torch.convert(
          scripted_model,
          (sample_input,)
      )
    except Exception as e2:
      print(f"Scripted conversion also failed: {e2}")
      raise RuntimeError("Both direct and scripted conversion failed")
  
  # Create output filename
  output_filename = f"{output_name_prefix}_{img_height}x{img_width}_ch{num_channels}_{quant_suffix}.tflite"
  output_file = os.path.join(output_path, output_filename)
  
  # Export the model
  print(f"Exporting model to: {output_file}")
  edge_model.export(output_file)
  
  print(f"✅ Successfully converted CRNN to TFLite!")
  print(f"Model source: {'Downloaded from GitHub' if checkpoint_path is None else checkpoint_path}")
  print(f"Output file: {output_file}")
  print(f"Quantization: {'INT8 (Dynamic)' if quant_suffix == 'q8' else 'Float32'}")
  print(f"Input shape: {sample_input.shape} (batch_size, channels, height, width)")
  print(f"Output shape: [sequence_length, batch_size, num_classes] where num_classes = {len(pytorch_model.alphabet) + 1}")
  print(f"Expected input: Grayscale images resized to {img_width}x{img_height}, normalized to [-1,1]")
  print(f"Model alphabet: {pytorch_model.alphabet}")
  
  # Check file size
  if os.path.exists(output_file):
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"Model size: {file_size:.2f} MB")


if __name__ == '__main__':
  app.run(main) 