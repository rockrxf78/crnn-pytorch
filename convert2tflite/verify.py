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

"""Verification script to compare PyTorch CRNN model with TFLite model."""

import os
import sys
import numpy as np
from absl import app
from absl import flags
import tensorflow as tf

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

import torch

# Define flags for verification
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    'The path to the model checkpoint. If None, downloads pretrained model from GitHub.',
)

_TFLITE_PATH = flags.DEFINE_string(
    'tflite_path',
    '/tmp/crnn_32x128_ch1_f32.tflite',
    'The path to the TFLite model for verification.',
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

_NUM_TESTS = flags.DEFINE_integer(
    'num_tests',
    5,
    'Number of random inputs to test.',
)

_TOLERANCE = flags.DEFINE_float(
    'tolerance',
    1e-4,
    'Tolerance for numerical comparison.',
)


def load_tflite_model(tflite_path):
  """Loads TFLite model and returns interpreter."""
  if not os.path.exists(tflite_path):
    raise FileNotFoundError(f"TFLite model not found at: {tflite_path}")
  
  interpreter = tf.lite.Interpreter(model_path=tflite_path)
  interpreter.allocate_tensors()
  return interpreter


def run_tflite_inference(interpreter, input_data):
  """Runs inference on TFLite model."""
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  # Set input tensor
  interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
  
  # Run inference
  interpreter.invoke()
  
  # Get output tensor
  output_data = interpreter.get_tensor(output_details[0]['index'])
  return output_data


def run_pytorch_inference(model, input_tensor):
  """Runs inference on PyTorch model."""
  with torch.no_grad():
    output = model(input_tensor)
  return output.numpy()


def compare_outputs(pytorch_output, tflite_output, tolerance=1e-4):
  """Compares outputs from PyTorch and TFLite models."""
  
  # Check shapes
  if pytorch_output.shape != tflite_output.shape:
    print(f"❌ Shape mismatch:")
    print(f"  PyTorch shape: {pytorch_output.shape}")
    print(f"  TFLite shape: {tflite_output.shape}")
    return False
  
  # Calculate differences
  abs_diff = np.abs(pytorch_output - tflite_output)
  max_abs_diff = np.max(abs_diff)
  mean_abs_diff = np.mean(abs_diff)
  
  # Calculate relative differences
  pytorch_abs = np.abs(pytorch_output)
  rel_diff = abs_diff / (pytorch_abs + 1e-8)  # Add small value to avoid division by zero
  max_rel_diff = np.max(rel_diff)
  mean_rel_diff = np.mean(rel_diff)
  
  print(f"Numerical comparison:")
  print(f"  Max absolute difference: {max_abs_diff:.6f}")
  print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
  print(f"  Max relative difference: {max_rel_diff:.6f}")
  print(f"  Mean relative difference: {mean_rel_diff:.6f}")
  
  # Check if within tolerance
  if max_abs_diff <= tolerance:
    print(f"✅ Outputs match within tolerance ({tolerance})")
    return True
  else:
    print(f"❌ Outputs exceed tolerance ({tolerance})")
    return False


def decode_and_compare_text(pytorch_output, tflite_output, alphabet):
  """Decodes both outputs to text and compares them."""
  
  # Convert numpy arrays to torch tensors for decoding
  pytorch_tensor = torch.from_numpy(pytorch_output)
  tflite_tensor = torch.from_numpy(tflite_output)
  
  # Decode predictions
  pytorch_text = crnn.decode_prediction(pytorch_tensor, alphabet, method='greedy')
  tflite_text = crnn.decode_prediction(tflite_tensor, alphabet, method='greedy')
  
  print(f"Text comparison:")
  print(f"  PyTorch decoded: '{pytorch_text}'")
  print(f"  TFLite decoded:  '{tflite_text}'")
  
  text_match = pytorch_text == tflite_text
  if text_match:
    print(f"✅ Decoded text matches!")
  else:
    print(f"❌ Decoded text differs!")
  
  return text_match, pytorch_text, tflite_text


def main(_):
  checkpoint_path = _CHECKPOINT_PATH.value
  tflite_path = _TFLITE_PATH.value
  img_height = _IMG_HEIGHT.value
  img_width = _IMG_WIDTH.value
  num_channels = _NUM_CHANNELS.value
  hidden_size = _HIDDEN_SIZE.value
  num_tests = _NUM_TESTS.value
  tolerance = _TOLERANCE.value
  
  print("🔍 CRNN Model Verification")
  print("=" * 50)
  
  # Load PyTorch model
  print(f"Loading PyTorch CRNN model...")
  pytorch_model = crnn.build_model(
      checkpoint_path=checkpoint_path,
      imgH=img_height,
      nc=num_channels,
      nh=hidden_size,
      download_if_missing=True
  )
  
  # Load TFLite model
  print(f"Loading TFLite model from: {tflite_path}")
  tflite_interpreter = load_tflite_model(tflite_path)
  
  # Get model info
  input_details = tflite_interpreter.get_input_details()
  output_details = tflite_interpreter.get_output_details()
  
  print(f"TFLite model info:")
  print(f"  Input shape: {input_details[0]['shape']}")
  print(f"  Input dtype: {input_details[0]['dtype']}")
  print(f"  Output shape: {output_details[0]['shape']}")
  print(f"  Output dtype: {output_details[0]['dtype']}")
  
  # Run verification tests
  print(f"\nRunning {num_tests} verification tests...")
  print("-" * 50)
  
  all_numerical_matches = []
  all_text_matches = []
  
  for i in range(num_tests):
    print(f"\nTest {i+1}/{num_tests}:")
    
    # Generate random input
    input_tensor = torch.randn(1, num_channels, img_height, img_width)
    input_numpy = input_tensor.numpy()
    
    # Run PyTorch inference
    pytorch_output = run_pytorch_inference(pytorch_model, input_tensor)
    
    # Run TFLite inference
    tflite_output = run_tflite_inference(tflite_interpreter, input_numpy)
    
    # Compare numerical outputs
    numerical_match = compare_outputs(pytorch_output, tflite_output, tolerance)
    all_numerical_matches.append(numerical_match)
    
    # Compare decoded text
    text_match, pytorch_text, tflite_text = decode_and_compare_text(
        pytorch_output, tflite_output, pytorch_model.alphabet
    )
    all_text_matches.append(text_match)
  
  # Summary
  print("\n" + "=" * 50)
  print("📊 VERIFICATION SUMMARY")
  print("=" * 50)
  
  numerical_success_rate = sum(all_numerical_matches) / len(all_numerical_matches) * 100
  text_success_rate = sum(all_text_matches) / len(all_text_matches) * 100
  
  print(f"Numerical accuracy: {numerical_success_rate:.1f}% ({sum(all_numerical_matches)}/{len(all_numerical_matches)})")
  print(f"Text decoding accuracy: {text_success_rate:.1f}% ({sum(all_text_matches)}/{len(all_text_matches)})")
  
  if numerical_success_rate == 100.0:
    print("✅ All numerical tests passed!")
  else:
    print("❌ Some numerical tests failed!")
  
  if text_success_rate == 100.0:
    print("✅ All text decoding tests passed!")
  else:
    print("❌ Some text decoding tests failed!")
  
  # Overall verdict
  if numerical_success_rate >= 95.0 and text_success_rate >= 95.0:
    print("\n🎉 VERIFICATION PASSED: Models are equivalent!")
    return 0
  else:
    print("\n❌ VERIFICATION FAILED: Models show significant differences!")
    return 1


if __name__ == '__main__':
  app.run(main) 