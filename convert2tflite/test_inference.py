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

"""Test script for CRNN model inference on text images."""

import os
import sys
import glob
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

# Define flags for inference testing
_IMAGE_PATH = flags.DEFINE_string(
    'image_path',
    None,
    'Path to a single image file or directory of images to process.',
)

_TFLITE_PATH = flags.DEFINE_string(
    'tflite_path',
    '/tmp/crnn_32x128_ch1_f32.tflite',
    'Path to the TFLite model.',
)

_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Path to PyTorch checkpoint for comparison (optional).',
)

_IMG_HEIGHT = flags.DEFINE_integer(
    'img_height',
    32,
    'Target image height.',
)

_IMG_WIDTH = flags.DEFINE_integer(
    'img_width',
    128,
    'Target image width.',
)

_NUM_CHANNELS = flags.DEFINE_integer(
    'num_channels',
    1,
    'Number of input channels (1 for grayscale, 3 for RGB).',
)

_HIDDEN_SIZE = flags.DEFINE_integer(
    'hidden_size',
    256,
    'Hidden size for LSTM layers (needed for PyTorch model).',
)

_COMPARE_PYTORCH = flags.DEFINE_bool(
    'compare_pytorch',
    False,
    'Whether to compare TFLite results with PyTorch model.',
)

_CREATE_SYNTHETIC = flags.DEFINE_bool(
    'create_synthetic',
    True,
    'Whether to create synthetic text images if no image path provided.',
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


def create_synthetic_text_image(text, img_height=32, img_width=128, num_channels=1):
  """Creates a synthetic text image for testing."""
  try:
    from PIL import Image, ImageDraw, ImageFont
  except ImportError:
    print("PIL not available, creating random noise image")
    # Return random noise as fallback
    if num_channels == 1:
      return np.random.randint(0, 256, (img_height, img_width), dtype=np.uint8)
    else:
      return np.random.randint(0, 256, (img_height, img_width, num_channels), dtype=np.uint8)
  
  # Create image
  if num_channels == 1:
    image = Image.new('L', (img_width, img_height), color=255)  # White background
  else:
    image = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
  
  draw = ImageDraw.Draw(image)
  
  # Try to use default font
  try:
    font = ImageFont.load_default()
  except Exception:
    font = None
  
  # Calculate text position (center)
  if font:
    try:
      bbox = draw.textbbox((0, 0), text, font=font)
      text_width = bbox[2] - bbox[0]
      text_height = bbox[3] - bbox[1]
    except AttributeError:
      # Fallback for older PIL versions
      text_width, text_height = draw.textsize(text, font=font)
  else:
    text_width = len(text) * 8  # Rough estimate
    text_height = 12
  
  x = max(0, (img_width - text_width) // 2)
  y = max(0, (img_height - text_height) // 2)
  
  # Draw text
  text_color = 0 if num_channels == 1 else (0, 0, 0)  # Black text
  draw.text((x, y), text, fill=text_color, font=font)
  
  return np.array(image)


def preprocess_image_array(image_array, img_height=32, img_width=128, num_channels=1):
  """Preprocesses image array for CRNN inference."""
  try:
    from PIL import Image
    import torchvision.transforms as transforms
  except ImportError:
    raise ImportError(
        "PIL and torchvision are required for image preprocessing. "
        "Install them with: pip install Pillow torchvision"
    )
  
  # Convert to PIL Image
  if len(image_array.shape) == 2:  # Grayscale
    image = Image.fromarray(image_array, 'L')
  elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
    image = Image.fromarray(image_array, 'RGB')
  else:
    raise ValueError(f"Unsupported image array shape: {image_array.shape}")
  
  # Convert to target format
  if num_channels == 1:
    image = image.convert('L')  # Grayscale
  else:
    image = image.convert('RGB')
  
  # Resize to target dimensions
  image = image.resize((img_width, img_height), Image.LANCZOS)
  
  # Convert to tensor and normalize
  if num_channels == 1:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
  else:
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  
  tensor = transform(image).unsqueeze(0)  # Add batch dimension
  return tensor


def decode_output_to_text(output_data, alphabet):
  """Decodes model output to text using CTC greedy decoding."""
  # Convert to torch tensor for decoding
  output_tensor = torch.from_numpy(output_data)
  return crnn.decode_prediction(output_tensor, alphabet, method='greedy')


def process_image_file(image_path, interpreter, alphabet, img_height, img_width, num_channels):
  """Processes a single image file."""
  print(f"\nProcessing: {image_path}")
  
  try:
    # Load and preprocess image
    input_tensor = crnn.preprocess_image(
        image_path, imgH=img_height, imgW=img_width, nc=num_channels
    )
    input_numpy = input_tensor.numpy()
    
    # Run TFLite inference
    output_data = run_tflite_inference(interpreter, input_numpy)
    
    # Decode to text
    predicted_text = decode_output_to_text(output_data, alphabet)
    
    print(f"  Predicted text: '{predicted_text}'")
    print(f"  Output shape: {output_data.shape}")
    print(f"  Confidence scores: min={output_data.min():.3f}, max={output_data.max():.3f}")
    
    return predicted_text
    
  except Exception as e:
    print(f"  Error processing image: {e}")
    return None


def main(_):
  image_path = _IMAGE_PATH.value
  tflite_path = _TFLITE_PATH.value
  checkpoint_path = _CHECKPOINT_PATH.value
  img_height = _IMG_HEIGHT.value
  img_width = _IMG_WIDTH.value
  num_channels = _NUM_CHANNELS.value
  hidden_size = _HIDDEN_SIZE.value
  compare_pytorch = _COMPARE_PYTORCH.value
  create_synthetic = _CREATE_SYNTHETIC.value
  
  print("🔤 CRNN Text Recognition Inference Test")
  print("=" * 50)
  
  # Load TFLite model
  print(f"Loading TFLite model from: {tflite_path}")
  interpreter = load_tflite_model(tflite_path)
  
  # Get alphabet (create dummy model to get alphabet)
  dummy_model = crnn.build_model(
      checkpoint_path=None, 
      imgH=img_height, 
      nc=num_channels, 
      nh=hidden_size,
      download_if_missing=False
  )
  alphabet = dummy_model.alphabet
  print(f"Model alphabet: {alphabet}")
  print(f"Number of classes: {len(alphabet) + 1} (including blank)")
  
  # Load PyTorch model for comparison if requested
  pytorch_model = None
  if compare_pytorch:
    print(f"Loading PyTorch model for comparison...")
    pytorch_model = crnn.build_model(
        checkpoint_path=checkpoint_path,
        imgH=img_height,
        nc=num_channels,
        nh=hidden_size,
        download_if_missing=True
    )
  
  # Determine what images to process
  image_files = []
  
  if image_path:
    if os.path.isfile(image_path):
      image_files = [image_path]
    elif os.path.isdir(image_path):
      # Look for common image extensions
      extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
      for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_path, ext)))
        image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))
    else:
      print(f"❌ Image path not found: {image_path}")
      return 1
  
  # Create synthetic images if no real images provided
  if not image_files and create_synthetic:
    print(f"No images provided, creating synthetic test images...")
    
    # Test texts
    test_texts = [
        "hello",
        "world", 
        "CRNN",
        "12345",
        "Test123",
        "AI",
        "pytorch",
        "tflite"
    ]
    
    for i, text in enumerate(test_texts):
      print(f"\n📝 Synthetic test {i+1}: '{text}'")
      
      # Create synthetic image
      image_array = create_synthetic_text_image(
          text, img_height, img_width, num_channels
      )
      
      # Preprocess
      input_tensor = preprocess_image_array(
          image_array, img_height, img_width, num_channels
      )
      input_numpy = input_tensor.numpy()
      
      # Run TFLite inference
      output_data = run_tflite_inference(interpreter, input_numpy)
      predicted_text = decode_output_to_text(output_data, alphabet)
      
      print(f"  Ground truth: '{text}'")
      print(f"  TFLite predicted: '{predicted_text}'")
      
      # Compare with PyTorch if available
      if pytorch_model:
        with torch.no_grad():
          pytorch_output = pytorch_model(input_tensor).numpy()
          pytorch_text = decode_output_to_text(pytorch_output, alphabet)
          print(f"  PyTorch predicted: '{pytorch_text}'")
          
          # Check if predictions match
          if pytorch_text == predicted_text:
            print(f"  ✅ PyTorch and TFLite predictions match!")
          else:
            print(f"  ❌ PyTorch and TFLite predictions differ!")
  
  # Process real image files
  if image_files:
    print(f"\nProcessing {len(image_files)} image files...")
    print("-" * 30)
    
    for image_file in image_files:
      predicted_text = process_image_file(
          image_file, interpreter, alphabet, 
          img_height, img_width, num_channels
      )
      
      # Compare with PyTorch if available
      if pytorch_model and predicted_text is not None:
        try:
          input_tensor = crnn.preprocess_image(
              image_file, imgH=img_height, imgW=img_width, nc=num_channels
          )
          with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor).numpy()
            pytorch_text = decode_output_to_text(pytorch_output, alphabet)
            print(f"  PyTorch predicted: '{pytorch_text}'")
            
            if pytorch_text == predicted_text:
              print(f"  ✅ PyTorch and TFLite predictions match!")
            else:
              print(f"  ❌ PyTorch and TFLite predictions differ!")
        except Exception as e:
          print(f"  Error in PyTorch comparison: {e}")
  
  if not image_files and not create_synthetic:
    print("❌ No images to process and synthetic image creation disabled")
    return 1
  
  print(f"\n✅ Inference testing completed!")
  return 0


if __name__ == '__main__':
  app.run(main) 