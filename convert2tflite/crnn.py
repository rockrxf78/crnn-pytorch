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

"""Example of building a CRNN model for text recognition."""

import os
import pathlib
import requests
from typing import Callable, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
  """Bidirectional LSTM layer for sequence modeling."""
  
  def __init__(self, nIn, nHidden, nOut):
    super(BidirectionalLSTM, self).__init__()
    self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
    self.embedding = nn.Linear(nHidden * 2, nOut)

  def forward(self, input):
    recurrent, _ = self.rnn(input)
    T, b, h = recurrent.size()
    t_rec = recurrent.view(T * b, h)
    output = self.embedding(t_rec)  # [T * b, nOut]
    output = output.view(T, b, -1)
    return output


class CRNN(nn.Module):
  """CRNN architecture for text recognition."""
  
  def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
    super(CRNN, self).__init__()
    assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]

    cnn = nn.Sequential()

    def convRelu(i, batchNormalization=False):
      nIn = nc if i == 0 else nm[i - 1]
      nOut = nm[i]
      cnn.add_module('conv{0}'.format(i),
                     nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
      if batchNormalization:
        cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
      if leakyRelu:
        cnn.add_module('relu{0}'.format(i),
                       nn.LeakyReLU(0.2, inplace=True))
      else:
        cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

    convRelu(0)
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
    convRelu(1)
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
    convRelu(2, True)
    convRelu(3)
    cnn.add_module('pooling{0}'.format(2),
                   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
    convRelu(4, True)
    convRelu(5)
    cnn.add_module('pooling{0}'.format(3),
                   nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
    convRelu(6, True)  # 512x1x16

    self.cnn = cnn
    self.rnn = nn.Sequential(
        BidirectionalLSTM(512, nh, nh),
        BidirectionalLSTM(nh, nh, nclass))

  def forward(self, input):
    # conv features
    conv = self.cnn(input)
    b, c, h, w = conv.size()
    assert h == 1, "the height of conv must be 1"
    conv = conv.squeeze(2)
    conv = conv.permute(2, 0, 1)  # [w, b, c]

    # rnn features
    output = self.rnn(conv)
    return output


def download_file(url: str, destination: str) -> bool:
  """Downloads a file from URL to destination."""
  try:
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
      for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
    
    print(f"Downloaded to {destination}")
    return True
  except Exception as e:
    print(f"Download failed: {e}")
    return False


def download_pretrained_model(model_dir: str) -> str:
  """Downloads the pretrained CRNN model from GitHub."""
  os.makedirs(model_dir, exist_ok=True)
  
  # Download the pretrained model
  model_url = "https://github.com/GitYCC/crnn-pytorch/raw/master/checkpoints/crnn_synth90k.pt"
  model_path = os.path.join(model_dir, "crnn_synth90k.pt")
  
  if not os.path.exists(model_path):
    if download_file(model_url, model_path):
      return model_path
    else:
      raise RuntimeError(f"Failed to download model from {model_url}")
  else:
    print(f"Model already exists at {model_path}")
    return model_path


def create_alphabet():
  """Creates the alphabet used by the CRNN model."""
  # Standard alphabet for Synth90k dataset (digits + lowercase + uppercase)
  alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  return alphabet


def build_model(
    checkpoint_path: Optional[str] = None,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    imgH: int = 32,
    nc: int = 1,
    nh: int = 256,
    download_if_missing: bool = True,
) -> nn.Module:
  """Builds and loads a CRNN model from checkpoint.
  
  Args:
    checkpoint_path: Path to model checkpoint or None to download pretrained
    custom_loader: Optional custom loader function (unused)
    imgH: Height of input images (must be multiple of 16)
    nc: Number of input channels (1 for grayscale, 3 for RGB)
    nh: Hidden size for LSTM layers
    download_if_missing: Whether to download pretrained model if no path provided
    
  Returns:
    Loaded CRNN model ready for inference
  """
  
  # Create alphabet and determine number of classes
  alphabet = create_alphabet()
  nclass = len(alphabet) + 1  # +1 for CTC blank token
  
  # Create model
  model = CRNN(imgH=imgH, nc=nc, nclass=nclass, nh=nh)
  
  # Load weights
  if checkpoint_path is None and download_if_missing:
    # Download pretrained model
    model_dir = os.path.expanduser("~/.cache/crnn_pytorch")
    checkpoint_path = download_pretrained_model(model_dir)
  
  if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading CRNN model from: {checkpoint_path}")
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in state_dict:
      state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
      state_dict = state_dict['model']
    
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model loaded successfully!")
  else:
    print("⚠️ No checkpoint found, using randomly initialized model")
  
  # Attach alphabet for reference
  model.alphabet = alphabet
  model.imgH = imgH
  model.nc = nc
  
  # Set to evaluation mode
  model.eval()
  
  return model


def get_sample_input(imgH: int = 32, imgW: int = 128, nc: int = 1):
  """Returns a sample input tensor for the model.
  
  Args:
    imgH: Image height (default 32)
    imgW: Image width (default 128) 
    nc: Number of channels (1 for grayscale, 3 for RGB)
    
  Returns:
    Sample input tensor of shape [1, nc, imgH, imgW]
  """
  return torch.randn(1, nc, imgH, imgW)


def preprocess_image(image_path: str, imgH: int = 32, imgW: int = 128, nc: int = 1):
  """Preprocesses an image for CRNN inference.
  
  Args:
    image_path: Path to the image file
    imgH: Target image height
    imgW: Target image width  
    nc: Number of channels (1 for grayscale, 3 for RGB)
    
  Returns:
    Preprocessed image tensor
  """
  try:
    from PIL import Image
    import torchvision.transforms as transforms
  except ImportError:
    raise ImportError(
        "PIL and torchvision are required for image preprocessing. "
        "Install them with: pip install Pillow torchvision"
    )
  
  # Load image
  image = Image.open(image_path)
  
  # Convert to grayscale or RGB based on nc
  if nc == 1:
    image = image.convert('L')  # Grayscale
  else:
    image = image.convert('RGB')
  
  # Resize to target dimensions
  image = image.resize((imgW, imgH), Image.LANCZOS)
  
  # Convert to tensor and normalize
  if nc == 1:
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


def decode_prediction(output, alphabet, method='greedy'):
  """Decodes CRNN output to text string.
  
  Args:
    output: Model output tensor [seq_len, batch, num_classes]
    alphabet: Alphabet string
    method: Decoding method ('greedy', 'beam_search')
    
  Returns:
    Decoded text string
  """
  if method == 'greedy':
    # Greedy decoding
    _, preds = output.max(2)
    preds = preds.squeeze(1)  # Remove batch dimension
    
    # Convert to string using CTC rules
    text = ''
    prev_char = ''
    for pred in preds:
      if pred.item() < len(alphabet):  # Not blank token
        char = alphabet[pred.item()]
        if char != prev_char:  # CTC rule: remove consecutive duplicates
          text += char
        prev_char = char
      else:
        prev_char = ''  # Reset on blank
    
    return text
  else:
    raise NotImplementedError(f"Decoding method '{method}' not implemented") 