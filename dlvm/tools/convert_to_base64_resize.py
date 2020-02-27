"""Converts image file to JSON array"""

import base64
from PIL import Image

INPUT_FILE = 'image.jpg'
OUTPUT_FILE = 'image_b64.json'


def convert_to_base64_resize(image_file):
  """Open image, resize it, encode it to b64 and save in JSON file"""
  img = Image.open(image_file).resize((240, 240))
  img.save(image_file)  
  with open(image_file, 'rb') as f:
    jpeg_bytes = base64.b64encode(f.read()).decode('utf-8')
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
    # Write JSON to file
    with open(OUTPUT_FILE, 'w') as f:
      f.write(predict_request)
    return predict_request

convert_to_base64_resize(INPUT_FILE)
