"""Converts image to Base64 for AI Platform ML API."""

import base64

INPUT_FILE = 'image.jpg'
OUTPUT_FILE = 'image_b64.json'

def convert_to_base64(image_file):
  """Open image and convert it to base64"""
  with open(image_file, 'rb') as f:
    jpeg_bytes = base64.b64encode(f.read()).decode('utf-8')
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
    # Write JSON to file
    with open(OUTPUT_FILE, 'w') as f:
      f.write(predict_request)
    return predict_request

convert_to_base64(INPUT_FILE)
