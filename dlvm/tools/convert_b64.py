import base64

INPUT_FILE = 'image.jpg'
OUTPUT_FILE = '/tmp/image_b64.json'

"""Open image and convert it to Base64"""
with open(INPUT_FILE, 'rb') as f:
  jpeg_bytes = base64.b64encode(f.read()).decode('utf-8')
  predict_request = '{"image_bytes": {"b64": "%s"}}' % jpeg_bytes
  # Write JSON to file
  with open(OUTPUT_FILE, 'w') as f:
    f.write(predict_request)
