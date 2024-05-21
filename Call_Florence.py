# Dependencies
# !pip install cognitive-service-vision-model-customization-python-samples
# !pip install Pillow

# Resource and key
import logging
import uuid
logging.getLogger().setLevel(logging.INFO)

import cv2
import numpy as np
from PIL import Image

from cognitive_service_vision_model_customization_python_samples import ResourceType
from cognitive_service_vision_model_customization_python_samples.clients import ProductRecognitionClient
from cognitive_service_vision_model_customization_python_samples.models import ProductRecognition
from cognitive_service_vision_model_customization_python_samples.tools import visualize_recognition_result

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set resource_type based on your needs
resource_type = ResourceType.SINGLE_SERVICE_RESOURCE  # or ResourceType.MULTI_SERVICE_RESOURCE

# Initialize variables for resource name and key
resource_name = None
multi_service_endpoint = None

# Retrieve the appropriate resource based on the ResourceType
if resource_type == ResourceType.SINGLE_SERVICE_RESOURCE:
    resource_name = os.getenv('CV_NAME')
    assert resource_name, "Resource name not found in environment variables."
else:
    multi_service_endpoint = os.getenv('MULTI_SERVICE_ENDPOINT')
    assert multi_service_endpoint, "Multi-service endpoint not found in environment variables."

resource_key = os.getenv('CV_KEY')
assert resource_key, "Resource key not found in environment variables."

# Run Product Recognition with Pre-built Model
client = ProductRecognitionClient(resource_type, resource_name, multi_service_endpoint, resource_key)
run_name = str(uuid.uuid4())
model_name = 'ms-pretrained-product-detection'
run = ProductRecognition(run_name, model_name)

with open('./BeverageShelf.jpg', 'rb') as f:
    img = f.read()

try:
    client.create_run(run, img, 'image/png')
    result = client.wait_for_completion(run_name, model_name)
finally:
    client.delete_run(run_name, model_name)

import json

# Other parts of your code where the client is set up and the image is processed...
# ...

try:
    client.create_run(run, img, 'image/png')
    result = client.wait_for_completion(run_name, model_name)
    # Assuming 'result.result' is the variable that contains the bounding box information

    # Save the bounding box information as a JSON file
    # with open('bounding_boxes.json', 'w') as json_file:
    #     json.dump(result.result, json_file)
    with open('bounding_boxes.json', 'w', encoding='utf-8') as json_file:
        json.dump(result.result, json_file, indent=4, ensure_ascii=False)

finally:
    client.delete_run(run_name, model_name)

# The rest of the visualization code can be removed if you no longer need it

# Convert img data to numpy array and decode numpy array to OpenCV BGR image
cv_img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

# Visualize the result
cv_img = visualize_recognition_result(result.result, cv_img)
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
display(Image.fromarray(cv_img))
