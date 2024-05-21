import requests
import time
import json

def load_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)  # onlt split at the first '='
            config[key] = value  # save the key-value pair to the dictionary
    return config

# log configurations
config = load_config('config.txt')
api_key = config['api_key']
api_url = config['api_base']

headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

# request body
data = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "what is in this image:"},
            {"type": "image_url", "image_url": {"url": "https://putyoururlhere"}}  
        ]}
    ],
    "max_tokens": 100,
    "stream": False
}

# send the request and time it
start_time = time.time()
response = requests.post(api_url, headers=headers, data=json.dumps(data))
end_time = time.time()

print(f"Response received in {end_time - start_time:.2f} seconds\n")
print(response.json())
