"""Test the FastAPI endpoint."""
import sys, json, time, threading, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_server():
    import uvicorn
    uvicorn.run('api.main:app', host='127.0.0.1', port=8002, log_level='warning')

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(4)

import requests
print("Testing /health ...")
r = requests.get('http://127.0.0.1:8002/health')
print(f"  Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))

print("\nTesting /analyze with a sample image ...")
test_image = os.getenv("TEST_IMAGE_PATH", "output/gradcam_pizza.png")
with open(test_image, 'rb') as f:
    r = requests.post('http://127.0.0.1:8002/analyze', files={'file': f})
print(f"  Status: {r.status_code}")
result = r.json()
print(json.dumps(result, indent=2))
