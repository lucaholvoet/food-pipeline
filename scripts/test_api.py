"""Test the FastAPI endpoint."""
import sys, json, time, threading, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_server():
    import uvicorn
    uvicorn.run('api.app:app', host='127.0.0.1', port=8002, log_level='warning')

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(4)

import requests
print("Testing / ...")
r = requests.get('http://127.0.0.1:8002/')
print(f"  Status: {r.status_code}")
print(f"  Name: {r.json()['name']}")
print(f"  Modules: {r.json()['modules']}")

print("\nTesting /status ...")
r = requests.get('http://127.0.0.1:8002/status')
print(f"  Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))

print("\nTesting /analyze with test_food.jpg ...")
with open('data/test_food.jpg', 'rb') as f:
    r = requests.post('http://127.0.0.1:8002/analyze', files={'image': f})
print(f"  Status: {r.status_code}")
result = r.json()
print(json.dumps(result, indent=2))
