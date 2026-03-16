import requests
from config import LLM_API_KEY, LLM_BASE_URL

def list_models():
    url = f"{LLM_BASE_URL}/models"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models = response.json()
            model_ids = [m['id'] for m in models.get('data', [])]
            print("Available models:")
            for mid in model_ids:
                print(f" - {mid}")
        else:
            print(f"Failed to fetch models. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_models()
