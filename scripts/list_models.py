
import os
from google import genai

# engine.py loads Gemini credentials from the Gemini/Google env vars.
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    # If not in env, we might be out of luck unless I hardcode it (don't have it) 
    # or if the user put it in the .env file? No .env file mentioned.
    # But wait, the user just ran the app and got an error. The app sets the env var inside the process.
    # The terminal session might NOT have the env var set if it was set inside the Streamlit session state/process.
    print("API Key not found in environment. Please export GEMINI_API_KEY='...'")
else:
    client = genai.Client(api_key=api_key)
    try:
        print("Listing models...")
        pager = client.models.list()
        for m in pager:
            print(m.name)
    except Exception as e:
        print(f"Error listing models: {e}")
