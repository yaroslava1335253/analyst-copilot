
import os
import google.generativeai as genai

# We need the API key to be set. 
# Attempt to read from environment or ask user to provide it in the code just for this test if needed.
# Since I can't interactively ask, I will rely on the user having set it in the terminal or app.
# But for now, app.py sets it in os.environ["OPENAI_API_KEY"] (which we are reusing for Gemini in engine.py:77).

# engine.py loads from os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    # If not in env, we might be out of luck unless I hardcode it (don't have it) 
    # or if the user put it in the .env file? No .env file mentioned.
    # But wait, the user just ran the app and got an error. The app sets the env var inside the process.
    # The terminal session might NOT have the env var set if it was set inside the Streamlit session state/process.
    print("API Key not found in environment. Please export GEMINI_API_KEY='...'")
else:
    genai.configure(api_key=api_key)
    try:
        print("Listing models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Error listing models: {e}")
