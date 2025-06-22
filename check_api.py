import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print(f"✅ API Key found: {api_key[:20]}...")
else:
    print("❌ API Key NOT FOUND!")
    print("💡 Need to create .env file with GEMINI_API_KEY") 