import os
from dotenv import load_dotenv
import requests

def test_alpha_vantage_connection():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("❌ Error: API Key not found in .env file")
        return
        
    print(f"✅ API Key found in .env file")
    
    # Test with a simple endpoint (TIME_SERIES_INTRADAY)
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey={api_key}'
    
    try:
        print("\n🌐 Testing API connection...")
        response = requests.get(url)
        print(f"📡 Status Code: {response.status_code}")
        print("\n📝 Response Content:")
        print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        
        data = response.json()
        
        if "Note" in data:
            print("\n⚠️ API Rate Limit Message:")
            print(data["Note"])
        elif "Error Message" in data:
            print("\n❌ API Error Message:")
            print(data["Error Message"])
        else:
            print("\n✅ API connection successful!")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    test_alpha_vantage_connection()
