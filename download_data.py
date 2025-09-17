import os
import requests

# Make data folder
os.makedirs('data', exist_ok=True)

# URLs to download
urls = {
    'states_cases': 'https://data.incovid19.org/csv/latest/states.csv',
    'vaccination_statewise': 'https://data.incovid19.org/csv/latest/cowin_vaccine_data_statewise.csv',
    'districts_cases': 'https://data.incovid19.org/csv/latest/districts.csv'
}

for name, url in urls.items():
    path = os.path.join('data', f'{name}.csv')
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        print(f"✅ Downloaded {name}.csv")
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")
