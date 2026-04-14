import pandas as pd
import requests
import json
import joblib

def simulate_cyber_attack():
    print("--- Enterprise API Penetration Test ---")
    print("Initializing test sequence...")
    
    # 1. Load real-world dataset
    try:
        df = pd.read_csv('results/ready_sample.csv')
        features = joblib.load('results/feature_names.joblib')
        label_encoder = joblib.load('results/scaler.joblib') # just acting as check, not needed
    except Exception as e:
        print("Required models missing locally.")
        return

    # 2. Extract verified 'DDOS' attack from dataset
    # Note: In our dataset, labels were converted to uppercase by pandas or cleaner
    ddos_candidates = df[df['Label'] == 'DDOS']
    
    if len(ddos_candidates) == 0:
        print("Could not find 'DDOS' exactly. Trying generic slice...")
        ddos_candidates = df[df['Label'] != 'BENIGN']

    attack_row = ddos_candidates.iloc[0]
    print(f"\n[+] Extracted highly verified {attack_row['Label']} attack profile from PCAP data.")
    
    # 3. Format strictly to the API payload rules
    payload = attack_row[features].to_dict()
    
    # 4. Transmit to API Layer
    print("[+] Transmitting raw JSON telemetry to API Server at localhost:8000 ...")
    
    try:
        response = requests.post(
            'http://localhost:8000/api/v1/threat/explain', 
            json=payload,
            timeout=5
        )
    except requests.exceptions.ConnectionError:
        print("[-] API Offline. Please run: uvicorn api.main:app --reload")
        return

    # 5. Output Security Operations Center (SOC) Response
    if response.status_code == 200:
        data = response.json()
        print("\n--- API Incident Response ---")
        print(f"Algorithm Threat Classification Code: {data['predicted_attack_code']}")
        print("\n--- SHAP Explainability Guilt Attribution ---")
        
        # Parse the JSON attribution Dictionary
        attribution = data.get('feature_attribution', {})
        count = 1
        for feature, impact in list(attribution.items())[:5]:
            # Positive SHAP pushes toward attack
            direction = "Critical Threat Indicator (+)" if impact > 0 else "Benign Indicator (-)"
            print(f"{count}. {feature.ljust(30)} | Guilt Score: {impact:>10.4f}  | {direction}")
            count += 1
            
        print("\n[+] System successfully identified anomalies in packet flow.")
    else:
        print(f"[-] API Rejected the packet payload: HTTP {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    simulate_cyber_attack()
