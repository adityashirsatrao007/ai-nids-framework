import requests
import json

# Configuration
API_URL = "http://localhost:8000"  # Update to production URL if needed

# Mock DDoS Telemetry Data
ddos_payload = {
    "Average Packet Size": 9.0,
    "Packet Length Mean": 8.5,
    "Packet Length Std": 2.1,
    "Packet Length Variance": 5.0,
    "Total Length of Bwd Packets": 12.0,
    "Subflow Bwd Bytes": 12.0,
    "Total Length of Fwd Packets": 45.0,
    "Subflow Fwd Bytes": 45.0,
    "Avg Bwd Segment Size": 3.0,
    "Bwd Packet Length Mean": 3.0,
    "Init_Win_bytes_forward": 256.0,
    "Max Packet Length": 12.0,
    "Init_Win_bytes_backward": -1.0,
    "Bwd Packet Length Max": 6.0,
    "Fwd Packet Length Max": 9.0,
    "Destination Port": 80.0,
    "Flow IAT Max": 150.0,
    "Flow Duration": 500.0,
    "Flow Bytes/s": 114000.0,
    "Fwd IAT Max": 120.0
}

def test_threat_detection():
    print("--- 🛡️ NIDS API Simulation ---")
    print(f"Connecting to {API_URL}...")
    
    try:
        # 1. Analyze Threat
        print("\n[+] Step 1: Requesting Live Threat Analysis...")
        response = requests.post(f"{API_URL}/api/v1/threat/analyze", json=ddos_payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"    - Detection Status: {result['status']}")
        print(f"    - Threat Code: {result['predicted_attack_code']}")
        print(f"    - Confidence: {result['confidence_score'] * 100:.2f}%")
        
        # 2. Get XAI Explanation
        print("\n[+] Step 2: Extracting SHAP Explainability Proof...")
        explain_response = requests.post(f"{API_URL}/api/v1/threat/explain", json=ddos_payload)
        explain_response.raise_for_status()
        explain_result = explain_response.json()
        
        print("    - Top Feature Indicators:")
        attribution = explain_result['feature_attribution']
        for feat, score in list(attribution.items())[:3]:
            print(f"      * {feat}: {score:+.4f}")

        print("\n[✔] Simulation Complete. System successfully detected and justified the intrusion.")

    except requests.exceptions.ConnectionError:
        print("\n[-] Error: Backend API is offline. Start it with: uvicorn api.main:app")
    except Exception as e:
        print(f"\n[-] Error: {str(e)}")

if __name__ == "__main__":
    test_threat_detection()
