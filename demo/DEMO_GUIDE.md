# Official NIDS Framework Demo Guide

This guide provides a step-by-step walkthrough for demonstrating the capabilities of the Research-Grade Network Intrusion Detection System.

## 1. Interactive Web Dashboard
The dashboard is the primary way to visualize the AI's decision-making process.

### Step-by-Step Walkthrough:
1.  **Open the App:** Navigate to the [Hugging Face Space](https://huggingface.co/spaces/adityashirsatrao007/AI-NIDS-Research-Framework).
2.  **Verify Metrics:** Click the **📈 Performance Metrics** tab. Highlight the **99.09% Accuracy** and the fact that 15 different attack classes are detected.
3.  **Simulate an Attack:**
    *   Go to **🔍 Live XAI Prediction**.
    *   In the left sidebar, click **🛡️ Load DDoS Attack Profile**.
    *   Notice how the input fields (Average Packet Size, etc.) automatically update.
    *   Click **Predict & Explain**.
4.  **Interpret the Result:**
    *   Observe the "Predicted Class: 2" (DDoS).
    *   Look at the **SHAP Waterfall Chart**. Explain that the **red bars** indicate features that pushed the model to suspect an attack (e.g., high Flow Bytes/sec).

---

### 2. Enterprise API Demo
For industry-side implementation, the system acts as a backend microservice.

#### Quick Test using Python:
We have provided a demo client script: `tests/demo_api_client.py`.
```bash
# Ensure the API is running (uvicorn api.main:app)
python tests/demo_api_client.py
```

#### Raw cURL Command:
You can also hit the endpoint directly from your terminal:
```bash
curl -X POST http://localhost:8000/api/v1/threat/analyze \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

---

### 3. Key Differentiators to Mention
When presenting, emphasize these two technical breakthroughs:
1.  **Attack-Priority Sampling:** Unlike standard models that overfit on "Normal" traffic, this model was trained on a balanced 500k-row subset where rare attacks are mathematically significant.
2.  **Explainable AI (XAI):** Most AI is a "black box." This system provides a mathematical reason for every conviction, making it legally and operationally defensible for real-world Security Operations Centers (SOCs).
