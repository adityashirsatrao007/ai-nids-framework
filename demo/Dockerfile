FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for modeling
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies, prioritizing CPU-only PyTorch to heavily reduce image size
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy everything
COPY . .

# Expose HF Spaces Port
EXPOSE 7860

# Launch Streamlit mapped exactly to Hugging Face's required settings
CMD ["streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
