# Base image — Python 3.11 slim (lightweight)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caches this layer)
# If requirements don't change, Docker skips reinstalling — faster builds
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/feature_columns.json ./data/processed/feature_columns.json

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
