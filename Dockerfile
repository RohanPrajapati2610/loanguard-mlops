FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir streamlit plotly pandas numpy scikit-learn xgboost

COPY --chown=user src/monitoring/dashboard.py .
COPY --chown=user models/ ./models/
COPY --chown=user data/processed/feature_columns.json ./data/processed/
COPY --chown=user .streamlit/ ./.streamlit/

CMD ["streamlit", "run", "dashboard.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
