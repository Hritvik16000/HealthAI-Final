FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision transformers sentence-transformers sacrebleu mlflow shap lime pillow

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
