FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src:/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY data/reference ./data/reference
COPY data/outputs ./data/outputs
COPY .streamlit ./.streamlit
EXPOSE 8001 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
