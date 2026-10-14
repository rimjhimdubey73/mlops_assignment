FROM python:3.14.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ROOT=/app/models

WORKDIR /app

COPY requirements-api.txt /app/requirements-api.txt
COPY models/ /app/models/

RUN model_name="$(cat /app/models/current_model.txt)" \
    && test -f "/app/models/${model_name}/requirements.txt" \
    && python -m pip install --no-cache-dir \
        -r /app/requirements-api.txt \
        -r "/app/models/${model_name}/requirements.txt"

COPY app.py /app/app.py
COPY src/ /app/src/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]