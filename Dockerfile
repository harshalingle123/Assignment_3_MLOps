FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY trained_models/ ./trained_models/
ENV PYTHONPATH=/app
CMD ["python", "src/predict.py"]