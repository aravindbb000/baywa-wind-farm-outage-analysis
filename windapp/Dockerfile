FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app/windapp/

# Data directory (mount at runtime)
VOLUME ["/data"]

ENV PYTHONPATH=/app/windapp
ENV FLASK_ENV=production

EXPOSE 5000

WORKDIR /data
CMD ["python", "/app/windapp/app.py"]
