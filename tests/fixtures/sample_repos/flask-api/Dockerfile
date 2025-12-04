FROM python:3.11-slim

WORKDIR /app

# Environment variables required by the application
ENV DATABASE_URL
ENV REDIS_URL
ENV SECRET_KEY
ENV LOG_LEVEL=info
ENV APP_PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
