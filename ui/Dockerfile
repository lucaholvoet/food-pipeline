FROM python:3.11-slim

WORKDIR /app

# fonts-dejavu-core provides DejaVuSans-Bold.ttf for PIL bbox label rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        fontconfig \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY db.py .
COPY profile.py .
COPY auth.py .
COPY app.py .

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
