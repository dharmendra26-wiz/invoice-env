FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV PYTHONPATH=/app

# Launch the Gradio demo UI (visible on HuggingFace Spaces)
CMD ["python", "app.py"]