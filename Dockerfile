FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Pin huggingface_hub FIRST so pip's resolver doesn't upgrade it
# when installing gradio (gradio 4.44 requires huggingface_hub<0.26)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "huggingface_hub==0.24.7" \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV PYTHONPATH=/app

# Launch the Gradio demo UI (visible on HuggingFace Spaces)
CMD ["python", "app/demo.py"]