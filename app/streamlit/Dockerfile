FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files
COPY app/utils.py ./utils.py
COPY app/streamlit/app.py ./app.py

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]