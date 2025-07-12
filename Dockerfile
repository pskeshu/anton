FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements_hf.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
CMD ["streamlit", "run", "streamlit_app_hf.py", "--server.port=8501", "--server.address=0.0.0.0"]