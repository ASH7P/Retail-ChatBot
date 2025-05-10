FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., for SQLite)
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose Gradio app port
EXPOSE 7860

# Run the retail AI app
CMD ["python", "main.py"]
