FROM python:3.12-slim

# System dependencies (ffmpeg + OpenCV + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 agentuser
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=agentuser:agentuser . .

# Create runtime directories
RUN mkdir -p /app/output /app/models /app/data && \
    chown -R agentuser:agentuser /app

USER agentuser

EXPOSE 7821

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7821/health || exit 1

ENTRYPOINT ["python", "-m", "agent.main"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7821"]
