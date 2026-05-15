# -------------------------------------------------
# BASE IMAGE (stable + ML friendly)
# -------------------------------------------------
FROM python:3.10-slim

# -------------------------------------------------
# ENV SETTINGS
# -------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# FAISS sometimes needs this
ENV LD_LIBRARY_PATH=/usr/local/lib

# -------------------------------------------------
# SYSTEM DEPENDENCIES
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# WORKDIR
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# COPY REQUIREMENTS FIRST (for caching)
# -------------------------------------------------
COPY requirements.txt .

# -------------------------------------------------
# PIP INSTALL
# -------------------------------------------------
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# COPY PROJECT
# -------------------------------------------------
COPY . .

# -------------------------------------------------
# PORT (FastAPI default)
# -------------------------------------------------
EXPOSE 8000

# -------------------------------------------------
# START COMMAND
# -------------------------------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]