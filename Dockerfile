FROM python:3.10-slim

# 1. Install FFmpeg, git, dan utilitas dasar
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-dev \
    libass9 \
    fonts-freefont-ttf \
    unzip \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Buat direktori kerja
WORKDIR /app

# 3. Copy skrip dan dependencies
COPY youtube_to_shorts.py .
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. (Opsional) Preload model Whisper agar tidak di-download saat runtime
# Uncomment baris berikut jika Anda ingin memaksa model ‘base’ ter-download saat build.
 RUN python -c "import whisper; whisper.load_model('base')"

# 6. Set environment variables yang digunakan skrip
ENV FFMPEG_PATH=ffmpeg \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output

# 7. Siapkan mount point untuk input/output
VOLUME /input
VOLUME /output

# 8. Jalankan skrip
CMD ["python", "youtube_to_shorts.py", "--input", "/input", "--output", "/output", "--model-size", "base"]
