FROM whisper-base:latest

WORKDIR /app

# Salin kode Python ke container
COPY youtube_to_shorts.py .
COPY nlp_utils.py .

# Jika Anda punya requirement tambahan (bukan spaCy/whisper)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set ENV
ENV FFMPEG_PATH=ffmpeg \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output

VOLUME /input
VOLUME /output

# Jalankan script utama
CMD ["python", "youtube_to_shorts.py", "--input", "/input", "--output", "/output", "--model-size", "base"]
