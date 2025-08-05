import os
import re
import tempfile
import shutil
import subprocess
import argparse
import logging
import whisper
import ffmpeg

from nlp_utils import analyze_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

def get_video_duration(path: str) -> float:
    cmd = [FFMPEG_PATH, "-i", path, "-f", "null", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    text = stderr.decode(errors="ignore")
    m = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", text)
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)

def trim_video(src: str, start: float, end: float, dst: str) -> bool:
    cmd = [FFMPEG_PATH, "-ss", str(start), "-to", str(end), "-i", src, "-c", "copy", "-y", dst]
    return subprocess.run(cmd, capture_output=True).returncode == 0

def resize_video(src: str, dst: str, aspect_w: int, aspect_h: int, target_w: int) -> bool:
    crop_expr = f"ih*{aspect_w}/{aspect_h}"
    filter_crop = f"crop={crop_expr}:ih:(iw-{crop_expr})/2:0"
    target_h = int(target_w * aspect_h / aspect_w)
    filter_scale = f"scale={target_w}:{target_h}"
    vf = f"{filter_crop},{filter_scale}"

    cmd = [FFMPEG_PATH, "-i", src,
           "-vf", vf,
           "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           "-c:a", "aac", "-b:a", "128k",
           "-y", dst]
    return subprocess.run(cmd, capture_output=True).returncode == 0

def ass_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int((sec - int(sec)) * 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def extract_audio(video: str, wav_out: str) -> bool:
    ffmpeg.input(video).output(wav_out, ac=1, ar=16000, format="wav", loglevel="error") \
        .overwrite_output().run()
    return os.path.isfile(wav_out)

def generate_ass_karaoke_whisper(video_path: str, ass_path: str, n_words: int, model_size: str):
    wav = video_path.rsplit(".", 1)[0] + ".wav"
    if not extract_audio(video_path, wav):
        logging.error("Gagal ekstrak audio")
        return None

    model = whisper.load_model(model_size)
    result = model.transcribe(wav, language="id", word_timestamps=True)
    os.remove(wav)

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "word":  w["word"].strip(),
                "start": w["start"],
                "end":   w["end"],
            })

    if not words:
        logging.info("Tidak ada kata terdeteksi oleh Whisper.")
        return None

    chunks = [words[i:i + n_words] for i in range(0, len(words), n_words)]

    ass_header = """[Script Info]
Title: Karaoke Highlight (Whisper)
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke, Arial, 62, &H00FFFFFF, &H0000FFFF, &H00000000, &H64000000,1,0,0,0,100,100,0,0,1,4,3,5,40,40,1060,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    ass_events = ""
    x_center = 1080 // 2
    y_offset = (1920 // 2) + 500
    for grp in chunks:
        start_ass = ass_time(grp[0]["start"])
        end_ass = ass_time(grp[-1]["end"])
        line = ""
        for w in grp:
            dur_cs = max(1, int((w["end"] - w["start"]) * 100))
            line += f"{{\\k{dur_cs}}}{w['word']} "
        ass_events += f"Dialogue: 0,{start_ass},{end_ass},Karaoke,,0,0,0,,{{\\pos({x_center},{y_offset})}}{line.strip()}\n"

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + ass_events)

    return words

def add_subtitles_to_video(video: str, ass: str, out: str) -> bool:
    vf = f"subtitles={os.path.abspath(ass)}"
    cmd = [FFMPEG_PATH, "-i", video, "-vf", vf, "-c:a", "copy", "-y", out]
    return subprocess.run(cmd, capture_output=True).returncode == 0

def save_summary_to_txt_per_clip(output_video_path, keywords, summary):
    base_name = os.path.splitext(os.path.basename(output_video_path))[0]
    output_txt_path = os.path.join(os.path.dirname(output_video_path), f"{base_name}.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Judul Rekomendasi: {summary}\n")
        f.write(f"Tags: {keywords}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--clip-duration", type=int, default=59)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--n-words", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    videos = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    ]

    for vid in videos:
        base = os.path.splitext(os.path.basename(vid))[0]
        duration = get_video_duration(vid)
        usable = duration - args.start_offset
        if usable < args.clip_duration:
            logging.info(f"Skipping karena terlalu pendek: {vid}")
            continue

        clips = int(usable // args.clip_duration)
        for i in range(clips):
            s = args.start_offset + i * args.clip_duration
            e = s + args.clip_duration
            tmpdir = tempfile.mkdtemp()

            try:
                clip_raw = os.path.join(tmpdir, f"clip_{i}.mp4")
                clip_resize = os.path.join(tmpdir, f"clip_{i}_r.mp4")
                clip_ass = os.path.join(tmpdir, f"clip_{i}.ass")
                clip_subt = os.path.join(tmpdir, f"clip_{i}_s.mp4")
                final_output = os.path.join(args.output, f"{base}_short_{i+1}.mp4")

                if not trim_video(vid, s, e, clip_raw):
                    continue
                if not resize_video(clip_raw, clip_resize, 9, 16, 1080):
                    continue

                words = generate_ass_karaoke_whisper(
                    clip_resize, clip_ass,
                    n_words=args.n_words,
                    model_size=args.model_size
                )

                if words:
                    text_transcript = " ".join([w["word"] for w in words])
                    result = analyze_text(text_transcript)

                    title = result["title"]
                    keywords = result["keywords"]

                    logging.info(f"📌 NLP - Keywords: {keywords}")
                    logging.info(f"📝 NLP - Judul: {title}")

                    if add_subtitles_to_video(clip_resize, clip_ass, clip_subt):
                        shutil.copy(clip_subt, final_output)
                    else:
                        shutil.copy(clip_resize, final_output)

                    # Tambahkan ekspor ke TXT
                    txt_path = os.path.join(args.output, f"{base}_short_{i + 1}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"Judul Rekomendasi: {title}\n")
                        f.write(f"Tags: {', '.join(keywords)}\n")

                logging.info(f"✅ Saved: {final_output}")

            finally:
                shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()
