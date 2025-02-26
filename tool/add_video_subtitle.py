import argparse
import whisper
import os
import subprocess
from transformers import MarianMTModel, MarianTokenizer


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def translate_text(text, model, tokenizer):
    # 生成翻译
    translated = model.generate(**tokenizer(text, return_tensors="pt"))
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translation


def extract_audio(video_path, audio_output):
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "copy", audio_output]
    subprocess.run(command, check=True)


def add_subtitle(video_path, srt_path, output):
    srt_str = "subtitles=" + srt_path
    command = ["ffmpeg", "-i", video_path, "-vf", f"{srt_str}", "-c:a", "copy", output]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add subtitles to a video.")
    parser.add_argument(
        "video_file", type=str, help="Path to the video file to be transcribed"
    )
    parser.add_argument(
        "output", type=str, default="output.mp4", help="Path to the output file"
    )
    args = parser.parse_args()

    audio_path = "temp.aac"
    srt_path = "temp.srt"
    for file_path in [audio_path, srt_path, args.output]:
        if os.path.exists(file_path):
            os.remove(file_path)
    extract_audio(args.video_file, audio_path)

    # tiny/base/small/medium/large/turbo
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, word_timestamps=True, verbose=True)

    # 加载模型和分词器一次
    if result["language"] == "en":
        model_name = "Helsinki-NLP/opus-mt-en-zh"
    else:
        model_name = "Helsinki-NLP/opus-mt-zh-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    with open(srt_path, "w", encoding="utf-8") as srt_file:
        total = len(result["segments"])
        for idx, segment in enumerate(result["segments"]):
            start = segment["start"]  # type: ignore
            end = segment["end"]  # type: ignore
            text = segment["text"]  # type: ignore

            trans_text = translate_text(text, model, tokenizer)
            start_time = format_time(start)
            end_time = format_time(end)

            srt_file.write(f"{idx + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n")
            srt_file.write(f"{trans_text}\n\n")
            print(f"Processing segment: {idx + 1}/{total}", end="\r")
    print(f"SRT file has been generated: {srt_path}")

    add_subtitle(args.video_file, srt_path, args.output)
