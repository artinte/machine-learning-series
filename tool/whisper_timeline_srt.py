import argparse
import whisper
from transformers import MarianMTModel, MarianTokenizer


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds * 1000) % 1000)
    return f'{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}'


def translate_text(text, model_name="Helsinki-NLP/opus-mt-zh-en"):
    # 加载预训练的模型和分词器
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translated = model.generate(**tokenizer(text, return_tensors="pt"))
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe an audio file using Whisper.'
    )
    parser.add_argument(
        'audio_file', type=str, help='Path to the audio file to be transcribed'
    )
    parser.add_argument('output_srt', type=str, help='Path to the output SRT file')
    args = parser.parse_args()

    # tiny/base/small/medium/large/turbo
    model = whisper.load_model('base')
    result = model.transcribe(args.audio_file, word_timestamps=True)
    with open(args.output_srt, 'w', encoding='utf-8') as srt_file:
        for idx, segment in enumerate(result['segments']):
            start = segment['start']  # type: ignore
            end = segment['end']  # type: ignore
            text = segment['text']  # type: ignore
            eng_text = translate_text(text)
            start_time = format_time(start)
            end_time = format_time(end)

            srt_file.write(f'{idx + 1}\n')
            srt_file.write(f'{start_time} --> {end_time}\n')
            srt_file.write(f'{text}\n')
            srt_file.write(f'{eng_text}\n\n')
    print(f'SRT file has been generated: {args.output_srt}')

