
1. 提取 pdf 中的所有图片

[get_image_from_pdf](https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/examples/extract-images/extract-from-pages.py)

```
python get_image_from_pdf.py input.pdf
```

2. 压缩图片

```
ffmpeg input.png output.png
```

3. ffmpeg 视频分割工具

```
python ffmpeg_video_split.py video.mp4 00:00:00 00:01:00
```

4. 字幕、视频合并

```
ffmpeg -i video.mp4 -vf "subtitles=subtitle.srt" -c:a copy output.mp4
```

5. 语音转带时间戳的文本

```
whisper_timeline_srt.py
```

6. MP3 裁剪

```
# 剪掉开头
ffmpeg -i input.mp3 -ss 00:00:02 -acodec copy output.mp3
# 剪掉结尾
ffmpeg -i input.mp3 -t 00:03:22 -acodec copy output.mp3
```

7. 文本转语音

```
text_to_tts.py
```

8. 给视频添加字幕

```
add_video_subtitle.py
```

9. Window 安装 FFmepg

```
winget install ffmpeg
```
