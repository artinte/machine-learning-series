
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
