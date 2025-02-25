import argparse
import subprocess


def split_video(input_video, timestamps, output_prefix='part'):
    for i in range(len(timestamps)):
        start_time = timestamps[i]
        end_time = timestamps[i + 1] if i + i < len(timestamps) else None
        output_file = f"{output_prefix}_{i+1:02d}.mp4"

        command = ["ffmpeg", "-i", input_video, "-ss", start_time]
        if end_time:
            command.extend(["-to", end_time])
        command.extend(["-c", "copy", output_file])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a video into multiple parts using ffmpeg.")
    parser.add_argument("input_video", type=str, help="Path to the input video file.")
    parser.add_argument("timestamps", type=str, nargs='+', help="List of timestamps for splitting the video (e.g., '00:00:00 00:01:00 00:07:47 00:14:27').")
    
    args = parser.parse_args()
    split_video(args.input_video, args.timestamps)
