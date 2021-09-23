from subprocess import call
import os
from tqdm import tqdm

# def run(video_path='/home/yangbang/VideoCaptioning/MSRVTT/all_videos', wav_path='/home/yangbang/VC_data/MSRVTT/all_wavs'):
#     os.makedirs(wav_path, exist_ok=True)
#     for fn in os.listdir(video_path):
#         vid = fn.split('.')[0]
#         video_file = os.path.join(video_path, fn)
#         wav_out = os.path.join(wav_path, '%s.wav'%vid)
#         if os.path.exists(wav_out): continue
#         call(["ffmpeg", "-i", video_file, "-f", "wav", wav_out])

def run(video_path='/work3/yangbang/VATEX/all_videos', wav_path='/work3/yangbang/VATEX/all_wavs'):
    os.makedirs(wav_path, exist_ok=True)
    for fn in os.listdir(video_path):
        vid = fn.split('.')[0]
        video_file = os.path.join(video_path, fn)
        wav_out = os.path.join(wav_path, '%s.wav'%vid)
        if os.path.exists(wav_out): continue
        call(["ffmpeg", "-i", video_file, "-f", "wav", wav_out])

if __name__ == "__main__":
    run()
