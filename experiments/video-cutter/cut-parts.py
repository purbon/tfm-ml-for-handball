from moviepy.editor import *
import pandas as pd


class VideoFile:

    def __init__(self, io_path):
        self.io_path = io_path
        self.clip = VideoFileClip(self.io_path)
        self.clips = {}

    def clip_video(self, label, start, end):
        self.clips[label] = self.clip.subclip(start, end)
        return label

    def length(self):
        return self.clip.duration

    def add_text(self, label, text):
        clip = self.clips[label]

        text_clip = (TextClip(text, fontsize=24, color='black', stroke_width=1.5, bg_color='white')
                     .set_duration(clip.duration)
                     .set_position((20, clip.h - 44))
                     .set_start(0))

        self.clips[label] = CompositeVideoClip([clip, text_clip])

    def save_clip(self, key, path="./"):
        try:
            self.clips[key].write_videofile(f'{path}/{key}.mp4', codec='libx264')
        except Exception as e:
            print(f"Something bad happened encoding {key}")
            print(e)

    def save_clips(self, path="./"):
        for k, v in self.clips.items():
            self.save_clip(key=k, path=path)


if __name__ == '__main__':
    for game_id in range(10, 11, 20):
        game = f"game{game_id}"
        video_path = f"/Users/purbon/Datasets/tfm/videos/p{game_id}/left-cut.mp4"
        df = pd.read_csv(f"game{game_id}.csv")

        game_start = 0  # (67) 1min and 7 secs, g2 (680): 11min 19sec

        video_file = VideoFile(io_path=video_path)

        for index, row in df.iterrows():
            poss_label = row['possession']
            print(f'Processing {poss_label}')
            start_time = max(row['min'] + game_start - 2, 0)
            end_time = row['max'] + game_start + 2
            if start_time < video_file.length():
                key = f'{game}-{poss_label}'
                video_file.clip_video(label=key, start=start_time, end=end_time)
                video_text = f"Game={game}, poss={poss_label}, start={row['min']}, end={row['max']}"
                video_file.add_text(label=key, text=video_text)
                video_file.save_clip(key=key, path=f"./{game}-left")
