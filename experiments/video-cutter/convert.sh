#!/usr/bin/env bash
# -crf 23

### Current video in progress: P5

ffmpeg -i right.wmv -c:v libx264 -crf 18 -c:a aac -q:a 100 right.mp4
ffmpeg -i left.wmv -c:v libx264 -crf 18 -c:a aac -q:a 100 left.mp4


ffmpeg -ss 00:01:07  -i game1.mp4 -c copy game1-cut.mp4
ffmpeg -ss 00:11:19  -i game2.mp4 -c copy game2-cut.mp4
ffmpeg -ss 00:02:36  -i game3.mp4 -c copy game3-cut.mp4
ffmpeg -ss 00:01:10  -i game4.mp4 -c copy game4-cut.mp4
ffmpeg -ss 00:00:30  -i game5.mp4 -c copy game5-cut.mp4
ffmpeg -ss 00:02:31  -i game6.mp4 -c copy game6-cut.mp4
ffmpeg -ss 00:03:45  -i game7.mp4 -c copy game7-cut.mp4
ffmpeg -ss 00:02:08  -i game8.mp4 -c copy game8-cut.mp4
ffmpeg -ss 00:06:04  -i game9.mp4 -c copy game9-cut.mp4
ffmpeg -ss 00:02:32  -i game10.mp4 -c copy game10-cut.mp4

## m2ts convert to mp4

ffmpeg -i "1a part-003.m2ts" -vcodec libx264 -crf 18 -acodec ac3 -vf "yadif" game1-p1.mp4
ffmpeg -i "2a part-001.m2ts" -vcodec libx264 -crf 18 -acodec ac3 -vf "yadif" game1-p2.mp4
