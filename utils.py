import argparse
from typing import Union
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Car tracker application')
    parser.add_argument('--video', type=str, default='video.mkv', help='Path to the video')
    parser.add_argument('--output', type=str, default='out.mp4', help='Path to the output result')
    parser.add_argument('--xml_path', type=str, default='models/FP32/vehicle-detection-adas-0002.xml',
                        help='Path to the xml pretrained model file')
    parser.add_argument('--bin_path', type=str, default='models/FP32/vehicle-detection-adas-0002.bin',
                        help='Path to the bin pretrained model file')

    return parser.parse_args()


def get_video_capture(path: Union[Path, str]):
    cap = cv2.VideoCapture(str(path))
    meta = dict(
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    return cap, meta


def get_video_writer(path: Union[Path, str], meta):
    shape = (meta['width'], meta['height'])
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MP4V'), meta['fps'], shape)
    return out


def draw_track(image, points, color):
    for index, item in enumerate(points):
        if index == len(points) - 1:
            break
        cv2.line(image, item, points[index + 1], color, 2)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color
