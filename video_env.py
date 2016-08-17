#TODO(mnuke)
# change pops to something that won't fail on video size 0


import glob
from gym import Env
from gym.spaces import Box, HighLow
import numpy as np
import os
import random
from scipy import misc
from PIL import Image, ImageDraw


class VideoEnv(Env):
    def __init__(self, videos_folder, frame_shape=(128, 128, 3)):
        # videos folder is directory containing all folders with frames
        # frame shape (width, height, channels)

        self.videos_input = VideosInput(videos_folder, frame_shape)

        # move right (-10 to 10) pixels
        # move down (-10 to 10) pixels
        self.action_space = HighLow(np.matrix([ [-10, 10, 0], [-10, 10, 0] ], dtype=np.int8))

        # image
        self.observation_space = Box(low=0, high=255, shape=frame_shape)

        # intersection / union
        self.reward_range = HighLow(np.matrix([ [0, 1, 2] ], dtype=np.float32))

    def _step(self, action):
        move_y, move_up = action
        self.bounding_box.move(move_x, move_y)
        reward = self.target.iou(self.bounding_box)
        self.frame, self.target, finished = self.videos_input.next_frame()

        return frame, reward, finished, {}

    def _reset(self):
        self.frame = self._get_new_video()
        return self.frame

    def _render(self):
        im = Image.fromarray(self.frame, 'RGB')
        draw = ImageDraw.Draw(im)
        draw.rectangle(self.bounding_box.values(), outline=(0, 0, 255))
        draw.rectangle(self.target.values(), outline=(255, 0, 0))
        im.show()
        pass

    def _close(self):
        pass

    def _configure(self, *args, **kwargs):
        pass

    def _seed(self, seed=None):
        random.seed(seed)
        random.shuffle(self.video_folders)

    def _get_new_video(self):
        self.videos_input.next_video()
        _, self.bounding_box = self.videos_input.next_frame()
        frame, self.target = self.videos_input.next_frame()
        return frame


class Rectangle(object):
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def scale(self, x, y):
        self.x_min = int(self.x_min * x)
        self.x_max = int(self.x_max * x)
        self.y_min = int(self.y_min * y)
        self.y_max = int(self.y_max * y)

    def move(self, x, y):
        self.x_min += x
        self.x_max += x
        self.y_min += y
        self.y_max += y

    def intersect(self, other):
        x = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        y = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)

        if x <= 0 or y  <= 0:
            return 0
        else:
            return x*y

    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def union(self, other):
        return self.area + other.area - self.intersect(other)

    def iou(self, other):
        return float(self.intersect(other)) / self.union(other)

    def values(self):
        return [x_min, y_min, x_max, y_max]


class VideosInput:
    def __init__(self, videos_dir, frame_shape):
        self.video_folders = [
            os.path.join(videos_dir,o) for o in os.listdir(videos_dir)
            if os.path.isdir(os.path.join(videos_dir,o))
        ]
        self.num_videos = len(self.video_folders)
        self.current = 0
        self.video_folder = self.video_folders[self.current]
        self.frame_shape = frame_shape
        self.frame_data = self._get_frame_data()

    def next_video(self):
        self.current = (self.current + 1)  % self.num_videos

    def next_frame(self):
        try:
            frame, bounding_box = self.frame_data.pop(0)
        except IndexError:
            return None, None, True
        else:
            return frame, bounding_box, False

    def _get_frame_data(self):
        video_frames = self._get_video_frames()
        bounding_boxes = self._get_bounding_boxes()
        return zip(video_frames, bounding_boxes)

    def _get_video_frames(self):
        image_files = glob.glob('{}/*.jpg'.format(self.video_folder))
        frames = [misc.imread(filename) for filename in image_files]
        # scale = (self.frame_shape[1] / float(frames[0].shape[1]),
                 # self.frame_shape[0] / float(frames[0].shape[0]))
        # resized_frames = [misc.imresize(frame, self.frame_shape)
                          # for frame in frames]

        return frames

    def _get_bounding_boxes(self, scale=(1,1)):
        bounding_boxes = []
        with open('{}/groundtruth.txt'.format(self.video_folder), 'r') as f:
            for coords in f:
                coordinates = [float(c) for c in coords.split(',')]
                x_min, y_min = coordinates[:2]
                x_max, y_max = coordinates[4:6]
                bounding_boxes.append(Rectangle(x_min, y_min, x_max, y_max))

        # return [box.scale(scale[0], scale[1]) for box in bounding_boxes]
        return bounding_boxes


if __name__ == '__main__':
    env = VideoEnv('/home/michael/Documents/se499/vot2016')
