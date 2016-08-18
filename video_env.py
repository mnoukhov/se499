#TODO(mnuke)
# change pops to something that won't fail on video size 0
# create videoinput for single folder
# figure out zero divison error
# add growing/shrinking box actions
# pre-resize images (or dont)
# change motion to random number of 1 pixel movements instead of 3?

import glob
import logging
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw

from gym import Env
from gym.spaces import Box, HighLow, Discrete


class VideoEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    _action_set = {
        0: (0,0),
        1: (0,3),
        2: (0,-3),
        3: (3,0),
        4: (-3,0),
        5: (3,3),
        6: (-3,-3),
    }

    def __init__(self, videos_folder, frame_shape=(256, 256, 3)):
        # videos folder is directory containing all folders with frames
        # frame shape (width, height, channels)

        self.videos_input = VideosInput(videos_folder, frame_shape)
        self.fig, self.axis = plt.subplots(1)
        plt.ion()
        plt.show()
        self.bbox_rect = patches.Rectangle((0,0), 0, 0,
            linewidth=2,edgecolor='r',facecolor='none')
        self.target_rect = patches.Rectangle((0,0), 0, 0,
            linewidth=1,edgecolor='b',facecolor='none')
        self.axis.add_patch(self.bbox_rect)
        self.axis.add_patch(self.target_rect)

        self.frame_shape = frame_shape

        # move right (-10 to 10) pixels
        # move down (-10 to 10) pixels
        # self.action_space = HighLow(np.matrix([ [-10, 10, 0], [-10, 10, 0] ], dtype=np.int8))
        self.action_space = Discrete(len(self._action_set))

        # image
        self.observation_space = Box(low=0, high=255, shape=frame_shape)

        # intersection / union
        self.reward_range = HighLow(np.matrix([ [0, 1, 2] ], dtype=np.float32))


    def _step(self, action):
        # move_y, move_x = action
        move_x, move_y = self._action_set[action]

        self.bounding_box.move(move_x, move_y, bounds=self.frame_shape)
        reward = self.target.iou(self.bounding_box)
        self.frame, self.target = self.videos_input.get_next_frame()

        return self.frame, reward, self.videos_input.finished, {}

    def _reset(self):
        self.frame = self._get_new_video()
        return self.frame

    def _render(self, mode='human', close=False):
        if close:
            self._close()

        # im = Image.fromarray(self.frame, 'RGB')
        # draw = ImageDraw.Draw(im)
        # draw.rectangle(self.bounding_box.values(), outline=(0, 0, 255))
        # draw.rectangle(self.target.values(), outline=(255, 0, 0))

        # img = cv2.imdecode(self.frame, 1)
        # cv2.rectangle(img, *self.bounding_box.vertices(), color=(0, 0, 255))
        # cv2.rectangle(img, *self.target.vertices(), color=(255, 0, 0))
        # cv2.imshow(self.window_name, img)
        # waitkey(0)
        self.axis.imshow(self.frame)
        self.bbox_rect.set_bounds(
            self.bounding_box.x_min,
            self.bounding_box.y_min,
            int(self.bounding_box.width),
            int(self.bounding_box.height))
        self.target_rect.set_bounds(
            self.target.x_min,
            self.target.y_min,
            int(self.target.width),
            int(self.target.height))
        plt.draw()
        plt.pause(0.1)
        plt.savefig('/home/michael/se499/out/{}_{}.png'.format(
            self.videos_input.cur_video, self.videos_input.cur_frame),
            bbox_inches='tight')

    def _close(self):
        plt.clf()

    def _configure(self, *args, **kwargs):
        pass

    def _seed(self, seed=None):
        pass
        # random.seed(seed)
        # random.shuffle(self.video_folders)

    def _get_new_video(self):
        self.videos_input.get_next_video()
        #TODO(mnuke): check if num frames > 1
        _, self.bounding_box = self.videos_input.get_next_frame()
        self.frame, self.target = self.videos_input.get_next_frame()
        return self.frame


class Rectangle(object):
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    @property
    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def lowerleft(self):
        return (int(self.x_min), int(self.y_min))

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min

    def scale(self, x, y):
        self.x_min = int(self.x_min * x)
        self.x_max = int(self.x_max * x)
        self.y_min = int(self.y_min * y)
        self.y_max = int(self.y_max * y)

    def move(self, x, y, bounds=None):
        if x < 0:
            x = max(-self.x_min, x)
        elif bounds is not None:
            x_bound = bounds[0]
            x = min(x_bound - x, x)

        if y < 0:
            y = max(-self.y_min, y)
        elif bounds is not None:
            y_bound = bounds[1]
            y = min(y_bound - y, y)

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

    def union(self, other):
        return self.area + other.area - self.intersect(other)

    def iou(self, other):
        try:
            return float(self.intersect(other)) / self.union(other)
        except ZeroDivisionError:
            logging.error('Zero division \n self: {} \n other: {}'.format(
                self.vertices(), other.vertices()))
            return 0

    def values(self):
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def vertices(self):
        return [(int(self.x_min), int(self.y_min)),
                (int(self.x_max), int(self.y_max))]


class VideosInput:
    def __init__(self, videos_dir, frame_shape=None):
        self.video_folders = [
            os.path.join(videos_dir, folder) for folder in
            self._get_video_folders(videos_dir)
        ]
        self.num_videos = len(self.video_folders)
        self.frame_shape = frame_shape

        self.cur_video = -1
        self.cur_frame = -1
        self.finished = False

    def get_next_video(self):
        self.cur_video = (self.cur_video + 1)  % self.num_videos
        self.frame_data = self._get_frame_data(
            self.video_folders[self.cur_video])
        self.cur_frame = -1
        self.finished = False

    def get_next_frame(self):
        """Get next frame in series

        if out of frames, return the last frame
        """
        if self.finished:
            frame, bounding_box = self.frame_data[-1]
            return frame, bounding_box
        else:
            self.cur_frame += 1
            try:
                frame, bounding_box = self.frame_data[self.cur_frame]
            except IndexError:
                self.finished = True
                frame, bounding_box = self.frame_data[-1]

        return frame, bounding_box

    def _get_video_folders(self, videos_dir):
        with open('{}/list.txt'.format(videos_dir), 'r') as f:
            return f.read().splitlines()

    def _get_frame_data(self, video_folder):
        video_frames, scale = self._get_video_frames(video_folder)
        bounding_boxes = self._get_bounding_boxes(video_folder, scale)
        return zip(video_frames, bounding_boxes)

    def _get_video_frames(self, video_folder):
        image_files = sorted(glob.glob('{}/*.jpg'.format(video_folder)))
        frames = [misc.imread(filename) for filename in image_files]
        scale = (1,1)
        if self.frame_shape is not None:
            scale = (self.frame_shape[1] / float(frames[0].shape[1]),
                    self.frame_shape[0] / float(frames[0].shape[0]))
            frames = [misc.imresize(frame, self.frame_shape)
                      for frame in frames]

        return frames, scale

    def _get_bounding_boxes(self, video_folder, scale=(1,1)):
        bounding_boxes = []
        with open('{}/groundtruth.txt'.format(video_folder), 'r') as f:
            for coords in f:
                coordinates = [float(c) for c in coords.split(',')]
                x_min, y_min = coordinates[:2]
                x_max, y_max = coordinates[4:6]
                box = Rectangle(x_min, y_min, x_max, y_max)
                box.scale(*scale)
                bounding_boxes.append(box)

        return bounding_boxes

