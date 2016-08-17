import glob
import tensorflow as tf

FOLDER = 'ball1'


class Rectangle(object):
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


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


class VisualTracking(object):
    motions = [(-5,0), (5,0), (0,-5), (0,5)]
    bounding_boxes = []
    images = []
    i = 0
    position = None

    def __init__(self):
        with open('{}/groundtruth.txt'.format(FOLDER), 'r') as f:
            for coords in f:
                coordinates = [int(c) for c in coords.split(',')]
                x_min, y_min = coordinates[:2]
                x_max, y_max = coordinates[4:6]
                self.bounding_boxes.append(Rectangle(xmin, y_min, x_max, y_max))

        self.position = bounding_boxes[0]

        image_files = glob.glob('{}/*.jpg'.format(FOLDER))
        self.images = [tf.image.decode_jpeg(filename) for filename in image_files]

    def observe(self):
        return self.images[self.i]

    def collect_reward(self):
        return self.position.intersect(bounding_boxes[self.i])

    def perform_action(self, action):
        self.move(*motions[action])

    def step(self, dt):
        self.i += dt

    def to_html(self, info=[]):
        pass
