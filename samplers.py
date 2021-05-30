import tensorflow as tf
import numpy as np
import random
import pickle
import cv2

class SequenceSampler:
    def __init__(self, path, height=256, width=256, seq_length=20):
        # load args
        self.height = height
        self.width = width
        self.seq_length = seq_length

        # Read dictionnary
        with open(path, 'rb') as handle:
            self.dataset = pickle.load(handle)
        self.num_samples = len(self.dataset.keys())

    def getDataset(self):
        generator = self._generator
        return tf.data.Dataset.from_generator(generator,
                              args=[],
                              output_types=(tf.float32, tf.float32),
                              output_shapes = (tf.TensorShape([self.seq_length, self.height, self.width, 1]),tf.TensorShape([self.seq_length, 2])))

    def _generator(self):
        # Generator (to act as dataset)
        keys = list(self.dataset.keys())
        random.shuffle(keys)

        for key in keys:
            pose_seq, img_seq, skip = self._generateSequence(self.dataset[key])
            if skip:
                continue
            else:
                img_seq = np.expand_dims(img_seq,-1)
            yield (img_seq, pose_seq)

    def _generateSequence(self, sample):
        skip = False
        positions = []
        images = []
        if sample['length'] < self.seq_length:
            skip = True
        else:
            padding = sample['length'] - self.seq_length
            offset = int(np.random.rand()*padding)
            start = sample['start'] + offset
            end = start + self.seq_length
            for index in range(start, end, 1):
                raw_img = cv2.imread(sample['data'][index]['img_path'],cv2.IMREAD_UNCHANGED)
                raw_img = cv2.resize(raw_img, (self.height, self.width))
                images.append(raw_img)
                positions.append(sample['data'][index]['center'])
        return np.array(positions)/512 - 0.5, np.array(images)/1.0, skip
