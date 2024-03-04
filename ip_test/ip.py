#!/usr/bin/env python3
import random

import numpy as np
import tensorflow as tf
from itertools import islice

from graphsage import GraphSage

# The aggregation number of layers.
SAMPLE_SIZES = [3, 3]
INTERNAL_DIM = 128

# Training parameters
BATCH_SIZE = 64
TRAINING_STEPS = 1000
LEARNING_RATE = 0.001

if __name__ == '__main__':
    # Initialize graphsage.
    graphsage = GraphSage(dim=len(raw_features[0]), samples=SAMPLE_SIZES, learning_rate=LEARNING_RATE)