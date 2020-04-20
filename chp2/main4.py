#!/usr/bin/env python

import numpy as np
import os
import seaborn
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import wget


from absl import app
from absl import flags

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
import tensorflow_probability as tfp
# Function shortcuts
con = tf.constant
tfd = tfp.distributions
tfb = tfp.bijectors



flags.DEFINE_float("param", None, "comment")
FLAGS = flags.FLAGS

def main(argv):
    del argv
    url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv'
    filename = wget.download(url)

if __name__ == '__main__':
    app.run(main)


