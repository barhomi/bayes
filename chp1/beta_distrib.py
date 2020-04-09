#!/usr/bin/env python
"""
This code to show what the beta distribution looks like under different parameters
"""

from absl import app
from absl import flags
import logging as log
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

flags.DEFINE_float("param", None, "comment")
FLAGS = flags.FLAGS

def generate_beta_cdf(alpha, beta):
    """
    Args:
        alpha: mean number of successes
        beta: mean number of failures

    """

    dis = tfd.Beta(concentration1=alpha, concentration0=beta)
    xs = tf.linspace(0., 1., 1001)
    pdf = dis.prob(xs)
    plt.plot(xs, pdf)

def main(argv):
    del argv
    generate_beta_cdf(0, 100)
    generate_beta_cdf(5, 6)
    generate_beta_cdf(5, 7)
    generate_beta_cdf(5, 20)
    generate_beta_cdf(5, 50)
    plt.show()

if __name__ == '__main__':
    app.run(main)
