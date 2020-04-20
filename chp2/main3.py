#!/usr/bin/env python

import numpy as np
import os
import seaborn
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


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

flags.DEFINE_integer("burnin", 15000, "comment")
flags.DEFINE_integer("steps", 40000, "comment")
flags.DEFINE_integer("total_count", 100, "comment")
flags.DEFINE_integer("total_yes", 35, "comment")
FLAGS = flags.FLAGS

def ca(input: tf.Tensor, dtype: tf.DType):
    return tf.cast(input, dtype=dtype)

def co(input, dtype: tf.DType):
    return tf.constant(input, dtype=dtype)

def coin_joint_log_proba(total_yes, total_count, lies_prob):
    rv_lies_prob = tfd.Uniform(low=0, high=1)
    cheated = tfd.Bernoulli(probs= ca(lies_prob, tf.float32)).sample(total_count)
    first_flip = tfd.Bernoulli(probs=0.5).sample(total_count)
    second_flip = tfd.Bernoulli(probs=0.5).sample(total_count)
    sampled_yeses = tf.reduce_sum(cheated * first_flip  + (1-first_flip)*second_flip)
    yes_prob = ca(sampled_yeses, tf.float32) / ca(total_count, tf.float32)
    rv_yeses = tfd.Binomial(total_count=ca(total_count, tf.float32), probs=yes_prob)
    return tf.reduce_sum(rv_yeses.log_prob(total_yes)) + rv_lies_prob.log_prob(lies_prob)

def mcmc(burnin: int, n_steps: int, total_count: int, total_yes: int):

    initial_chain_state = [co(0.4, tf.float32)]

    unnormalized_posterior_log_proba = lambda *args: coin_joint_log_proba(total_yes, total_count, *args)

    metropolis = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=unnormalized_posterior_log_proba,
                                               seed=54)

    # sample from the chani:
    [posterior_p], kernel_results =  tfp.mcmc.sample_chain(num_results=n_steps,
                                                           num_burnin_steps=burnin,
                                                           current_state=initial_chain_state,
                                                           kernel=metropolis,
                                                           parallel_iterations=1,
                                                           )
    return posterior_p

def main(argv):
    del argv
    posterior_p = mcmc(burnin=FLAGS.burnin,
                       n_steps=FLAGS.steps,
                       total_count=FLAGS.total_count,
                       total_yes=FLAGS.total_yes,
                       )

    plt.hist(posterior_p, bins=100, histtype='stepfilled')
    plt.show()


if __name__ == '__main__':
    app.run(main)