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


# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

# Function shortcuts
con = tf.constant
tfd = tfp.distributions
tfb = tfp.bijectors

flags.DEFINE_float("prob_true", 0.05, "comment")
flags.DEFINE_integer("n", 1500, "comment")
flags.DEFINE_integer("burnin", 25000, "comment")
flags.DEFINE_integer("steps", 48000, "comment")
FLAGS = flags.FLAGS


def generate_data(prob_true=0.05, size=1500):
    rv_x = tfd.Bernoulli(probs=prob_true)
    occ = rv_x.sample(size, seed=10)
    occ_mean = tf.reduce_mean(occ)
    print("Observed frequency in group A: {:.2f}".format(occ_mean.numpy()))
    print("True frequency: {}".format(prob_true))
    return occ

def joint_log_prob(occurences, prob_A):
    rv_prob_A = tfd.Uniform(low=0., high=1.)
    rv_x = tfd.Bernoulli(probs=prob_A)
    joint = rv_prob_A.log_prob(prob_A) + tf.reduce_sum(rv_x.log_prob(occurences))
    return joint

def build_chain(data: tf.Tensor,
                joint_log_prob,
                burnin: int=25000,
                leapfrog_steps: int=2):

    burnin_steps = tf.cast(burnin * 0.8, dtype=tf.int32)

    unconstrained_bijectors = [tfp.bijectors.Identity()]

    unnormalized_posterior_log_prob = lambda *args: joint_log_prob(data, *args)

    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_posterior_log_prob,
                                                  num_leapfrog_steps=leapfrog_steps,
                                                  step_size=0.5,
                                                  # step_size_update_fn=step_size_update_fn,
                                                  state_gradients_are_stopped=True
                                                  )

    kernel = tfp.mcmc.TransformedTransitionKernel(inner_kernel=inner_kernel, bijector=unconstrained_bijectors)
    hmc = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=kernel, num_adaptation_steps=burnin_steps)

    return hmc

def sample_from_posterior(hmc,
                          prior_proba_A: tf.Tensor,
                          burnin: int = 25000,
                          n_steps: int = 48000):
    initial_chain_state = [prior_proba_A]

    [posterior_prob_A], kernel_results = tfp.mcmc.sample_chain(num_results=n_steps,
                                                               num_burnin_steps=burnin,
                                                               current_state=initial_chain_state,
                                                               kernel=hmc
                                                               )

    acc_rate = tf.reduce_mean(tf.cast( kernel_results.inner_results.inner_results.is_accepted, dtype=tf.float32)).numpy()
    print("Acceptance rate: {:.2f}".format(acc_rate))
    return posterior_prob_A[burnin:]

def show_posterior(post_prob_A: np.ndarray):
    print(np.histogram(post_prob_A, bins=100))
    plt.hist(post_prob_A, bins=100, histtype='stepfilled')
    plt.show()

def main(argv):
    del argv
    data = generate_data(prob_true=FLAGS.prob_true, size=FLAGS.n)
    prior_proba_A = tf.reduce_mean(tf.cast(data, dtype=tf.float32))
    hmc = build_chain(data, joint_log_prob, burnin=FLAGS.burnin)
    post_prob_A = sample_from_posterior(hmc, prior_proba_A, burnin=FLAGS.burnin, n_steps=FLAGS.steps)
    show_posterior(post_prob_A.numpy())

if __name__ == '__main__':
    app.run(main)