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

flags.DEFINE_float("prob_a", 0.05, "comment")
flags.DEFINE_float("prob_b", 0.04, "comment")
flags.DEFINE_integer("n_a", 1500, "comment")
flags.DEFINE_integer("n_b", 750, "comment")
flags.DEFINE_integer("burnin", 25000, "comment")
flags.DEFINE_integer("steps", 48000, "comment")
FLAGS = flags.FLAGS


def generate_data(prob_a=0.05,prob_b=0.04, size_a=1500, size_b=750):
    rv_a = tfd.Bernoulli(probs=prob_a)
    rv_b = tfd.Bernoulli(probs=prob_b)

    occ_a = rv_a.sample(size_a, seed=10)
    occ_b = rv_b.sample(size_b, seed=10)

    return occ_a, occ_b


def joint_log_prob(occurences_A, occurences_B, prob_A, prob_B):
    rv_prob_A = tfd.Uniform(low=0., high=1.)
    rv_prob_B = tfd.Uniform(low=0., high=1.)

    rv_a = tfd.Bernoulli(probs=prob_A)
    rv_b = tfd.Bernoulli(probs=prob_B)

    joint = rv_prob_A.log_prob(prob_A) + tf.reduce_sum(rv_a.log_prob(occurences_A))
    joint += rv_prob_B.log_prob(prob_B) + tf.reduce_sum(rv_b.log_prob(occurences_B))

    return joint

def build_chain(occ_A: tf.Tensor,
                occ_B: tf.Tensor,
                joint_log_prob,
                burnin: int=25000,
                leapfrog_steps: int=2):

    burnin_steps = tf.cast(burnin * 0.8, dtype=tf.int32)

    unconstrained_bijectors = [ tfp.bijectors.Identity(),
                                tfp.bijectors.Identity()
                                ]

    unnormalized_posterior_log_prob = lambda *args: joint_log_prob(occ_A, occ_B, *args)

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
                          prior_proba_B: tf.Tensor,
                          burnin: int = 25000,
                          n_steps: int = 48000):
    initial_chain_state = [prior_proba_A, prior_proba_B]

    [posterior_prob_A, posterior_prob_B], kernel_results = tfp.mcmc.sample_chain(num_results=n_steps,
                                                                                 num_burnin_steps=burnin,
                                                                                 current_state=initial_chain_state,
                                                                                 kernel=hmc
                                                                                 )

    acc_rate = tf.reduce_mean(tf.cast( kernel_results.inner_results.inner_results.is_accepted, dtype=tf.float32)).numpy()
    print("Acceptance rate: {:.2f}".format(acc_rate))
    return posterior_prob_A[burnin:], posterior_prob_B[burnin:]

def show_posteriors(post_prob_A: np.ndarray, post_prob_B: np.ndarray):
    print(np.histogram(post_prob_A, bins=100))
    fig, axes = plt.subplots(2, 1)
    axes[0].hist(post_prob_A, bins=100, histtype='stepfilled')
    axes[1].hist(post_prob_B, bins=100, histtype='stepfilled')
    plt.show()

def main(argv):
    del argv
    occ_a, occ_b = generate_data(prob_a=FLAGS.prob_a, prob_b=FLAGS.prob_b, size_a=FLAGS.n_a, size_b = FLAGS.n_b)
    prior_proba_A = tf.reduce_mean(tf.cast(occ_a, dtype=tf.float32))
    prior_proba_B = tf.reduce_mean(tf.cast(occ_b, dtype=tf.float32))

    hmc = build_chain(occ_a, occ_b, joint_log_prob, burnin=FLAGS.burnin)
    post_prob_A, post_prob_B= sample_from_posterior(hmc, prior_proba_A, prior_proba_B, burnin=FLAGS.burnin, n_steps=FLAGS.steps)
    show_posteriors(post_prob_A.numpy(), post_prob_B.numpy())

if __name__ == '__main__':
    app.run(main)
