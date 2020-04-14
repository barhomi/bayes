#!/usr/bin/env python

import numpy as np
import os
import seaborn
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import seaborn

import tensorflow as tf
import tensorflow_probability as tfp
from absl import app
from absl import flags
import logging as log

flags.DEFINE_float("param", None, "comment")
FLAGS = flags.FLAGS

# Function shortcuts
con = tf.constant
tfd = tfp.distributions
tfb = tfp.bijectors

msgs_per_day = con([13,  24,   8,  24,   7,  35,  14,  11,  15,  11,  22,  22,  11,  57,
                    11,  19,  29,   6,  19,  12,  22,  12,  18,  72,  32,   9,   7,  13,
                    19,  23,  27,  20,   6,  17,  13,  10,  14,   6,  16,  15,   7,   2,
                    15,  15,  19,  70,  49,   7,  53,  22,  21,  31,  19,  11,  18,  20,
                    12,  35,  17,  23,  17,   4,   2,  31,  30,  13,  27,   0,  39,  37,
                    5,  14,  13,  22,
                    ], dtype=tf.float32)


def cast(tensor, type_str='f'):
    if type_str == 'f':
        return tf.cast(tensor, dtype=tf.float32)
    elif type_str == 'i':
        return tf.cast(tensor, dtype=tf.int32)
    else:
        raise ValueError

def generate_coin_flip_observations(proba, size):
    prior = tfd.Bernoulli(probs=proba, dtype=tf.int32)
    return prior.sample(size)

def show_distr_pdf(ax, distr, title, resolution=1000):
    xs = tf.linspace(0., 1., resolution)
    pdf = distr.prob(xs)
    ax.plot(xs, pdf)
    ax.set_title(title)
    return ax

def coin_flips_exp():
    sampling_sizes = con([1, 2, 3, 45, 8, 15, 50, 500, 1000, 2000])
    data = generate_coin_flip_observations(proba=0.3, size=sampling_sizes[-1])
    # data = tf.pad(data, paddings=con([[1, 0]]), mode= "CONSTANT")
    # cropping the trials to just the first sampling_sizes to simuluate multiple runs with differnet sizes
    cum = tf.gather(tf.cumsum(data), sampling_sizes)
    beta_distribs = tfd.Beta(concentration1=tf.cast(cum, tf.float32),
                             concentration0=tf.cast(sampling_sizes-cum, tf.float32))
    fig, axes = plt.subplots(1, 1)
    for ind, sampling_size in enumerate(sampling_sizes):
        axes = show_distr_pdf(axes, beta_distribs[ind], title="Sampling={}".format(sampling_size))
    plt.show()


def joint_log_prob(msgs_per_day,lambda_1, lambda_2, tau):
    n_days = tf.size(msgs_per_day)
    alpha = 1. / tf.reduce_mean(msgs_per_day)
    rv_l1 = tfd.Exponential(rate=alpha)
    rv_l2 = tfd.Exponential(rate=alpha)
    rv_tau = tfd.Uniform()

    n_days_b4_tau= tau * cast(n_days, 'f')

    days = cast(tf.range(n_days), 'f')
    ind_days_b4_tau = cast(n_days_b4_tau <= days, 'i')
    lambda_ = tf.gather([lambda_1, lambda_2], indices=ind_days_b4_tau)
    rv_observation = tfd.Poisson(rate=lambda_)

    msgs_log_probas = tf.reduce_sum(rv_observation.log_prob(msgs_per_day))
    joint_proba = rv_l1.log_prob(lambda_1) + rv_l2.log_prob(lambda_2) + rv_tau.log_prob(tau) + msgs_log_probas
    return joint_proba

def unormalized_log_posterior(lambda_1, lambda_2, tau):
    return joint_log_prob(msgs_per_day, lambda_1, lambda_2, tau)

@tf.function(autograph=False)
def graph_sample_chain(*args, **kwargs):
    return tfp.mcmc.sample_chain(*args, **kwargs)

def init_chain(msg_per_day):
    lambda_1 = tf.reduce_mean(msgs_per_day, name='init_lambda1')
    lambda_2 = tf.reduce_mean(msgs_per_day, name='init_lambda2')
    tau = con(0.5, dtype=tf.float32)
    return  [lambda_1, lambda_2, tau]

def plot_samples(samples, title, axe):
    hist, edges = np.histogram(samples, bins=100)
    axe.bar(edges[:-1], hist)
    axe.set_title(title)
    return axe

def main(argv):
    del argv
    # coin_flips_exp()
    n_burnin_steps = 500
    n_results = 200

    initial_chain_state = init_chain(msgs_per_day)
    unconstrained_bijectors = [
        tfp.bijectors.Exp(), # maps a positive real to R
        tfp.bijectors.Exp(), # maps a positive real to R
        tfp.bijectors.Sigmoid() # maps [0,1] to R
    ]
    step_size = 0.2

    inner_kernel= tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unormalized_log_posterior,
                                                 num_leapfrog_steps=2,
                                                 step_size=step_size,
                                                 state_gradients_are_stopped=True)
    kernel = tfp.mcmc.TransformedTransitionKernel(inner_kernel=inner_kernel, bijector=unconstrained_bijectors)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=kernel,
                                               num_adaptation_steps=int(n_burnin_steps*0.8))

    # sampling from the chain:

    samples, kernel_results = graph_sample_chain(num_results=n_results,
                                                 num_burnin_steps=n_burnin_steps,
                                                 current_state=initial_chain_state,
                                                 kernel=kernel
                                                 )

    [lambda_1_samples, lambda_2_samples, posterior_tau] = samples
    n_days = cast(tf.size(msgs_per_day), 'f')
    tau_samples = tf.floor(posterior_tau * n_days)
    samples =[lambda_1_samples, lambda_2_samples, tau_samples]

    fig, axes = plt.subplots(3, 1)
    for rv_samples, ax, rv_name in zip(samples, axes, ['lambda 1', 'lambda 2', 'tau']):
        ax = plot_samples(rv_samples.numpy(), '{} distribution'.format(rv_name), ax)
    plt.show()

if __name__ == '__main__':
    app.run(main)