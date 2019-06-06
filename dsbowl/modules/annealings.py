import numpy as np


def annealing_no(start, end, pct):
    "No annealing, always return `start`."
    return start


def annealing_linear(start, end, pct):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end-start)


def annealing_exp(start, end, pct):
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end/start) ** pct


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out


def do_annealing_poly(start, end, pct, degree):
    "Helper function for `anneal_poly`."
    return end + (start-end) * (1-pct)**degree
