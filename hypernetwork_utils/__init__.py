""" 
A package with utilities for training hypernetwork-inr combinations
This is in similar vain to inr_utils.
This package heavily relies on the common_dl_utils from https://github.com/SimonKoop/common_dl_utils and common_jax_utils from https://github.com/SimonKoop/common_jax_utils
These can be installed/updated through `pip install git+https://github.com/SimonKoop/common_jax_utils --upgrade`


submodules:
    metrics: a module containing metrics for monitoring training
    callbacks: a module containing callbacks for training
    training: a module containing training routines

Some terms used in this module that may be confusing:
    datapoint: e.g. an image that is to be recreated as an INR
    target function: this is the function to be approximated by the hypernetwork. That means it takes a datapoint as input,
        and returns a field (i.e. a callable). The INR returned by the hypernetwork is meant to approximate this field.
        A good example of a target function is inr_utils.images.ArrayInterpolator
    Sampler: (from inr_utils.sampling) takes a jax prng key and returns a batch of locations/coordinates at which to evaluate a (neural) field
        this needs to be vmappable and jittable for everything to work


NB the above setup stems from the specific requirements of the research project this repo is based on. You may want to make
some changes, e.g. you may want to get rid of this notion of a "target function".
"""
import hypernetwork_utils.metrics as metrics
import hypernetwork_utils.callbacks as callbacks
import hypernetwork_utils.training as training
import inr_utils  # for convenience
