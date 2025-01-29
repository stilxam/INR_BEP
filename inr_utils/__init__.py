""" 
Package with utils for training single INRs / coordinate MLPs / neural fields.

Contents:
losses: module containing loss functions for training INRs
sampling: module containing a Sampler class and subclasses used for sampling locations at which to evaluate INRs during training
images: module containing utilities for working with images, e.g. for turning images into continuous functions through interpolation
    or for extracting images from INRs
metrics: module containing metrics for monitoring the training of INRs
callbacks: module containing custom callbacks for training INRs
training: module containing functions for training INRs
"""
import inr_utils.states as states
import inr_utils.images as images
import inr_utils.nerf_utils as nerf_utils
import inr_utils.sampling as sampling
import inr_utils.losses as losses
import inr_utils.metrics as metrics
import inr_utils.callbacks as callbacks
import inr_utils.training as training
import inr_utils.post_processing as post_processing
import inr_utils.parallel_training as parallel_training
import inr_utils.sdf as sdf