""" 
Module with training utilities for hypernetworks
The most important class is Trainer, which is a class that can be used to train a hypernetwork
Additionally, the ValidationLoop class can be used to perform validation during training
    It is implemented as a metric, so it can be used in the MetricCollector
"""
from typing import Callable, Union, Any, Optional

import jax
import equinox as eqx
import optax
import numpy as np
from torch.utils.data import DataLoader

from inr_utils.sampling import Sampler
from hypernetwork_utils import metrics
from hypernetwork_utils import callbacks
from common_jax_utils.types import register_type
from common_jax_utils import key_generator


register_type(DataLoader)

def make_loss_evaluator(
        loss_function: Callable, 
        target_function: Callable,
        )->Callable:
    """ 
    Make a function that evaluates the loss on a single datapoint
    :parameter loss_function: a function that takes two arrays and returns a scalar
    :parameter target_function: a function that takes a datapoint and returns the corresponding field that the inr should try to match
        it should be a mapping *data_point -> (locations -> field_value)
    :returns: a function, evaluate_loss_on_datapoint, that takes a hypernetwork, a datapoint, and locations, and returns the loss

    from the docstring of evaluate_loss_on_datapoint:
        :parameter hyper_network: the hypernetwork
        :parameter data_point: the original datapoint
        :parameter locations: the locations/coordinates at which to evaluate the field and INR
        :returns: the loss
    """
    def evaluate_loss_on_datapoint(
            hyper_network: eqx.Module,
            data_point: Union[jax.Array, tuple[jax.Array,...]],  # or any PyTree
            locations: jax.Array
            ):
        """ 
        Evaluate the loss on a single datapoint
        :parameter hyper_network: the hypernetwork
        :parameter data_point: the original datapoint
        :parameter locations: the locations at which to evaluate the field and INR
        :returns: the loss
        """
        inr = hyper_network(data_point)
        ground_truth_function = target_function(data_point)
        pred = jax.vmap(inr)(locations)
        ground_truth = jax.vmap(ground_truth_function)(locations)
        return loss_function(pred, ground_truth)
    return evaluate_loss_on_datapoint

def make_batched_loss_evaluator(loss_evaluator):
    """ 
    Take a loss evaluator (i.e. an evaluate_loss_on_datapoint function as procuced by make_loss_evaluator)
    and return a function that evaluates the loss on a batch of datapoints and takes the mean of the resulting losses
    """
    def evaluate_loss_on_batch(
            hyper_network, 
            batch,
            locations
            ):
        """ 
        Evaluate the loss on a batch of datapoints
        :parameter hyper_network: the hypernetwork
        :parameter data_point: the original batch of datapoints
        :parameter locations: the locations at which to evaluate the fields and INRs
            NB the first dimension, 0, of this should be the batch dimension corresonding to the batch of datapoints,
            the second dimension, 1, should be the batch dimension corresponding to the batch of locations
            so this should have shape (batch_size, num_locations, num_dimensions)
        :returns: the loss (a scalar array)
        """
        return jax.vmap(loss_evaluator, in_axes=(None, 0, 0))(
                    hyper_network,
                    batch,
                    locations
                    ).mean()
    return evaluate_loss_on_batch

def make_train_step_function(
        loss_function: Callable,
        target_function: Callable, # takes datapoint and returns the corresponding field that the inr should try to match
        location_sampler: Sampler,
        optimizer: optax.GradientTransformation,
        sub_steps_per_datapoint: int
        )->Callable:
    """ 
    A function that creates a training step function

    :parameter loss_function: a function that takes two arrays and returns a scalar
    :parameter target_function: a function that takes a datapoint and returns the corresponding field that the inr should try to match
        it should be a mapping *data_point -> (locations -> field_value)
    :parameter location_sampler: an inr_utils.sampling.Sampler that samples locations
        this should take a jax prng key and return a batch of locations
        this should be vmappable
    :parameter optimizer: an optax optimizer
    :parameter sub_steps_per_datapoint: the number of gradient steps to take per datapoint
        for each step, new locations will be sampled
    :returns: a train_step function
    """
    evaluate_loss_on_datapoint = make_loss_evaluator(
        loss_function=loss_function, 
        target_function=target_function
    )
    evaluate_loss_on_batch = make_batched_loss_evaluator(evaluate_loss_on_datapoint)
    
    @eqx.filter_jit
    def train_step(
            hyper_network:eqx.Module,
            data: Union[jax.Array, tuple[jax.Array,...]],
            optimizer_state: optax.OptState,
            key:jax.Array
            ):
        """ 
        Perform a training step (i.e. a number of gradient steps on a batch of data)
        :parameter hyper_network: the hypernetwork
        :parameter data: the batch of data
        :parameter optimizer_state: the optimizer state
        :parameter key: the jax prng key
        :returns: the updated hypernetwork, updated optimizer state, and the loss
        """
        if isinstance(data, jax.Array):
            batch_size = data.shape[0]
        else:
            batch_size = data[0].shape[0]

        key_1 = key
        loss_total = 0.
        for _ in range(sub_steps_per_datapoint):  # in hindsight, it would probably have been better to do this using jax.lax.scan instead of a Python for-loop
            key_1, key_2 = jax.random.split(key_1, 2)
            keys = jax.random.split(key_2, batch_size)
            locations = jax.vmap(location_sampler)(keys)
            loss, grad = eqx.filter_value_and_grad(evaluate_loss_on_batch)(
                hyper_network,
                data,
                locations
            )
            loss_total += loss

            updates, optimizer_state = optimizer.update(grad, optimizer_state, hyper_network)
            hyper_network = eqx.apply_updates(hyper_network, updates)
        loss_total /= sub_steps_per_datapoint
        return hyper_network, optimizer_state, loss_total
    return train_step


# bunch of dummy classes for type annotations
@register_type
class Schedule:
    pass


# collection of code for creating more elaborate optimizers
class OptimizerFactory:
    """ 
    This is a collection of functions for creating optimizers
    Because much of optax works with factory functions, these don't always
    play nice with the config_realization tools from common_dl_utils when it comes to type annotations
    (e.g. the type annotation for mask in adamw is Optional[Union[Any, Callable[[optax.Params], Any]]])
    """
    @staticmethod
    def single_optimizer(
            optimizer_type: type[optax.GradientTransformation],
            optimizer_config: dict,
            optimizer_mask:Optional[Callable]=None,
            learning_rate_schedule:Union[None, Schedule, list[Schedule], tuple[Schedule, ...]]=None,
            schedule_boundaries:Optional[list[int]]=None,
            )->optax.GradientTransformation:
        """single_optimizer
        Create a single optimizer (without synced parameters) optionally with a learning rate schedule and a mask function

        :param optimizer_type: factory for an optax.GradientTransformation (e.g. optax.adam)
        :param optimizer_config: configuration for the optimizer
        :param optimizer_mask: optional callable for masking parameters, defaults to None
            if this is not None, it is added to (a copy of) optimizer_config under the key 'mask'
        :param learning_rate_schedule: optional learning rate schedule, defaults to None
            if this is not None, it is added to (a copy of) optimizer_config under the key 'learning_rate'
            if this is a list or tuple, it is joined using optax.join_schedules with schedule_boundaries for boundaries
        :param schedule_boundaries: boundaries for optax.join_schedules
            only used if learning_rate_schedule is a list or tuple
        :return: an optax optimizer
        """
        optimizer_config = dict(optimizer_config)
        if learning_rate_schedule is not None:
            if isinstance(learning_rate_schedule, (list, tuple)):
                learning_rate_schedule = optax.join_schedules(learning_rate_schedule, schedule_boundaries)
            optimizer_config['learning_rate'] = learning_rate_schedule
        if optimizer_mask is not None:
            optimizer_config['mask'] = optimizer_mask
        return optimizer_type(**optimizer_config)


class ValidationLoop(metrics.Metric, eqx.Module):  
    """ 
    Validation loop as a common_dl_utils.metrics.Metric
    """
    # this way the validation loop is optional 
    # as it may slow training down quite significantly
    # and other, cheaper, metrics might already give enough insight into generalization
    required_kwargs = set({'hypernetwork', 'validation_loader', 'key'})

    loss_evaluator_single: Callable
    loss_evaluator_batch: Callable
    location_sampler: Sampler

    frequency: metrics.MetricFrequency

    def __init__(
            self, 
            target:eqx.Module,
            loss_function:Callable,
            location_sampler:Sampler,
            frequency:str = 'every_epoch'
            ):
        """ 
        :parameter target: the target function that the hypernetwork should approximate 
            this should be a mapping *data_point -> (locations -> field_value)
        :parameter loss_function: a function that takes two arrays and returns a scalar
        :parameter location_sampler: an inr_utils.sampling.Sampler that samples locations
            this should take a jax prng key and return a batch of locations
            this should be vmappable
        :parameter frequency: the frequency at which to evaluate the validation loop
            default is 'every_epoch'
            should be one of:
                'every_batch'
                'every_n_batches'
                'every_epoch'
                'every_n_epochs'

        The type hints here should be taken with a grain of salt
        """
        self.loss_evaluator_single = make_loss_evaluator(
            loss_function=loss_function,
            target_function=target,
        )
        #self.loss_evaluator_batch = make_batched_loss_evaluator(self.loss_evaluator_single)
        self.loss_evaluator_batch = jax.vmap(self.loss_evaluator_single, in_axes=(None, 0, 0))
        # ^ for computing the std of the loss, we need all loss values in stead of the mean over the batch
        self.location_sampler = location_sampler
        self.frequency = metrics.MetricFrequency(frequency)

    @eqx.filter_jit
    def compute_loss(self, hypernetwork:eqx.Module, batch:Union[jax.Array, tuple[jax.Array, ...]], key: jax.Array)->jax.Array:
        """ 
        Compute the loss on each element of a batch
        :parameter hypernetwork: the hypernetwork that is to be evaluated
        :parameter batch: a batch of datapoints on which to evaluate
        :parameter key: a prng key
        :return: the loss for each element of the batch (so with shape (batch_size, ))
        """
        if isinstance(batch, jax.Array):
            batch_size = batch.shape[0]
        else: 
            batch_size = batch[0].shape[0]
        keys = jax.random.split(key=key, num=batch_size)
        locations = jax.vmap(self.location_sampler)(keys)

        return self.loss_evaluator_batch(hypernetwork, batch, locations)
    
    def compute(self, **kwargs):
        """ 
        Implementation of the common_dl_utils.metrics.Metric.compute abstract method
        :parameter **kwargs: a dictionary containing *at least* the keys 'hypernetwork', 'validation_loader', and 'key'
           hypernetwork: eqx.Module which is the hypernetwork that is to be evaluated
           validation_loader: torch.utils.data.DataLoader with all validation data
           key: prng key
        """
        # get the relevant info from kwargs
        hypernetwork:eqx.Module = kwargs['hypernetwork']
        validation_loader:DataLoader = kwargs['validation_loader']
        key:jax.Array = kwargs['key']
        
        num_batches = len(validation_loader)
        batch_size = validation_loader.batch_size
        keys = jax.random.split(key=key, num=num_batches)

        losses = np.empty(shape=(num_batches, batch_size), dtype=np.float32)

        print("    Start Validation Loop")
        for index, (batch, key) in enumerate(zip(validation_loader, keys)):
            losses[index] = self.compute_loss(hypernetwork, batch, key)
        
        # prepare an output for reporting to e.g. wandb
        results = {
            'validation/loss': np.mean(losses),
            'validation/loss-std': np.std(losses)
            }
        # also put this on the stdout
        print(f"    validation loss: {results['validation/loss']} +/- {results['validation/loss-std']}")
        return results


@register_type
class Trainer:
    """ 
    Trainer class for training hypernetworks
    """
    loss_function: Callable
    target: Callable
    train_step: Callable
    location_sampler: Sampler

    optimizer: optax.GradientTransformation
    optimizer_state: optax.OptState

    hypernetwork: eqx.Module

    metric_collector: metrics.MetricCollector

    after_step_callback: Union[None, Callable, callbacks.Callback]
    after_epoch_callback: Union[None, Callable, callbacks.Callback]

    compute_loss_on_datapoint: Callable

    use_wandb: bool
    display_func: Union[None, Callable]
    epochs: int
    sub_steps_per_datapoint: int
    train_loader: DataLoader
    validation_loader: DataLoader
    example_batch: Union[jax.Array, tuple[jax.Array, ...]]
    _step: int

    def __init__(
            self, 
            hypernetwork: eqx.Module,
            train_loader: DataLoader,
            validation_loader: DataLoader,
            loss_function: Callable,
            target: eqx.Module,
            location_sampler:Sampler,
            optimizer:optax.GradientTransformation,
            sub_steps_per_datapoint:int,
            epochs: int,
            metric_collector: metrics.MetricCollector,
            after_step_callback: Union[None, Callable, callbacks.Callback],
            after_epoch_callback: Union[None, Callable, callbacks.Callback],
            optimizer_state: Union[None, optax.OptState]=None,
            use_wandb: bool=True,
            ):
        """ 
        :parameter hypernetwork: the hypernetwork to be trained
        :parameter train_loader: the training data loader
        :parameter validation_loader: the validation data loader
        :parameter loss_function: a function that takes two arrays and returns a scalar
        :parameter target: the target function that the hypernetwork should approximate
            this should be a mapping *data_point -> (locations -> field_value)
        :parameter location_sampler: an inr_utils.sampling.Sampler that samples locations
            this should take a jax prng key and return a batch of locations
            this should be vmappable
        :parameter optimizer: an optax optimizer
        :parameter sub_steps_per_datapoint: the number of gradient steps to take per datapoint
            for each step, new locations will be sampled
        :parameter epochs: the number of epochs to train for
        :parameter metric_collector: a MetricCollector for monitoring training
        :parameter after_step_callback: an optional callback to be called after each step
            This can be used for logging, checkpointing, quitting early, etc.
        :parameter after_epoch_callback: an optional callback to be called after each epoch
        :parameter optimizer_state: optional optimizer state to start from
            if None is provided, the optimizer state is initialized as 
                optimizer.init(eqx.filter(hypernetwork, eqx.is_array))
        :parameter use_wandb: whether to use wandb for logging

        If a validation loop is desired, it can be added to the metric collector
        (e.g. using the ValidationLoop class)
        """
        self.hypernetwork = hypernetwork
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.example_batch = next(iter(validation_loader))

        self.train_step = make_train_step_function(
            loss_function=loss_function,
            target_function=target,
            optimizer=optimizer,
            location_sampler=location_sampler,
            sub_steps_per_datapoint=sub_steps_per_datapoint
        )
        self.compute_loss_on_datapoint = make_loss_evaluator(
            loss_function=loss_function,
            target_function=target
        )

        self.loss_function = loss_function
        self.target = target
        self.location_sampler = location_sampler
        self.optimizer = optimizer
        self.optimizer_state = optimizer_state if optimizer_state is not None else self.optimizer.init(
            eqx.filter(hypernetwork, eqx.is_array)
        )

        self.metric_collector = metric_collector
        self.after_step_callback = after_step_callback
        self.after_epoch_callback = after_epoch_callback
        self.use_wandb = use_wandb
        self._step = 0
        self.epochs = epochs
        self.sub_steps_per_datapoint = sub_steps_per_datapoint
        self.losses = []
    
    def train(self, key: jax.Array)->None:
        """ 
        Train the hypernetwork
        NB this replaces self.hypernetwork with the trained hypernetwork
        :parameter key: a jax prng key
        """
        if self.use_wandb:
            import wandb
        key_gen = key_generator(key)
        batches_per_epoch = len(self.train_loader)
        print(f"Start training for {self.epochs} epochs with {batches_per_epoch} batches per epoch and {self.sub_steps_per_datapoint} gradient steps per batch.")
        for epoch in range(1, self.epochs+1):
            print(f"Start epoch {epoch}.")
            epoch_losses = np.empty(shape=(batches_per_epoch,), dtype=np.float32)
            for index, batch in enumerate(self.train_loader):
                self._step += 1
                self.hypernetwork, self.optimizer_state, loss = self.train_step(
                    self.hypernetwork,
                    batch, 
                    self.optimizer_state,
                    next(key_gen)
                )
                # do book keeping and logging
                epoch_losses[index] = loss
                
                report = {'batch/loss': loss, 'batch/global_step': self._step,
                            'epoch': epoch, 'batch/step_within_epoch': index}
                metrics_report = self.metric_collector.on_batch_end(
                    hypernetwork=self.hypernetwork,
                    loss=loss,
                    step=self._step,
                    step_within_epoch=index+1,
                    example_batch=self.example_batch,
                    optimizer_state=self.optimizer_state,
                    epoch=epoch,
                    key=next(key_gen)
                )
                metrics_report = metrics.add_prefix_to_dictionary_keys(metrics_report, prefix='batch/')
                report.update(metrics_report)
                if self.after_step_callback is not None:
                    self.after_step_callback(
                        step_within_epoch=index+1,
                        epoch=epoch,
                        loss=loss,
                        hypernetwork=self.hypernetwork,
                        optimizer_state=self.optimizer_state,
                        report=report,
                    )
                if self.use_wandb:
                    wandb.log(report)

            # book keeping, logging, and validation after epoch
            average_loss = np.mean(epoch_losses)
            print(f"    Finished epoch {epoch} with average loss: {average_loss}.")
            self.losses.append(average_loss)
            report = {'epoch/loss':average_loss}
            metrics_report = self.metric_collector.on_epoch_end(
                hypernetwork=self.hypernetwork,
                loss=average_loss,
                step=self._step,
                example_batch=self.example_batch,
                validation_loader=self.validation_loader,
                optimizer_state=self.optimizer_state,
                epoch=epoch,
                key=next(key_gen)
            )
            metrics_report = metrics.add_prefix_to_dictionary_keys(metrics_report, 'epoch/')
            report.update(metrics_report)
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(
                    step_within_epoch=None,
                    epoch=epoch,
                    loss=average_loss,
                    hypernetwork=self.hypernetwork,
                    optimizer_state=self.optimizer_state,
                    report=report
                )
            if self.use_wandb:
                wandb.log(report)
        # training is done
        print("Finished training.")
        if self.use_wandb:
            print("Uploading model to wandb.")
            import tempfile
            with tempfile.NamedTemporaryFile() as f:
                eqx.tree_serialise_leaves(f, self.hypernetwork)
                wandb.log_model(path=f.name, name='hypernetwork.eqx')
