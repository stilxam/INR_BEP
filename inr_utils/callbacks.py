""" 
Module containing callbacks for the training.train_inr function.
"""
import abc
import pprint
from typing import Callable, Union, Any

from jax import numpy as jnp

from common_dl_utils.metrics import MetricCollector
from common_dl_utils.type_registry import register_type


@register_type
class Callback(abc.ABC):
    @abc.abstractmethod
    def __call__(self, step, loss, inr, state, optimizer_state)->Any:
        pass


def print_loss(step, loss, inr, state, optimizer_state, after_every=200):
    if (step+1)%after_every == 0:
        print(f"Loss at step {step+1} is {loss}.")

# ===========================================================================================
# the callbacks below return dictionaries containing logs that can be collected by other callbacks

class MetricCollectingCallback(Callback):
    def __init__(self, metric_collector:MetricCollector):
        self.metric_collector = metric_collector
    def __call__(self, step, loss, inr, state, optimizer_state):
        return self.metric_collector.on_batch_end(inr=inr, state=state, loss=loss, optimizer_state=optimizer_state, step=step)

def report_loss(step, loss, inr, state, optimizer_state):
    return {'loss': loss}


# ===========================================================================================
# the callback below raises an error if the loss contains NaNs
# this way we don't keep training while only passing around NaNs
def raise_error_on_nan(step, loss, inr, state, optimizer_state):
    if jnp.any(jnp.isnan(loss)):
        raise RuntimeError(f"NaN occurred in loss at step {step}.")

# ===========================================================================================
# the one callback to rule them all

class ComposedCallback(Callback):
    def __init__(
            self, 
            *callbacks:Union[Callback, Callable], 
            use_wandb:bool, 
            show_logs:bool, 
            display_func:Callable=pprint.pprint
            ):
        self._wandb = None
        if use_wandb: # wanted to keep wandb an optional dependency
            import wandb
            self._wandb = wandb
        self.callbacks = list(callbacks)
        self.show_logs = show_logs
        self.display_func = display_func
    
    def __call__(self, step, loss, inr, state, optimizer_state):
        logs = {}
        for callback in self.callbacks:
            log = callback(step, loss, inr, state, optimizer_state)
            if log is not None:
                logs.update(log)
        if self._wandb is not None and logs:
            self._wandb.log(logs)
        if self.show_logs and logs:
            self.display_func(logs)

# Add this new callback class after the existing callbacks

@register_type
class AudioMetricsCallback(Callback):
    """Callback for computing and logging audio metrics during training."""
    
    def __init__(self, 
                 metric_collector: MetricCollector,
                 print_metrics: bool = True,
                 print_frequency: int = 100,
                 save_final_audio: bool = True,
                 save_path: str = './reconstructed_audio.wav'):
        """
        Args:
            metric_collector: MetricCollector instance containing AudioMetricsOnGrid
            print_metrics: Whether to print metrics to console
            print_frequency: How often to print metrics (in steps)
            save_final_audio: Whether to save the final reconstructed audio
            save_path: Path where to save the final audio
        """
        self.metric_collector = metric_collector
        self.print_metrics = print_metrics
        self.print_frequency = print_frequency
        self.save_final_audio = save_final_audio
        self.save_path = save_path
        self.final_audio = None

    def __call__(self, step, loss, inr, state, optimizer_state):
        metrics = self.metric_collector.on_batch_end(
            inr=inr, 
            state=state, 
            loss=loss, 
            optimizer_state=optimizer_state, 
            step=step
        )
        
        if metrics and 'reconstructed_audio' in metrics:
            self.final_audio = metrics.pop('reconstructed_audio')  # Store but don't log the audio array
        
        if self.print_metrics and step % self.print_frequency == 0 and metrics:
            print(f"\nAudio Metrics at step {step}:")
            for name, value in metrics.items():
                if name.startswith('audio_'):
                    print(f"{name.replace('audio_', '')}: {value:.4f}")
        
        # Save final audio at the last step if requested
        if self.save_final_audio and self.final_audio is not None:
            import soundfile as sf
            sf.write(self.save_path, self.final_audio, self.metric_collector.metrics[0].sr)
        
        return metrics

    def get_final_audio(self):
        """Return the final reconstructed audio array."""
        return self.final_audio
