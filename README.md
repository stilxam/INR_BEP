# Repository for education on INRs
This repository contains old code I (Simon) used for a research project a while back. You'll likely want to make some adjustments to it to fit your specific needs. E.g. in that research project, the target function we wanted to approximate was an actual continuous function, so most code is built around the idea that the ground truth is a continuous function. For most of you, this will likely not be the case, so you might want to make some adjustments. Additionally, you should probably not trust this code blindly, but check that it does what you think it does. 

Still, this code can serve as an example of how you can implement your models, training loops, etc. And it contains quite some useful utilities as well. If you have any questions about it, feel free to contact me!

However, it is by no means mandatory to make use of this repository. If you want to e.g. apply the method of paper A on the dataset of paper B, and paper A comes with a working implementation by the authors, you can absolutely just clone the repository from paper A and start working from there. If you do so, the advice on not trusting the code blindly still stands: check that the version of the repo that you've found, actually does what the paper says, and make note of any differences.


## Contents
- **inr_utils**: a package with generic utilities for training INRs. This includes a training loop, some metrics, utilities for workin with INRs in the context of images, etc.
- **hypernetwork_utils**: a package with generic utilities for training hypernetworks. Similar to `inr_utils` but with a slightly different setup.
- **model_components**: a package with layer and model classes and other components relevant for building INRs and hypernetworks.

NB `hypernetwork_utils` imports some stuff from `inr_utils`, so if you start extending or changing these modules, make sure not to create circular imports where A imports B but B tries to import A. 



# Words of advice
You have very limited time to conduct all experiments, but also many hands that can work on this. Try to do all the coding as a group instead of letting one group member do all the coding. Ideally, you'll have one person work on the model architecture, another person on the train step, yet another on the metrics used during training,  etc. The following are some words of advice on how to make this go smoothly:
## Modularity
Try to set things up in a modular fashion. That way different group members can work on different aspects of the project independently and in parallel. This includes:
* Don't fuse things that should be separate. E.g. don't make the train step part of your model. If you keep them separate, one person can work on the model, and another can work on the trainingloop. If you put everything into one class/function, this becomes very difficult. 
* Don't hard-code things that you are not sure about. E.g. if you're writing an implementation of a train step for a Neural Radiance field, maybe don't hard-code the way you approximate the integrals in the volume rendering into that train step: if you later want to try different ways of approximating those integrals, or of volume rendering in general, that becomes much more difficult to do if it's hard-coded.
## Documentation
In order to be able to use eachothers code, it is of paramount importance that you document your code well. That means writing docstrings (and ideally type hints) for **all** your functions and classes, and writing comments detailing complex approaches or reasoning behind code that might not be immediately obvious.

I know writing docstrings can feel like a chore, but in the end, it likely takes less time than texting the same information to your group mate who's unsure about how to use your function/class, and it definitely takes less time than your group mate will otherwise spend on trying to figure out by themselves what kind of input your code expects exactly, and what it returns. 

## Self explanatory code
Try to write code that is as self explanatory as possible (although it being self explanatory is no excuse for not writing docstrings and comments!). This includes giving functions and classes names that explain what they do, and giving variables names that explain what they represent.

If you use a single-letter variable name from some paper because you don't know a better name to give to it, consider adding a comment with a link to the paper so your group mates can see the formulae. 

## Git
Make good use of Git! To be honest, I myself am no git expert, so I can't tell you much about best practices here. But if you know you are going to make code changes that may break other things (temporarily), consider doing so in a new branch that you only merge with the main branch again when you know everything works again. This way, your groupmates can still do their work while you are temporarily breaking things.  

## Start small
This one is less about working together and more about not getting stuck with a model that isn't working and a sea of parameters you could tweak without knowing which is the right one. 

Try to start small and build things up. E.g. if you want to implement and train a hypernetwork on your dataset, instead of starting with the full hypernetwork, consider the following approach:
1. Try to train an INR on a single datapoint and experiment with what a good INR architecture is for your dataset.
1. Implement the hypernetwork and apply it to a single datapoint. Then try training the resulting INR alone (without the hypernetwork) on that datapoint. This way you can catch some bugs early on, and you can see if the output of the hypernetwork early during training lends itself well to being trained on that data. (Remember: the performance of some INR architectures can be very sensitive to the specifics of the weight initialization scheme.)
1. Only after succesfully finishing the above two steps try to train the full hypernetwork.

## Debug locally, run on Snellius
Try not to debug anything on Snellius. Your script will have to wait in a queue, so it'll take an unpredictable amount of extra time before you have your stack trace. Instead try to debug your code on some local computer, and then sync the debugged code with Snellius using Github.

## Define success clearly
Before you start running experiments, come up with criteria of succes. Define and implement metrics that tell you how good a model is compared to other models, and implement baselines to compare to.

Don't just compare loss values unless you are entirely certain that they are comparable. Comparing losses for different types of model can often be like comparing apples to oranges. Objective metrics that are model-independent give a much better view of how well your model is performing.

When it comes to finding good hyperparameters for your model, you'll **need** a metric for this to see what picks for your hyperparameters work best. So selecting and implementing metrics is not something that can wait until the end of the project. 

If you can, try to get an idea beforehand of what good values are both for your metrics and, if possible, for your loss function.

## Try not to watch models train
At some point you're going to have some models training. Maybe you're debugging your code, and you're finally managing to get a model to fully train. Or maybe you're looking for good hyperparameters by training a few models to completion. Unfortunately however, the loss, or some metric, is not quite as low as you had hoped.

It can be very tempting in that situation to just make a few adjustments and to try again. And again. And again. Maybe training a model only takes like 5 minutes, so it's not even such a long wait, right? Or maybe training a model takes a full hour, so you tell yourself that you can just work on other things on the side, and just keep half an eye on the loss function.

**Don't do that.** Please, for the sake of your own sanity, don't do that. In the scenario where training a model takes only 5 minutes, you'll end up spending hours hardly doing anything, but never really relaxing either. In the case it takes two hours, if you keep looking at the loss in the mean while, it's hard to really focus on the other task you *should* be doing. Either way, the experience is draining, and you'll likely end up feeling very dissatisfied with how you spent your time. 

If the model is training okay-ish, but you're not entirely happy with the performance, don't manually adjust hyperparameters, just setup a hyperparameter sweep instead and have snellius automatically try a whole bunch of things. Look at the examples in `inr_hyperparameter_sweep_example.ipynb` and... for this.

If the model is performing horribly no matter what hyperparameters you pick, don't keep trying new hyper parameters, but see if there is some bug in e.g. the data loading. Are you sure that the data has the shape you expect (maybe channels are last where you expected first, or vice versa)? Did you accidentally rescale the data twice (from [0, 255] to [0, 1] and then accidentaly to [0, 1/255])? Can you maybe get it to work on a mini version of your dataset? Can you explain what all the code is doing to one of your group mates, or to a [rubber duck](https://rubberduckdebugging.com/)?

# Tools from `common_dl_utils` and `common_jax_utils`
When doing experiments in deep learning, you get to the point where, you've written your model and an architecture, you've written your training loop, you've written your metrics and code to log those metrics either to Weights and Biases or so some log files on your file system, and now you want to run your experiment.

For this you need some code that creates the model, and runs the train loop. But you want it to be flexible: maybe it should read the hyperparameters from some config file or object. This starts out easy enough: you know what the parameters are that need to be decided, so you can just fill them out like

```python
model = MyModelClass(
    input_size = config["input_size"],
    output_size = config["output_size"],
    hidden_size = config["hidden_size"],
    depth = config["depth"],
    w0 = config["w0"]
)
```
Easy enough! But then, while running experiments, you decide that you want to see if maybe the Gaussian layers work better than the Siren layers. Okay, well, just add an if-statement

```python
if config["layer"] == "siren":
    model = MyModelClass(
        input_size = config["input_size"],
        output_size = config["output_size"],
        hidden_size = config["hidden_size"],
        depth = config["depth"],
        w0 = config["w0"]
    )
elif config["layer"] == "gauss":
    model = MyModelClassGauss(
        input_size = config["input_size"],
        output_size = config["output_size"],
        hidden_size = config["hidden_size"],
        depth = config["depth"],
        inverse_scaling = config["inverse_scaling"]
    )
else:
    raise NotImplementedError(f"No model implemented for {config["layer"]=}")
```
You keep experimenting and over the course of the project, you add if-statements on whether to use all sorts of things, like stochastic weight averaging in the training loop, what exact loss function to use, 2 more types of models, and of course everything to do with the baselines. During this process, this "glue" code holding everything together can become quite complicated, and each time you want to make an adjustment, the cost of doing so rises because you have more code to modify.

This is the problem that the tools from `common_dl_utils` and `common_jax_utils` try to fix: they provide this "glue" code in a generic way. The idea is as follows: while writing the functions for creating your model, train loop, etc. (e.g. the `__init__` method, or some `from_params` class-method), you indicate what type each argument should be. `common_dl_utils` keeps a **registry of types that require special attention**. If an argument doesn't need special attention, e.g. it's an `int` or a `float`, we can just pull it from the config automatically by matching the parameter name. If it does need special attention, e.g. because it is a class that itself needs initializing, we just first prepare that class from config in the same way.

So for example, if your model code in `my_model_code.py` looks like
```python
import jax
import equinox as eqx
from model_components.inr_layers import INRLayer

class AutoEncoder(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    auxiliary_regressor: eqx.Module

    def __init__(self, encoder:eqx.Module, decoder:eqx.Module, auxiliary_regressor: eqx.Module):
        ...
    
    def __call__(self, x):
        z = self.encoder(x)
        y_pred = self.auxiliary_regressor(z)
        x_pred = self.decoder(z)
        return x_pred, y_pred

class ConvEncoder(eqx.Module):
    ...
    def __init__(self, data_channels:int, latent_size:int, hidden_channels:int, kernel_size:int, depth:int, ndim:int, key: jax.Array):
        ...


class Regressor(eqx.Module):
    ...
    def __init__(self, latent_size:int, pred_size: int, depth:int, key:jax.Array):
        ...


class INRDecoder(eqx.Module):
    ...
    def __init__(
            self, 
            latent_size: int, 
            mlp_depth:int, 
            mlp_width: int, 
            ndim: int, 
            inr_depth: int
            data_channels: int,
            inr_layer_type: type[INRLayer],
            inr_layer_kwargs: dict,
            key: jax.Array
            ):
        ...
    
    def __call__(self, z:jax.Array)->eqx.Module:
        ...
        return x_pred_inr

class ConvDecoder(eqx.Module):  # baseline
    ...
    def __init__(
        self,
        latent_size:int,
        data_channels:int,
        hidden_channels:int,
        depth:int,
        kernel_size:int,
        key: jax.Array
    ):
    ...

    def __call__(self, z:jax.Array)->jax.Array:
        ...
        return x_pred
```
then you can automatically match this with a config like
```python
config = dict(
    architecture = "my_model_code.py",
    model_type = "AutoEncoder",
    encoder = "ConvEncoder",
    encoder_config = dict(
        hidden_channels = 64,
        kernel_size = 3,
        depth = 6
    ),
    auxiliary_regressor = "Regressor",
    auxiliary_regressor_config = dict(
        pred_size = 10,
        depth = 4
    ),
    decoder = "INRDecoder",
    decoder_config = dict(
        mlp_depth = 3,
        mlp_width = 1024,
        inr_depth = 6,
        inr_layer_type = "inr_layers.SirenLayer",  # add "./model_components" to default modules
        # because 'inr_layer_type' has a typehint like `type[...]`, this too will result in the appropriate class
        # in the appropriate module, but without that class getting initialized.
        inr_layer_kwargs = {"w0": 30.}
    )
    # the following are shared between the encoder, auxiliary_regressor, and decoder
    latent_size = 64,
    data_channels = 3,
    ndim = 2,
)
```
and create a model from it using `common_jax_utils.run_utils.get_model_from_config_and_key` like
```python
from jax import random
from common_jax_utils.run_utils import get_model_from_config_and_key
import model_components

model = get_model_from_config_and_key(
    prng_key = jax.random.key(1234),
    config = config,
    additional_model_default_modules = [model_components]  # so that we could easily refer to inr_layers.SirenLayer
)
```
NB this goes well because `common_jax_utils`, upon import, automatially adds `equinox.Module` to the registry of types that require special attention. If you want to add another type to this registry, you can do so by using `common_jax_utils.types.register_type` or `common_dl_utils.type_registry.register_type`. With this, type annotations such as `Optional[equinox.Module]`, or `list[equinox.Module]` are also recognised.

NB you can also add more layers of nesting in your config if you want. See `inr_example.ipynb` and `hyper_network_example.ipynb` for more examples.

Now if you later decide that you want to experiment with a res-net for your encoder, and write a corresponding `eqx.Module` in some `res_architecute.py` module, like
```python
class ResidualConvEncoder(eqx.Module):
    ...
    def __init__(self, data_channels:int, latent_size:int, hidden_channels:int, kernel_size:int, num_blocks:int, ndim:int, key:jax.Array):
        ...
```
you can get this from your config by modifiying it slightly like
```python
config["encoder"] = ("./res_architecture.py", "ResidualConvEncoder")
config["encoder_config"] = dict(
    hidden_channels=64,
    kernel_size=3,
    num_blocks=8
)
```
without having to write extra logic aroud this.

NB setting `config["encoder"]` to a tuple of two strings `(path, name)` tells `get_model_from_config_and_key` to import the module found at `path` and then look for the class specified by `name` within that module. If you specify it as a single string `name`, it will look for `name` in the given default modules (i.e. the one specified by `config["architecture"]` and those specified by `additional_model_default_modules`). You can also specify it as `(path:str, name:str, sub_config:dict)`, in which case the parameters of the class/callable specified by `name` will first be filled in with those found in `sub_config`, then by andy default parameters of said class, and finally by those found in `config`. Finally, `name` need not refer to just a class, but can also refer to a method, e.g. `('./model_components', 'inr_layers.SirenLayer.from_config', {"is_first_layer"=True})`

## Using these tools
You are free to use these tools if you want, or to ignore them if you don't want to use them.
If you run into any bugs in `common_dl_utils` or `common_jax_utils` (or any of the code in this repository) please let me know!
If you have any questions on how to do things with them, don't hesitate to contact me.

Also, if providing correct type hints for your code is for some reason too inconvenient, these tools don't actually do type checking, so if you want, you can just use incorrect/incomplete type hints as long as they indicate whether the argument is something that will need initializing with its own arguments or not.