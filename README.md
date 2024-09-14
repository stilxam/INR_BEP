# Repository for education on INRs
This repository contains old code I (Simon) used for a research project a while back. You'll likely want to make some adjustments to it to fit your specific needs. E.g. in that research project, the target function we wanted to approximate was an actual continuous function, so most code is built around the idea that the ground truth is a continuous function. For most of you, this will likely not be the case, so you might want to make some adjustments.

Still, this code can serve as an example of how you can implement your models, training loops, etc. And it contains quite some useful utilities as well. If you have any questions about it, feel free to contact me!


## Contents
- **inr_utils**: a package with generic utilities for training INRs 
- **hypernetwork_utils**: a package with generic utilities for training hypernetworks
- **model_components**: a package with layer and model classes and other components relevant for building INRs and hypernetworks.

# Tools from `common_dl_utils` and `common_jax_utils`
TODO write about how to use these tools

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

## Start small
This one is less about working together and more about not getting stuck with a model that isn't working and a sea of parameters you could tweak without knowing which is the right one. 

Try to start small and build things up. E.g. if you want to implement and train a hypernetwork on your dataset, instead of starting with the full hypernetwork, consider the following approach:
1. Try to train an INR on a single datapoint and experiment with what a good INR architecture is for your dataset.
1. Implement the hypernetwork and apply it to a single datapoint. Then try training the resulting INR alone (without the hypernetwork) on that datapoint. This way you can catch some bugs early on, and you can see if the output of the hypernetwork early during training lends itself well to being trained on that data. (Remember: the performance of some INR architectures can be very sensitive to the specifics of the weight initialization scheme.)
1. Only after succesfully finishing the above two steps try to train the full hypernetwork.