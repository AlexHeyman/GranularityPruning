# GranularityPruning

This codebase is a tweaked version of the one used to run the experiments in the forthcoming paper "Fine Granularity is Critical for Intelligent Neural Network Pruning", published to enable easier reproduction of those experiments.

## Requirements

This codebase was developed for Python 3.11.7, with PyTorch 2.3.0 and Torchvision 0.18.0 as dependencies. CUDA is technically optional, but the codebase was developed for use with CUDA 12.1. Earlier or later versions of the requirements may or may not work.

## Usage

This codebase's front end is the file `run.py`.
```
python run.py <args>
```
See the comment at the top of `run.py` for a full list and explanation of the valid arguments.

The usual first step is to initialize a network and store the initialization as a checkpoint file, for instance:
```
python run.py -d data/cifar10 -c checkpoints/resnet20 -n resnet20 -g channels -p init --save init.pt
```
(Note that `-g` must be specified for simplicity-of-implementation reasons, but when initializing a network, it affects nothing aside from the network's size in memory.)

Then, you can perform a sequence of iterative pruning rounds with a command like this:
```
python run.py -d data/cifar10 -c checkpoints/resnet20 -n resnet20 -g weights -p magnitude -f 0.125 -r 5 --source init.pt --save iterative.pt --output_path output/resnet20/iterative.txt
```
After that, if you wanted to do a random reinitialization experiment on the pruned network, you could use a command like this:
```
python run.py -d data/cifar10 -c checkpoints/resnet20 -n resnet20 -g weights -p none --source none --masks_source iterative.pt --save reinit.pt --output_path output/resnet20/reinit.txt
```
Or if you wanted to do a layer-matched random pruning experiment, you could use a command like this:
```
python run.py -d data/cifar10 -c checkpoints/resnet20 -n resnet20 -g weights -p random --pbl_source iterative.pt -r 100 --source none --save layer_matched.pt --output_path output/resnet20/layer_matched.txt
```
If you wanted to do a normal, non-layer-matched random-pruning-at-initialization experiment, you could use a command like this:
```
python run.py -d data/cifar10 -c checkpoints/resnet20 -n resnet20 -g weights -p random -f 0.125 -l all -r 100 --source init.pt --save random.pt --output_path output/resnet20/random.txt
```
If you just wanted to train the network without any pruning, you could use a command like this:
```
python run.py -d data/cifar10 -c checkpoints/resnet20 -n resnet20 -g weights -p none --source init.pt --save vanilla.pt --output_path output/resnet20/vanilla.txt
```
If you wanted to (for instance) compute and output the top-3 accuracy and the layer dimensionalities of an already-trained network, you could use a command like this, setting `max_iteration` to 0 to ensure no further training is performed:
```
python run.py -d data/cifar10 -c checkpoints/resnet20 -n resnet20 -g weights -p none --source iterative.pt --max_iteration 0 --accuracy_k 3 --report_dims true --output_path output/resnet20/iterative_metrics.txt
```