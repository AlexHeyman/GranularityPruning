'''
Usage: python run.py <args>
Arguments:
-d, --dataset_root_dir: dataset root directory
-c, --checkpoint_dir: checkpoint directory
-o, --output_path: path to file where text output should be written; default:
  sys.stdout
--save: path relative to -c where model checkpoint should be saved after all
  training/pruning/other operations on the model are done. If "none", the model
  will not be saved. If unspecified, the model will be saved directly in -c with
  a filename based on its pruning parameters.
-n, --network_type: "resnet20" or "dbsn"
-p, --pruning_mode: "init", "none", "magnitude" (weights only), "bad_magnitude"
  (weights only), "mean_mag" (no weights), "bad_mean_mag" (no weights), "snip",
  "bad_snip", "stat" (channels or nodes only), "bad_stat" (channels or nodes
  only), or "random". "Bad" versions of non-random pruning modes remove the
  network components with the highest scores and leave the ones with the lowest
  scores, instead of vice versa.
-g, --granularity: "weights" (both networks), "kernels" (resnet20 only),
  "channels" (resnet20 only), or "nodes" (dbsn only)
-i, --init_iteration: iteration to train up to as part of initialization;
  0 by default
-f, --fraction_to_prune: total fraction of prunable network components to prune
-r, --rounds: number of iterative pruning rounds; 100 for one-shot pruning at
  initialization. Irrelevant if -p == "init" or -p == "none".
-l, --prune_by_layer: "none", "all", or "by_res_type" (resnet20 only);
  default: "none"
--pbl_source: if specified, path relative to -c of checkpoint file whose
fractions of pruned elements in each layer should be reproduced by this
run's pruning. Makes -f and -l irrelevant.
-s, --source: if -p != "init", path relative to -c of checkpoint file to
  initialize the network to and reset it to after each iterative pruning round;
  default: "pruning_<-i>.pt". If (-p == "none" or -r == 100) and -s == "none",
  the network will be freshly randomly initialized.
--masks_source: if -p == "none", path relative to -c of checkpoint file to
  initialize the network's masks from; default: same as -s
--num_workers: number of worker processes to use to load data; default: 2
--batch_size: network batch size; default: 128
--train_iterations: number of iterations to train for in each round;
  default: 30000
--print_every: frequency in iterations with which running loss is printed and
  reset and validation accuracy on the test set is computed and printed;
  default: 1000. If this is 0 or negative, these periodic evaluations will
  never occur.
--snip_batch_size: if -p == "snip" or "bad_snip", number of training samples to
  use for each round of SNIP; default: 2000
--stat_alpha: if -p == "stat" or "bad_stat", statistical criterion "alpha"
  parameter; default: 1
--stat_beta: if -p == "stat" or "bad_stat", statistical criterion "beta"
  parameter; default: 1
'''

import sys
from os import path
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import masked_models
import prune
from training_system import TrainingSystem
from utilities import compute_accuracy, ScaledTanh, DualTransformableMNIST,\
     DBSNTrainDualTransform

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_root_dir', type=str, default=None)
parser.add_argument('-c', '--checkpoint_dir', type=str, default=None)
parser.add_argument('-o', '--output_path', type=str, default=None)
parser.add_argument('--save', type=str, default=None)
parser.add_argument('-n', '--network_type', type=str,
                    choices=['resnet20', 'dbsn'], default=None)
parser.add_argument('-p', '--pruning_mode', type=str, default=None)
parser.add_argument('-g', '--granularity', type=str, default=None)
parser.add_argument('-i', '--init_iteration', type=int, default=0)
parser.add_argument('-f', '--fraction_to_prune', type=float, default=None)
parser.add_argument('-r', '--rounds', type=int, default=None)
parser.add_argument('-l', '--prune_by_layer', type=str,
                    choices=['none', 'all', 'by_res_type'], default='none')
parser.add_argument('--pbl_source', type=str, default=None)
parser.add_argument('-s', '--source', type=str, default=None)
parser.add_argument('--masks_source', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_iterations', type=int, default=30000)
parser.add_argument('--print_every', type=int, default=1000)
parser.add_argument('--snip_batch_size', type=int, default=2000)
parser.add_argument('--stat_alpha', type=float, default=1)
parser.add_argument('--stat_beta', type=float, default=1)

args = parser.parse_args()
dataset_root_dir = args.dataset_root_dir
checkpoint_dir = args.checkpoint_dir

output_path = args.output_path
if output_path is None:
  output_file = sys.stdout
else:
  output_file = open(output_path, 'w')

save_rel_path = args.save
network_type = args.network_type
pruning_mode = args.pruning_mode
granularity = args.granularity
init_iteration = args.init_iteration
fraction_to_prune = args.fraction_to_prune
num_pruning_rounds = args.rounds
prune_by_layer = args.prune_by_layer
pbl_source_checkpoint = args.pbl_source

source_checkpoint = args.source
default_source_checkpoint = ('pruning_%d.pt' % init_iteration)
if source_checkpoint is None:
  source_checkpoint = default_source_checkpoint

masks_source_checkpoint = args.masks_source
num_workers = args.num_workers
batch_size = args.batch_size
train_iterations = args.train_iterations
print_every = args.print_every
snip_batch_size = args.snip_batch_size
stat_alpha = args.stat_alpha
stat_beta = args.stat_beta

device = torch.device('cpu')
if torch.cuda.is_available():
  device = torch.device('cuda')

print('Using device: %s' % device, file=output_file)

if network_type == 'resnet20':
  stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

  train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32), padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
    ])
  trainset = torchvision.datasets.CIFAR10(root=dataset_root_dir, train=True,
                                      download=False, transform=train_transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers)

  test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
    ])
  testset = torchvision.datasets.CIFAR10(root=dataset_root_dir, train=False,
                                      download=False, transform=test_transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
             'ship', 'truck')
  update_lr_every = 1
  
  def learning_rate_fn(iteration):
    lr = 3e-2
    if iteration >= 25000:
      lr *= 1e-2
    elif iteration >= 20000:
      lr *= 1e-1
    else:
      lr *= iteration / 20000
    return lr

  def make_optimizer(parameters_to_optimize):
    return torch.optim.SGD(parameters_to_optimize, lr=1, momentum=0.9,
                           weight_decay=0.0001)

  def init_params(m):
    if isinstance(m, (nn.Conv2d, masked_models.Conv2dWeightTensorMasked,
                      nn.Linear)):
      nn.init.xavier_normal_(m.weight, gain=1)
  
  if granularity == 'weights':
    model = masked_models.resnet20_weight_masked(
      num_classes=len(classes)).to(device)
    valid_pruning_modes = ['magnitude', 'bad_magnitude', 'snip', 'bad_snip',
                           'random']
  elif granularity == 'kernels':
    model = masked_models.resnet20_kernel_masked(
      num_classes=len(classes)).to(device)
    valid_pruning_modes = ['mean_mag', 'bad_mean_mag', 'snip', 'bad_snip',
                           'random']
  elif granularity == 'channels':
    model = masked_models.resnet20_channel_masked(
      num_classes=len(classes)).to(device)
    valid_pruning_modes = ['mean_mag', 'bad_mean_mag', 'snip', 'bad_snip',
                           'stat', 'bad_stat', 'random']
  else:
    print('Granularity "%s" is not valid for network type "%s"'\
                      % (granularity, network_type), file=output_file)
    output_file.close()
    exit()
else: # DBSN
  stats = ((0.5,), (0.5,)) # Maps [0, 1] to [-1, 1]
  
  train_dual_transform = DBSNTrainDualTransform()
  trainset = DualTransformableMNIST(root=dataset_root_dir, train=True,
                                    download=False,
                                    dual_transform=train_dual_transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers)

  test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(*stats)
      ])
  testset = torchvision.datasets.MNIST(root=dataset_root_dir, train=False,
                                    download=False, transform=test_transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers)

  classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
  update_lr_every = 500
  
  def learning_rate_fn(iteration):
    return 2e-5
  
  def make_optimizer(parameters_to_optimize):
    return torch.optim.Adam(parameters_to_optimize, lr=1)

  def init_params(m):
    if isinstance(m, (nn.Linear, masked_models.LinearWeightMasked)):
      nn.init.uniform_(m.weight, -0.05, 0.05)
      nn.init.constant_(m.bias, 0)

  layer_widths = [2500, 2000, 1500, 1000, 500]
  nonlinearities = [ScaledTanh(1.7159, 0.6666) for _ in range(len(layer_widths))]
  
  if granularity == 'weights':
    model = masked_models.SimpleMNISTNetWeightMasked(
      layer_widths, nonlinearities).to(device)
    valid_pruning_modes = ['magnitude', 'bad_magnitude', 'snip', 'bad_snip',
                           'random']
  elif granularity == 'nodes':
    model = masked_models.SimpleMNISTNetNodeMasked(
      layer_widths, nonlinearities).to(device)
    valid_pruning_modes = ['mean_mag', 'bad_mean_mag', 'snip', 'bad_snip',
                           'stat', 'bad_stat', 'random']
  else:
    print('Granularity "%s" is not valid for network type "%s"'\
                      % (granularity, network_type), file=output_file)
    output_file.close()
    exit()

model.sync_masks(device)
model.apply(init_params)

ts = TrainingSystem('pruning_%s' % network_type, device, model,
                    make_optimizer(model.non_mask_parameters()),
                    learning_rate_fn, update_lr_every, print_every,
                    [], checkpoint_dir,
                    trainloader, testloader, print_file=output_file)

if pruning_mode == 'init':
  print('Initializing network', file=output_file)
  ts.train(init_iteration)
  
  if save_rel_path is None:
    save_rel_path = ('pruning_%d.pt' % init_iteration)
  if save_rel_path != 'none':
    print('Saving checkpoint at %s' % path.join(checkpoint_dir, save_rel_path),
          file=output_file)
    ts.save_checkpoint(filename=save_rel_path)
  
  output_file.close()
  exit()

if pruning_mode == 'none':
  id_string = ''
  
  if source_checkpoint == 'none':
    ts.train(init_iteration)
  else:
    if source_checkpoint != default_source_checkpoint:
      id_string += ('s_(%s)_' % source_checkpoint)
    ts.load_checkpoint(filename=source_checkpoint, masks_special='exclude')
  
  if masks_source_checkpoint is not None:
    id_string += ('m_(%s)_' % masks_source_checkpoint)
    ts.load_checkpoint(filename=masks_source_checkpoint, load_other=False,
                       masks_special='only')
  
  if len(id_string) == 0:
    id_string = 'vanilla_'

  id_string = id_string[:-1]

  print('Beginning non-pruning training (%s)\n' % id_string, file=output_file)
else:
  if pruning_mode not in valid_pruning_modes:
    print('Pruning mode "%s" is not valid for network type "%s" and granularity "%s"'\
          % (pruning_mode, network_type, granularity), file=output_file)
    output_file.close()
    exit()
  
  id_string = '%s_%d_%s_%d' % (granularity, init_iteration, pruning_mode,
                               num_pruning_rounds)

  if pbl_source_checkpoint is not None:
    id_string += '_pbls_(%s)' % pbl_source_checkpoint
    groups = 'all_separate'
    pbl_model_state_dict = torch.load(path.join(checkpoint_dir,
      pbl_source_checkpoint))['model_state_dict']
    fraction_to_prune = []
    for i in range(len(model.masks_flat)):
      mask_tensor = pbl_model_state_dict['mask%d' % i]
      fraction_to_prune.append(1 - (torch.count_nonzero(mask_tensor).item()
                                    / torch.numel(mask_tensor)))
  else:
    id_string += '_%.3f' % fraction_to_prune
    if prune_by_layer == 'all':
      id_string += '_by_layer'
      groups = 'all_separate'
    elif prune_by_layer == 'by_res_type':
      id_string += '_by_res_type'
      groups = [[0], list(range(1, 18 + 1, 2)), list(range(2, 18 + 1, 2))]
    else:
      groups = 'all_together'
  
  if prune_by_layer != 'none':
    id_string += ('_pbl_%s' % prune_by_layer)
  
  if pruning_mode == 'magnitude' or pruning_mode == 'bad_magnitude':
    score_func = prune.get_weight_magnitude_scores
  elif pruning_mode == 'mean_mag' or pruning_mode == 'bad_mean_mag':
    score_func = prune.get_block_mean_magnitude_scores
  elif pruning_mode == 'snip' or pruning_mode == 'bad_snip':
    id_string += ('_%d' % snip_batch_size)
    score_func = lambda device, model: prune.get_snip_scores(
      device, model, ts.criterion, trainset, snip_batch_size)
  elif pruning_mode == 'stat' or pruning_mode == 'bad_stat':
    id_string += ('_%.3f_%.3f' % (stat_alpha, stat_beta))
    score_func = lambda device, model: prune.get_stat_scores(
      device, model, stat_alpha, stat_beta, trainset, batch_size)
  else: # random
    score_func = prune.get_random_scores
  
  prune_largest_scores = (pruning_mode == 'bad_magnitude'
                          or pruning_mode == 'bad_mean_mag'
                          or pruning_mode == 'bad_snip'
                          or pruning_mode == 'bad_stat')
  
  print('Beginning pruning (%s)\n' % id_string, file=output_file)
  
  if num_pruning_rounds == 100:
    if source_checkpoint == 'none':
      ts.train(init_iteration)
    else:
      if source_checkpoint != default_source_checkpoint:
        id_string += ('s_(%s)_' % source_checkpoint)
      ts.load_checkpoint(filename=source_checkpoint, masks_special='exclude')
    
    scores = score_func(device, model)
    prune.prune(model, scores, groups, fraction_to_prune, prune_largest_scores)
  else:
    ts.load_checkpoint(filename=source_checkpoint, masks_special='exclude')
    
    prune.prune_iteratively(device, ts, groups, num_pruning_rounds,
      fraction_to_prune, train_iterations, score_func, prune_largest_scores,
      checkpoint_dir, source_checkpoint)

ts.train(train_iterations)

if save_rel_path is None:
  save_rel_path = ('pruning_%s.pt' % id_string)
if save_rel_path != 'none':
  print('Saving checkpoint at %s' % path.join(checkpoint_dir, save_rel_path),
        file=output_file)
  ts.save_checkpoint(filename=save_rel_path)

model.eval()
print('Train accuracy: %f' % compute_accuracy(device, model, trainloader),
      file=output_file)
print('Test accuracy: %f' % compute_accuracy(device, model, testloader),
      file=output_file)
output_file.close()
