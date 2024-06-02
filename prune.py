import sys
import numbers
import torch

def get_weight_magnitude_scores(device, model):
  # Assumes model is a WeightMaskedModule
  return [torch.abs(weight) for weight in model.masked_weight_tensors]

def get_block_mean_magnitude_scores(device, model):
  # http://www.jfrankle.com/lth-block-sparsity.pdf
  # Assumes model is a BlockMaskedModule
  scores = []

  for i in range(len(model.masks_flat)):
    mask_blocks = model.blocks[i]
    mask_scores = torch.zeros(model.masks_flat[i].shape, device=device)
    mask_scores_flattened = mask_scores.view(-1)

    for j in range(len(mask_blocks)):
      mask_scores_flattened[j] = torch.mean(torch.abs(mask_blocks[j]))
    
    scores.append(mask_scores)
  
  return scores

def get_snip_scores(device, model, criterion, dataset, num_samples_to_use):
  model.eval()

  # Enable gradients on masks so we can compute their gradients
  for mask in model.masks_flat:
    mask.requires_grad = True

  # Compute sum over samples of magnitudes of gradients of loss with respect to
  # each mask element
  magnitudes = [torch.zeros(mask.shape, device=device)
                for mask in model.masks_flat]

  for mask in model.masks_flat:
    if mask.grad is not None:
      mask.grad.zero_()
  
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
  num_samples_used = 0
  for data in dataloader:
    if num_samples_used == num_samples_to_use:
      break
    num_samples_used += 1

    inputs, labels = data
    outputs = model(inputs.to(device))
    loss = criterion(outputs, labels.to(device))
    loss.backward(inputs=model.masks_flat)
    for i in range(len(model.masks_flat)):
      magnitudes[i] += torch.abs(model.masks_flat[i].grad)

  # Disable gradients on masks again so we can modify them in-place manually
  # and training doesn't mess with their values
  for mask in model.masks_flat:
    mask.requires_grad = False

  return magnitudes

def get_stat_scores(device, model, alpha, beta, dataset, batch_size):
  # https://ieeexplore.ieee.org/document/8914512
  # Assumes model is an ExposedMaskedModule
  model.eval()

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False)

  # Use the following facts:
  # - Best estimate of expected value is sum / count
  # - var(X) = E(X^2) - E(X)^2
  # - std(X) = sqrt(var(X))
  position_means = []
  position_devs = []
  num_samples = 0
  
  for data in dataloader:
    inputs, _ = data
    model(inputs.to(device))

    if num_samples == 0:
      for i in range(len(model.exposed_tensors)):
        data_position_sums = torch.sum(model.exposed_tensors[i].detach(), dim=0)
        position_means.append(data_position_sums)
        position_devs.append(torch.square(data_position_sums))
    else:
      for i in range(len(model.exposed_tensors)):
        data_position_sums = torch.sum(model.exposed_tensors[i].detach(), dim=0)
        position_means[i] += data_position_sums
        position_devs[i] += torch.square(data_position_sums)
    
    num_samples += inputs.shape[0]

  means = []
  mean_devs = []
  
  for i in range(len(position_means)):
    position_means[i] /= num_samples
    position_devs[i] = torch.sqrt(position_devs[i] / num_samples
                                  - torch.square(position_means[i]))
    if len(position_means[i].shape) == 1:
      means.append(position_means[i])
      mean_devs.append(position_devs[i])
    else:
      extra_dims = tuple(range(1, len(position_means[i].shape)))
      means.append(torch.mean(position_means[i], dim=extra_dims))
      mean_devs.append(torch.mean(position_devs[i], dim=extra_dims))
  
  eps = 1e-8
  return [(-alpha * 1 / (means[i] + eps) + mean_devs[i] / beta).view(
    model.masks_flat[i].shape) for i in range(len(model.masks_flat))]

def get_random_scores(device, model):
  return [torch.rand(mask.shape, device=device) for mask in model.masks_flat]

def set_masked_out_scores(model, scores, value):
  for i in range(len(scores)):
    scores[i] = torch.where(model.masks_flat[i] == 0,
      torch.Tensor([value]).type(scores[i].dtype).to(scores[i].device),
      scores[i])

def prune(model, scores, groups, fraction_to_keep, largest=False):
  # groups is an ordered collection of ordered collections of mask indices,
  # or else one of the strings 'all_separate' or 'all_together'
  # fraction_to_keep can be either a number for all groups or an ordered
  # collection of numbers, one for each group
  if groups == 'all_separate':
    groups = [[i] for i in range(len(model.masks_flat))]
  elif groups == 'all_together':
    groups = [[i for i in range(len(model.masks_flat))]]
  
  if isinstance(fraction_to_keep, numbers.Number):
    fraction_to_keep = [fraction_to_keep for group in groups]
  
  for i in range(len(groups)):
    flattened_scores = torch.cat([scores[j].view(-1) for j in groups[i]])
    num_indices_to_prune = round(
      (1 - fraction_to_keep[i]) * len(flattened_scores))
    indices_to_prune = torch.topk(flattened_scores, num_indices_to_prune,
                                  largest=largest, sorted=False).indices
    group_masks_array = torch.ones(flattened_scores.shape,
                                   device=flattened_scores.device)
    group_masks_array[indices_to_prune] = 0
    begin_index = 0
    for j in groups[i]:
      flattened_mask = model.masks_flat[j].view(-1)
      end_index = begin_index + len(flattened_mask)
      flattened_mask *= group_masks_array[begin_index:end_index]
      begin_index = end_index

def prune_iteratively(device, ts, groups, num_rounds, fraction_to_keep,
                      max_iteration, score_func, largest=False,
                      checkpoint_dir=None, checkpoint_name=None):
  # groups is an ordered collection of ordered collections of mask indices,
  # or else one of the strings 'all_separate' or 'all_together'
  # fraction_to_keep can be either a number for all groups or an ordered
  # collection of numbers, one for each group
  # score_func takes (device, model) and returns a list of float tensors
  model = ts.model
  
  if groups == 'all_separate':
    groups = [[i] for i in range(len(model.masks_flat))]
  elif groups == 'all_together':
    groups = [[i for i in range(len(model.masks_flat))]]
  
  if isinstance(fraction_to_keep, numbers.Number):
    fraction_to_keep = [fraction_to_keep for group in groups]
  
  fraction_per_round = [f ** (1 / num_rounds) for f in fraction_to_keep]
  desired_fraction_left = [1 for group in groups]
  for i in range(num_rounds):
    print(('Round %d of %d' % (i + 1, num_rounds)), file=ts.print_file)
    
    for j in range(len(groups)):
      desired_fraction_left[j] *= fraction_per_round[j]
    
    ts.train(max_iteration)
    print('', file=ts.print_file)
    scores = score_func(device, model)
    if largest:
      set_masked_out_scores(model, scores, float('inf'))
    else:
      set_masked_out_scores(model, scores, float('-inf'))
    prune(model, scores, groups, desired_fraction_left, largest)
    
    ts.load_checkpoint(checkpoint_dir, checkpoint_name, masks_special='exclude')
