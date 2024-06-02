import time
from os import path
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


class ScaledTanh(torch.nn.Module):

  def __init__(self, a, b):
    super().__init__()
    self.a = a
    self.b = b

  def forward(self, x):
    return self.a * torch.tanh(self.b * x)


class DualTransformableMNIST(torchvision.datasets.MNIST):

  def __init__(self, root, train=True, transform=None, target_transform=None,
               dual_transform=None, download=False):
    super().__init__(root, train, transform, target_transform, download)
    self.dual_transform = dual_transform

  @property
  def raw_folder(self):
    return path.join(self.root, "MNIST", "raw")

  @property
  def processed_folder(self):
    return path.join(self.root, "MNIST", "processed")
    
  def __getitem__(self, index):
    img, target = super().__getitem__(index)
    
    if self.dual_transform is not None:
      img, target = self.dual_transform(img, target)
    
    return img, target


class RandomStretchTransform(torch.nn.Module):
    
  def __init__(self, gamma_x, gamma_y):
    super().__init__()
    self.gammas = torch.Tensor([gamma_x, gamma_y])

  def forward(self, img):
    if isinstance(img, torch.Tensor):
        height, width = img.shape[-2:]
    else: # img is a PIL image
        width, height = img.size
    rand = torch.rand(2) * 2 - 1
    scales = 1 + (self.gammas * rand)
    resized_height = round(height * scales[1].item())
    resized_width = round(width * scales[0].item())
    img = TF.resize(img, (resized_height, resized_width),
                    InterpolationMode.BILINEAR, None, True)
    return TF.center_crop(img, (height, width))


class DBSNTrainDualTransform(torch.nn.Module):
  
  def __init__(self, stats):
    super().__init__()
    self.big_rotation = transforms.RandomRotation(
      (-15, 15), InterpolationMode.BILINEAR)
    self.small_rotation = transforms.RandomRotation(
      (-7.5, 7.5), InterpolationMode.BILINEAR)
    self.final_transform = transforms.Compose([
      RandomStretchTransform(0.2, 0.2),
      transforms.ToTensor(),
      transforms.Normalize(*stats)
      ])

  def forward(self, img, target):
    if target == 1 or target == 7:
      img = self.small_rotation(img)
    else:
      img = self.big_rotation(img)

    return self.final_transform(img), target


def compute_accuracy(device, model, dataloader, k=1):
  correct = 0
  total = 0

  with torch.no_grad():
      for data in dataloader:
          inputs, labels = data
          outputs = model(inputs.to(device))
          topk = torch.topk(outputs, k, dim=1, sorted=False).indices
          total += labels.size(0)
          correct += torch.eq(labels.to(device)[:, None], topk)\
                     .any(dim=1).sum().item()
  
  return correct / total


def compute_layer_dims(device, model, layers, dataset, batch_size, output_file):
  # Computes the effective dimensionality of each layer of the model listed in
  # <layers> and reports them alongside the actual dimensionalities
  # See Eqn. 1 in https://www.cell.com/neuron/pdfExtended/S0896-6273(17)30054-5
  # Assumes model is an ExposedMaskedModule
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False)
  
  # Use the following facts:
  # - Best estimate of mean (expected value) from a sample is sum / count
  # - cov(X, Y) = E(XY) - E(X)E(Y)
  actual_dims = []
  node_means = []
  node_covs = []
  num_samples = 0
  
  start_time = time.time()
  
  print_every = 1000
  next_print = print_every
  
  for data in dataloader:
    if num_samples >= next_print:
      print('%d samples processed' % next_print, file=output_file)
      next_print += print_every
    
    inputs, _ = data
    model(inputs.to(device))

    if num_samples == 0:
      for i in range(len(layers)):
        batch_node_values = model.exposed_tensors[layers[i]].detach()
        batch_node_values = batch_node_values.view(inputs.shape[0], -1)
        actual_dims.append(batch_node_values.shape[1])
        node_means.append(torch.sum(batch_node_values, dim=0))
        batch_nv_products = torch.matmul(batch_node_values.unsqueeze(2),
                                         batch_node_values.unsqueeze(1))
        node_covs.append(torch.sum(batch_nv_products, dim=0))
    else:
      for i in range(len(layers)):
        batch_node_values = model.exposed_tensors[layers[i]].detach()
        batch_node_values = batch_node_values.view(inputs.shape[0], -1)
        node_means[i] += torch.sum(batch_node_values, dim=0)
        batch_nv_products = torch.matmul(batch_node_values.unsqueeze(2),
                                         batch_node_values.unsqueeze(1))
        node_covs[i] += torch.sum(batch_nv_products, dim=0)
    
    num_samples += inputs.shape[0]

  samples_finish_time = time.time()
  diff = samples_finish_time - start_time
  print('All samples processed in %.3f seconds' % diff, file=output_file)
  
  effective_dims = []
  
  for i in range(len(layers)):
    print('Computing effective dimensionality of layer %d' % layers[i])
    
    node_means[i] /= num_samples
    node_covs[i] /= num_samples
    
    node_covs[i] -= torch.matmul(node_means[i].unsqueeze(1),
                                 node_means[i].unsqueeze(0))
    
    # Take the absolute values of the eigenvalues just to be safe
    eigenvalues = torch.abs(torch.linalg.eigvalsh(node_covs[i]))
    
    if torch.count_nonzero(eigenvalues) == 0:
      effective_dims.append(torch.tensor(0, device=device))
    else:
      effective_dims.append(torch.square(torch.sum(eigenvalues))\
                            / torch.sum(torch.square(eigenvalues)))
  
  dim_finish_time = time.time()
  diff = dim_finish_time - samples_finish_time
  print('All dimensionalities computed in %.3f seconds' % diff,
        file=output_file)
  
  return effective_dims, actual_dims
