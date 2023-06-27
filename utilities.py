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
  
  def __init__(self):
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


def compute_accuracy(device, model, dataloader):
  correct = 0
  total = 0

  with torch.no_grad():
      for data in dataloader:
          inputs, labels = data
          outputs = model(inputs.to(device))
          # Find the class with the highest energy for each sample in the batch
          _, predicted = torch.max(outputs, dim=1)
          total += labels.size(0)
          correct += (predicted == labels.to(device)).sum().item()
  
  return correct / total
