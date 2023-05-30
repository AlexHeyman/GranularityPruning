import sys
from os import path
import time
import torch
from utilities import compute_accuracy


class TrainingSystem:

  def __init__(self, name, device, model, optimizer, learning_rate_fn,
               update_lr_every, print_every, iterations_to_save,
               default_checkpoint_dir, trainloader, validloader=None,
               print_file=sys.stdout):
    self.name = name
    self.device = device
    self.model = model
    self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    self.optimizer = optimizer
    self.learning_rate_fn = learning_rate_fn
    self.update_lr_every = update_lr_every
    self.print_every = print_every
    self.iterations_to_save = list(iterations_to_save)
    self.its_index = 0
    self.default_checkpoint_dir = default_checkpoint_dir
    self.trainloader = trainloader
    self.validloader = validloader
    self.print_file = print_file
    self.iterations_ran = 0
    self.running_loss = 0
    self.training_time = 0
    self.update_lr()

  def update_lr(self):
    new_lr = self.learning_rate_fn(self.iterations_ran)
    for g in self.optimizer.param_groups:
      g['lr'] = new_lr

  def save_checkpoint(self, directory=None, filename=None):
    if directory is None:
      directory = self.default_checkpoint_dir
    if filename is None:
      filename = '%s_%d.pt' % (self.name, self.iterations_ran)
    torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iterations_ran': self.iterations_ran,
            'running_loss': self.running_loss,
            'training_time': self.training_time
            }, path.join(directory, filename))
  
  def load_checkpoint(self, directory=None, filename=None, load_model=True,
                      load_other=True, masks_special=None):
    # If masks_special == 'exclude', do not load masks from checkpoint
    # If masks_special == 'only', load masks but not other model parameters
    # Both assume this TrainingSystem's model is a MaskedModule and only apply
    # if load_model == True
    
    if directory is None:
      directory = self.default_checkpoint_dir
    if filename is None:
      filename = '%s_%d.pt' % (self.name, self.iterations_ran)
    
    checkpoint = torch.load(path.join(directory, filename))

    if load_model:
      model_state_dict = checkpoint['model_state_dict']
      if masks_special == 'exclude':
        mask_keys = [key for key in model_state_dict if key.startswith('mask')]
        
        for key in mask_keys:
          del model_state_dict[key]
        
        self.model.load_state_dict(model_state_dict, strict=False)
      elif masks_special == 'only':
        for i in range(len(self.model.masks_flat)):
          self.model.masks_flat[i].data.copy_(
            model_state_dict['mask%d' % i].data)
      else:
        self.model.load_state_dict(model_state_dict, strict=False)

    if load_other:
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      self.iterations_ran = checkpoint['iterations_ran']
      self.running_loss = checkpoint['running_loss']
      self.training_time = checkpoint['training_time']

      self.its_index = 0
      while self.its_index < len(self.iterations_to_save)\
      and self.iterations_to_save[self.its_index] <= self.iterations_ran:
        self.its_index += 1

      self.update_lr()
  
  def train(self, max_iteration, save_directory=None):
    self.model.train()
    start_time = time.time()

    while self.iterations_ran < max_iteration:
        for data in self.trainloader:
            if self.its_index < len(self.iterations_to_save)\
            and self.iterations_ran == self.iterations_to_save[self.its_index]:
              self.training_time += (time.time() - start_time)

              print('Training time up to iteration %d: %.3f seconds'\
                    % (self.iterations_ran, self.training_time),
                    file=self.print_file)
              self.save_checkpoint(directory=save_directory)
              self.its_index += 1
              
              start_time = time.time()
            
            if self.iterations_ran % self.update_lr_every == 0:
              self.update_lr()
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs.to(self.device))
            loss = self.criterion(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()

            # print statistics
            self.running_loss += loss.item()
            if self.iterations_ran % self.print_every == (self.print_every - 1):
                self.training_time += (time.time() - start_time)

                print('Iteration %d train loss: %.3f' %
                      (self.iterations_ran + 1,
                       self.running_loss / self.print_every),
                      file=self.print_file)
                self.running_loss = 0
                
                if self.validloader is not None:
                    self.model.eval()
                    print('Validation accuracy: %f' % compute_accuracy(
                      self.device, self.model, self.validloader),
                          file=self.print_file)
                    self.model.train()

                start_time = time.time()

            self.iterations_ran += 1
            if self.iterations_ran >= max_iteration:
                break

    self.training_time += (time.time() - start_time)

    print('Training time up to iteration %d: %.3f seconds'\
          % (self.iterations_ran, self.training_time),
          file=self.print_file)
    if self.its_index < len(self.iterations_to_save)\
    and self.iterations_ran == self.iterations_to_save[self.its_index]:
      self.save_checkpoint(directory=save_directory)
      self.its_index += 1
