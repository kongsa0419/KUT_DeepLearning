import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from datetime import datetime
import os
import wandb

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)
print("BASE_PATH", BASE_PATH)
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(BASE_PATH)

from c_trainer import ClassificationTrainer
from _01_code._99_common_utils.utils import is_linux, get_num_cpu_cores


def get_ready():
  if not os.path.isdir(os.path.join(CURRENT_FILE_PATH, "checkpoints")):
    os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))


def get_data():
  data_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "j_cifar10")

  cifar10_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
  cifar10_train, cifar10_validation = random_split(cifar10_train, [45_000, 5_000])

  print("Num Train Samples: ", len(cifar10_train))
  print("Num Validation Samples: ", len(cifar10_validation))

  num_data_loading_workers = get_num_cpu_cores() if is_linux() else 0
  print("Number of Data Loading Workers:", num_data_loading_workers)

  train_data_loader = DataLoader(
    dataset=cifar10_train, batch_size=wandb.config.batch_size, shuffle=True,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  validation_data_loader = DataLoader(
    dataset=cifar10_validation, batch_size=wandb.config.batch_size,
    pin_memory=True, num_workers=num_data_loading_workers
  )

  cifar10_transforms = nn.Sequential(
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)),
    nn.Flatten(),
  )

  return train_data_loader, validation_data_loader, cifar10_transforms


def get_model():
  class MyModel(nn.Module):
    def __init__(self, n_input, n_output):
      super().__init__()

      self.model = nn.Sequential(
        nn.Linear(n_input, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, n_output),
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 3 * 32 * 32 = 3072
  my_model = MyModel(n_input=3_072, n_output=10)

  return my_model


def main(args):
  get_ready()

  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'learning_rate': args.learning_rate
  }

  wandb.init(
    mode="online" if args.use_wandb else "disabled",
    project="fcn_cifar10",
    notes="cifar10 experiment",
    tags=["fcc", "cifar10"],
    name=run_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, cifar10_transforms = get_data()
  model = get_model()
  model.to(device)
  wandb.watch(model)

  optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate)

  classification_trainer = ClassificationTrainer(
    "cifar10", model, optimizer, train_data_loader, validation_data_loader, cifar10_transforms,
    run_time_str, wandb, device
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "-w", "--use_wandb", type=bool, default=False, help="True or False"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=10_000,
    help="Number of training epochs (int, default: 10,000)"
  )

  parser.add_argument(
    "-b", "--batch_size", type=int, default=2_048,
    help="Batch size (int, default: 2,048)"
  )

  parser.add_argument(
    "-r", "--learning_rate", type=float, default=1e-3,
    help="Learning rate (float, default: 1e-3)"
  )

  parser.add_argument(
    "-v", "--validation_intervals", type=int, default=10,
    help="Number of training epochs between validations (int, default: 10)"
  )

  args = parser.parse_args()

  main(args)