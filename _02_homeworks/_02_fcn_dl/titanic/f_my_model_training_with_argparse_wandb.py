import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from datetime import datetime
import wandb
import argparse

from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl

import sys
sys.path.append(BASE_PATH)

# from _01_code._03_real_world_data_to_tensors.k_california_housing_dataset_dataloader \
#   import CaliforniaHousingDataset

# 타이타닉 데이터에 맞게 데이터 import 수정
from _02_homeworks._02_fcn_dl.titanic.titanic_dataset import get_preprocessed_dataset

def get_data():
  # 타이타닉 데이터셋을 가져옴
  train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()
  train_data_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))

  return train_data_loader, validation_data_loader


class MyModel(nn.Module):
  def __init__(self, n_input, n_output):
    super().__init__()

# 바람직한 코드 관습
    self.model = nn.Sequential(
      nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
      nn.ReLU(),
      nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
      nn.ReLU(),
      nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
    )

  def forward(self, x):
    x = self.model(x)
    return x


def get_model_and_optimizer():
  my_model = MyModel(n_input=11, n_output=1) #torch shape을 맞춰줌
  optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate) # SGD: 확률적 경사하강법

  return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader):
  n_epochs = wandb.config.epochs #설정해둔 에포크 수
  loss_fn = nn.MSELoss()  # Use a built-in loss function
  next_print_epoch = 100 # 조절 가능


  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    num_trains = 0
    for train_batch in train_data_loader:
      # print("#"*10)
      # print(train_batch['input']) #인풋데이터
      # print(model(train_batch['input'])) #모델의 결과값
      # print(train_batch['target']) #타겟데이터
      # print(model(train_batch['target'])) #타겟
      # print("#" * 10)

      output_train = model(train_batch['input'])
      train_batch['target'] = torch.unsqueeze(train_batch['target'], dim=1) #shape을 맞추기 위해 unsqueeze()
      loss = loss_fn(output_train, train_batch['target'].float())

      # log 'train data' per epoch here
      # wandb에 log를 남기기 위함
      wandb.log({
        "[TEST] sample:": num_trains,
        "loss:": loss.item()
      })
      loss_train += loss.item()
      num_trains += 1

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_validation = 0.0
    num_validations = 0
    with torch.no_grad():
      for validation_batch in validation_data_loader:
        output_validation = model(validation_batch['input'])
        ##### TODO check
        validation_batch['target'] = torch.unsqueeze(validation_batch['target'], dim=1)
        loss = loss_fn(output_validation, validation_batch['target'].float())
        # log 'validation data' per epoch here
        # wandb에 log를 남기기 위함
        wandb.log({
          "[VALIDATION] sample:" : num_validations,
          "loss:" : loss.item()
        })
        loss_validation += loss.item()
        num_validations += 1

    wandb.log({
      "Epoch": epoch,
      "Training loss": loss_train / num_trains,
      "Validation loss": loss_validation / num_validations
    })

    if epoch >= next_print_epoch:
      print(
        f"Epoch {epoch}, "
        f"Training loss {loss_train / num_trains:.4f}, "
        f"Validation loss {loss_validation / num_validations:.4f}"
      )
      next_print_epoch += 100


def main(args):
  current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': 1e-3,
    'n_hidden_unit_list': [30, 30],
  }

  wandb.init(
    mode="online" if args.wandb else "disabled",
    project="my_model_training",
    notes="My first wandb experiment",
    tags=["my_model", "california_housing"],
    name=current_time_str,
    config=config
  )
  print(args)
  print(wandb.config)

  train_data_loader, validation_data_loader = get_data()

  linear_model, optimizer = get_model_and_optimizer()

  wandb.watch(linear_model)

  print("#" * 50, 1)

  training_loop(
    model=linear_model,
    optimizer=optimizer,
    train_data_loader=train_data_loader,
    validation_data_loader=validation_data_loader
  )
  wandb.finish()


# https://docs.wandb.ai/guides/track/config
if __name__ == "__main__":

  parser = argparse.ArgumentParser()

# default=True로 바꿔줌 (log 보기 위함)
  parser.add_argument(
    "--wandb", action=argparse.BooleanOptionalAction, default=True, help="True or False"
  )
# 배치사이즈와 에포크를 조절해줌
  parser.add_argument(
    "-b", "--batch_size", type=int, default=16, help="Batch size (int, default: 16)"
  )

  parser.add_argument(
    "-e", "--epochs", type=int, default=400, help="Number of training epochs (int, default:100)"
  )

  args = parser.parse_args()

  main(args)

