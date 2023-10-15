import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class CaliforniaHousingDataset(Dataset):
  def __init__(self):
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    data_mean = np.mean(housing.data, axis=0) # 모든 feature에 대한 mean (중앙값)
    # print(data_mean)
    data_var = np.var(housing.data, axis=0) # 모든 feature에 대한 var(분산)
    # print(data_var)
    # print("*"*10); print(housing.target); print("*"*10);
    self.data = torch.tensor((housing.data - data_mean) / np.sqrt(data_var), dtype=torch.float32)
    self.target = torch.tensor(housing.target, dtype=torch.float32).unsqueeze(dim=-1) # 마지막에 축에 차원을 하나 추가 [8] => [(8,1)]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sample_data = self.data[idx]
    sample_target = self.target[idx]
    return {'input': sample_data, 'target': sample_target}

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.data), self.data.shape, self.target.shape
    )
    return str


if __name__ == "__main__":
  california_housing_dataset = CaliforniaHousingDataset()

  print(california_housing_dataset)

  print("#" * 50, 1)

  for idx, sample in enumerate(california_housing_dataset):
    print("{0} - {1}: {2}".format(idx, sample['input'].shape, sample['target'].shape))

  train_dataset, validation_dataset, test_dataset = random_split(california_housing_dataset, [0.7, 0.2, 0.1])

  print("#" * 50, 2)

  print(len(train_dataset), len(validation_dataset), len(test_dataset))

  print("#" * 50, 3)

  train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
  )

  for idx, batch in enumerate(train_data_loader):
    print("{0} - {1}: {2}".format(idx, batch['input'].shape, batch['target'].shape))
