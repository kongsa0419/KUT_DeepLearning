'''
Questions
 1. csv 파일에 비어있는 데이터(결측치)? : 채우는 과정이 필요함
 2. validation, test의 차이 :
    validation은 훈련한 것을 확인하는 용도. 보통 train 데이터셋의 80%를 훈련용으로 사용하고 나머지 20%를 검증용으로 사용한다.
    반면 test 데이터는 아예 train 데이터와 별개의, 모델이 처음 접하는 데이터를 말한다. test 데이터를 통한 성능평가는 모델의 신뢰성을 높여주고, 이 데이터를 활용하여 모델의 성능을 외부에 보고할 수 있다.
'''
import os
import pandas as pd # data 분석에 필요한 라이브러리
import torch
import csv
from torch.utils.data import Dataset, DataLoader, random_split


class TitanicDataset(Dataset):
  def __init__(self, X, y):  # self. 키워드: 각 instance마다 X,y를 갖음
    self.X = torch.FloatTensor(X)
    self.y = torch.LongTensor(y)

  def __len__(self):
    return len(self.X) # 가장 큰 단위 배열의 길이

  def __getitem__(self, idx):
    feature = self.X[idx]
    target = self.y[idx]
    return {'input': feature, 'target': target}

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.X), self.X.shape, self.y.shape
    )
    return str


class TitanicTestDataset(Dataset):
  def __init__(self, X):
    self.X = torch.FloatTensor(X)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    feature = self.X[idx]
    return {'input': feature}

  def __str__(self):
    str = "Data Size: {0}, Input Shape: {1}".format(
      len(self.X), self.X.shape
    )
    return str


'''데이터 전처리 총괄'''
def get_preprocessed_dataset():
    # CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__)) # 현재위치
    CURRENT_FILE_PATH = os.path.abspath('')  # jupyter notebook 용도

    train_data_path = os.path.join(CURRENT_FILE_PATH, "train.csv") # 파일의 URI
    test_data_path = os.path.join(CURRENT_FILE_PATH, "test.csv")

    train_df = pd.read_csv(train_data_path) # csv 형식의 파일을 읽어들임
    test_df = pd.read_csv(test_data_path)


    #### CHECK: train_df가 'survived' column을 포함하고 있음. 이럴경우 test_df와 concat()연산하면 test_df의 survived가 NULL로 채워지나?
    #### ==> NAN 처리 됌
    all_df = pd.concat([train_df, test_df], sort=False)

    # 데이터 전처리
    all_df = get_preprocessed_dataset_1(all_df)
    all_df = get_preprocessed_dataset_2(all_df)
    all_df = get_preprocessed_dataset_3(all_df)
    all_df = get_preprocessed_dataset_4(all_df)
    all_df = get_preprocessed_dataset_5(all_df)
    all_df = get_preprocessed_dataset_6(all_df)


    # all_df의 Survived(target) 열이 null이 아닌 행만 추출한 다음, Survived열을 삭제함. 그리고 index를 새로 갱신.
    train_X = all_df[~all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)
    train_y = train_df["Survived"] #훈련 에측 정답표

    # test_X에는 "Survived" column이 없어야 해서 있다면 지워줌(예측해야 하니까)
    test_X = all_df[all_df["Survived"].isnull()].drop("Survived", axis=1).reset_index(drop=True)

    dataset = TitanicDataset(train_X.values, train_y.values) # 학습지와 정답표를 쥐어주며 훈련시킬 수 있음
    #print(dataset)
    # random_split(): training(80%), validation(20%)로 데이터레코드에 대해 랜덤하게 나눔
    train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])

    test_dataset = TitanicTestDataset(test_X.values)
    #print(test_dataset)

    return train_dataset, validation_dataset, test_dataset







''' 궁금해서 혼자 찍어본 용도인데요, 지저분하여 지울까하다가 주석은 다다익선이라고 하셔서 남겨뒀습니다.

    print("##########" * 2)
    a = all_df[["Pclass", "Fare"]].groupby("Pclass")
    print(a)
    print()
    b = a.mean()
    print(b)
    print()
    c = b.reset_index()
    print(c)
    print()
    print(Fare_mean.columns)
    print()
    print("##########" * 2)
    exit(0)
'''


# Pclass별 Fare 평균값을 사용하여 Fare 결측치 메우기
def get_preprocessed_dataset_1(all_df):

    # Pclass별로 Fare의 평균값을 구한다음, reset_index()로 새로운 데이터프레임을 반환
    Fare_mean = all_df[["Pclass", "Fare"]].groupby("Pclass").mean().reset_index()

    # 데이터프레임의 컬럼이름을 변경(Fare_mean)
    Fare_mean.columns = ["Pclass", "Fare_mean"]

    # Pclass열을 기준으로 all_df와 Fare_mean 데이터프레임을 병합. 단, left에 해당하는 all_df의 Pclass를 기준으로 Fare_mean df값을 왼쪽에 병합
    all_df = pd.merge(all_df, Fare_mean, on="Pclass", how="left")
    # Fare은 그대로 남고 Fare_mean 컬럼 새롭게 생김?
    
    # 결측치가 있는 Fare 값을 해당 행의 Fare_mean 값으로 대체
    all_df.loc[(all_df["Fare"].isnull()), "Fare"] = all_df["Fare_mean"] 

    return all_df




def get_preprocessed_dataset_2(all_df):
    # name을 세 개의 컬럼으로 분리하여 다시 all_df에 합침
    name_df = all_df["Name"].str.split("[,.]", n=2, expand=True) # Name값을 최대 2번까지 ','과 '.'을 구분자로 하여 분리
    name_df.columns = ["family_name", "honorific", "name"] # 각 컬럼명 지정
    name_df["family_name"] = name_df["family_name"].str.strip()
    name_df["honorific"] = name_df["honorific"].str.strip()
    name_df["name"] = name_df["name"].str.strip()
    all_df = pd.concat([all_df, name_df], axis=1) # name_df를 all_df의 column 방향으로 이어붙임 (axis=1)

    return all_df


def get_preprocessed_dataset_3(all_df):
    # honorific별 Age 평균값을 사용하여 Age 결측치 메우기

    # 호칭별로 나이값의 중앙값을 구하여 이를 정수화(반올림)해준 뒤 새로운 데이터프레임으로 반환
    honorific_age_mean = all_df[["honorific", "Age"]].groupby("honorific").median().round().reset_index()

    # 그 데이터프레임(df)의 컬럼명을 다음과 같이 설정
    honorific_age_mean.columns = ["honorific", "honorific_age_mean", ]
    # honorific 열을 기준으로 두 df를 합친다. 이 때 all_df의 honorific를 기준으로하고, honorific_age_mean df의 값을 왼쪽에 병합함.
    all_df = pd.merge(all_df, honorific_age_mean, on="honorific", how="left")
    # all_df의 Age열의 결측치를 확인하고, 그러한 위치에는 honorific_age_mean값으로 대체함.
    all_df.loc[(all_df["Age"].isnull()), "Age"] = all_df["honorific_age_mean"]
    # honorific_age_mean 열을 all_df에서 삭제
    all_df = all_df.drop(["honorific_age_mean"], axis=1)

    return all_df


def get_preprocessed_dataset_4(all_df):
    # 가족수(family_num) 컬럼 새롭게 추가
    all_df["family_num"] = all_df["Parch"] + all_df["SibSp"]

    # 혼자탑승(alone) 컬럼 새롭게 추가
    all_df.loc[all_df["family_num"] == 0, "alone"] = 1
    all_df["alone"].fillna(0, inplace=True)

    # 학습에 불필요한 컬럼 제거
    all_df = all_df.drop(["PassengerId", "Name", "family_name", "name", "Ticket", "Cabin"], axis=1)

    return all_df


def get_preprocessed_dataset_5(all_df):
    # honorific 값 개수 줄이기
    all_df.loc[
    ~(
            (all_df["honorific"] == "Mr") |
            (all_df["honorific"] == "Miss") |
            (all_df["honorific"] == "Mrs") |
            (all_df["honorific"] == "Master")
    ),
    "honorific"
    ] = "other"
    all_df["Embarked"].fillna("missing", inplace=True) # Embarked 컬럼의 결측치를 missing으로 채운다.

    return all_df


def get_preprocessed_dataset_6(all_df):
    # 카테고리 변수를 LabelEncoder를 사용하여 수치값으로 변경하기
    category_features = all_df.columns[all_df.dtypes == "object"] # dtypes=="object" : 문자열인 열들을 선택
    from sklearn.preprocessing import LabelEncoder # 카테고리 변수를 숫자로 인코딩 하는데에 사용
    for category_feature in category_features:
        le = LabelEncoder()
        if all_df[category_feature].dtypes == "object":
          # Sex, Embarked, honorific
          le = le.fit(all_df[category_feature]) # labelEncoder를 category 변수에 맞게 훈련시키는 단계
          all_df[category_feature] = le.transform(all_df[category_feature]) # 카테고리 변수를 숫자로 변환하고 원래 데이터프레임에 저장

    return all_df


# Neural Network
from torch import nn
class MyModel(nn.Module): # nn.Module: base class for all PyTorch models
  def __init__(self, n_input, n_output): # number of input & output features
    super().__init__()


    # fully connected layers with ReLU activation functions.
    # This network consists of three linear layers.
    self.model = nn.Sequential(
      nn.Linear(n_input, 30),
      nn.ReLU(),
      nn.Linear(30, 30),
      nn.ReLU(),
      nn.Linear(30, n_output),
    )

  # forward 메서드는 input 'x'를 생성자에 정의된 sequential 모델에 넘겨주고, 출력값을 반환해줌.
  def forward(self, x):
    x = self.model(x)
    return x


def test(test_data_loader): # 매개변수 test_data_loader : 전형적인 PyTorch DataLoader object (test_data를 담고있는)
  print("[TEST]")
  batch = next(iter(test_data_loader)) # 테스트 데이터로더의 한 배치를 준비
  print("{0}".format(batch['input'].shape)) # torch.Size([418, 11])
  my_model = MyModel(n_input=11, n_output=2) # NN의 입력 피쳐로 11개를, 출력(예측값)으로 2개를 설정
  output_batch = my_model(batch['input']) # input batch에 따른 예측을 하는데에 사용. 모델의 예측값을 포함하는 torch.Size([418, 2])
  prediction_batch = torch.argmax(output_batch, dim=1) # 2번째 차원에서(행,dim=1) 가장 예측률이 높은 index를 고름 # torch.Size([418])

  # print("$" * 20)
  # print(output_batch)
  # print(output_batch.shape) # torch.Size([418, 2])
  # print(output_batch[prediction_batch])
  # print(prediction_batch)
  # print(prediction_batch.shape) # 418
  # print("$"*20)

  # CSV 파일로 데이터 저장
  with open('submission.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['PassengerId', 'Survived'])  # CSV 파일의 헤더
      # csv 파일의 데이터
      for idx, prediction in enumerate(prediction_batch, start=892):
          writer.writerow([idx, prediction.item()])

  # 출력으로 다시 확인
  for idx, prediction in enumerate(prediction_batch, start=892):
      print(idx, prediction.item())  # 될때 있고 안될때 있네?








if __name__ == "__main__":
  #데이터 전처리
  train_dataset, validation_dataset, test_dataset = get_preprocessed_dataset()

  print("train_dataset: {0}, validation_dataset.shape: {1}, test_dataset: {2}".format(
    len(train_dataset), len(validation_dataset), len(test_dataset)
  ))
  print("#" * 50, 1)

  # 훈련용 데이터셋을 읽어봄 (확인용)
  for idx, sample in enumerate(train_dataset):
    print("{0} - {1}: {2}".format(idx, sample['input'], sample['target']))

  print("#" * 50, 2)

  train_data_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
  validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=16, shuffle=True)
  test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

# 데이터로더를 이용한 확인
  print("[TRAIN]")
  for idx, batch in enumerate(train_data_loader):
    print("{0} - {1}: {2}".format(idx, batch['input'].shape, batch['target'].shape))

  print("[VALIDATION]")
  for idx, batch in enumerate(validation_data_loader):
    print("{0} - {1}: {2}".format(idx, batch['input'].shape, batch['target'].shape))

  print("#" * 50, 3)

  test(test_data_loader)
