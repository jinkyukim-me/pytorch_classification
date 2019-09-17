# Pytorch 라이브러리 임포트
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# scikit-learn 라이브러리 임포트
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
# Pandas 라이브러리 임포트
import pandas as pd
# 와인 데이터 읽어 들이기
wine = load_wine()
wine
#데이터프레임에 담긴 설명변수 출력
pd.DataFrame(wine.data, columns=wine.feature_names)
