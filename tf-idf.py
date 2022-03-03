import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import os
print(os.listdir('input'))
data=pd.read_csv('input/IMDB Dataset.csv')
print(data.shape)
print(data.head(10))