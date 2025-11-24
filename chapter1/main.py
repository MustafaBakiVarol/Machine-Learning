import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

HOUSING_PATH = "HandsOnMachineLearning\\chapter1\\data\\housing.csv"

class DataSet:

    def __init__(self, housing_path = HOUSING_PATH):
        self.housing_path= housing_path
        self.data = pd.read_csv(self.housing_path)

    def loading_housing_data(self):
        print(self.data.info())
        return self.data.head(10)

    def column_counts(self):
        df_ocean = self.data["ocean_proximity"].value_counts()
        return df_ocean

    def dataset_describe(self):
        df_desc = self.data.describe()
        return df_desc
    
    def draw_data(self):
        self.data.hist(bins = 50, figsize=(20,15))
        plt.show()
    
    def split_train_test(self, test_ratio):
        shuffled_indices = np.random.permutation(len(self.data))
        test_set_size = int(len(self.data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return self.data.iloc[train_indices], self.data.iloc[test_indices]
    
    def test_set_check(self, identifier, test_ratio):
        return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

    def split_train_test_by_id(self, test_ratio, id_column):
        ids = self.data[id_column]
        in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio))
        return self.data.loc[~in_test_set], self.data.loc[in_test_set]

    
if __name__ == "__main__":
    data = DataSet() 

    # print(data.loading_housing_data())
    # print(data.column_counts())
    # print(data.dataset_describe())
    # print(data.draw_data())
    # train_set, test_set = data.split_train_test(0.2)
    # data.data = data.data.reset_index()    
    # train_set_by_id, test_set_by_id = data.split_train_test_by_id(0.2, "index")
    # data.data["id"] = data.data["longitude"] * 1000 + data.data["latitude"]
    # train_set_id, test_set_id = data.split_train_test_by_id(0.2, "id")
    # print(train_set_id)
    # print(test_set_id)
    # print(train_set_by_id)
    # print(test_set_by_id)
    # print(len(train_set))
    # print(len(test_set))
    train_set, test_set = train_test_split(data.data, test_size= 0.2, random_state=42)
    print(train_set)
    print(test_set)
    data.data["income_cat"] = pd.cut(data.data["median_income"],
                                     bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                                     labels = [1, 2, 3, 4, 5])
    
    print(data.data["income_cat"].hist())
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

    for train_index, test_index in split.split(data.data, data.data["income_cat"]):
        strat_train_set = data.data.loc[train_index]
        strat_test_set = data.data.loc[test_index]
        print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    for set_ in (strat_train_set, strat_test_set):
        print(set_.drop("income_cat", axis = 1, inplace = True))

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    print(scatter_matrix(data.data[attributes], figsize=(12, 8)))
    data.data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)