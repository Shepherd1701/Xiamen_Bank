# coding:utf-8

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
np.random.seed(7)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


class BaseLineModel(object):
    def __init__(self, *, path):
        self.__path = path
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label, self.__test_index = [None for _ in range(2)]

        self.__xgb = None

    def read_data(self):
        # feature
        self.__train_feature = pd.read_csv(os.path.join(self.__path, "train.csv"), na_values=[-999])
        self.__test_feature = pd.read_csv(os.path.join(self.__path, "test.csv"), na_values=[-999])
        # label
        self.__train_label = pd.read_csv(os.path.join(self.__path, "train_label.csv"))

    def prepare_data(self):
        self.__train_label = self.__train_label["target"]
        self.__test_index = self.__test_feature[["id"]]

        self.__train_feature = self.__train_feature.drop(["id"], axis=1)
        self.__test_feature = self.__test_feature.drop(["id"], axis=1)

        self.__train_feature = self.__train_feature.drop(
            ["certId", "dist", "residentAddr", "bankCard", "certValidBegin", "certValidStop"], axis=1)
        self.__test_feature = self.__test_feature.drop(
            ["certId", "dist", "residentAddr", "bankCard", "certValidBegin", "certValidStop"], axis=1)

        self.__train_feature["age"] = np.clip(self.__train_feature["age"], a_min=18, a_max=60)
        self.__test_feature["age"] = np.clip(self.__test_feature["age"], a_min=18, a_max=60)

    def model_fit_predict(self):
        self.__xgb = XGBClassifier(n_jobs=-1, seed=7)
        self.__xgb.fit(self.__train_feature, self.__train_label)
        self.__test_index["target"] = self.__xgb.predict_proba(self.__test_feature)[:, 1]

    def data_write(self):
        self.__test_index.to_csv(os.path.join(self.__path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    blm = BaseLineModel(path="D:\\Kaggle\\Xiamen_Bank")
    blm.read_data()
    blm.prepare_data()
    blm.model_fit_predict()
    blm.data_write()




