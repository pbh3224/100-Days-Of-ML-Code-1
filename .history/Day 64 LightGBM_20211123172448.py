# 本文用到的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import lightgbm as lgb  # sklearn接口形式
from lightgbm import LGBMClassifier


def getDataSet():
    dataSet = [
        ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460, 1],
        ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.774, 0.376, 1],
        ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.634, 0.264, 1],
        ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.608, 0.318, 1],
        ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.556, 0.215, 1],
        ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.403, 0.237, 1],
        ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 0.481, 0.149, 1],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 0.437, 0.211, 1],
        ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0.666, 0.091, 0],
        ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0.243, 0.267, 0],
        ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0.245, 0.057, 0],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0.343, 0.099, 0],
        ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0.639, 0.161, 0],
        ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657, 0.198, 0],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.360, 0.370, 0],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593, 0.042, 0],
        ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0.719, 0.103, 0],
    ]

    features = [
        "color",
        "root",
        "knocks",
        "texture",
        "navel",
        "touch",
        "density",
        "sugar",
        "good",
    ]
    dataSet = np.array(dataSet)
    df = pd.DataFrame(dataSet, columns=features)
    for feature in features[0:6]:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    df.iloc[:, 6:8] = df.iloc[:, 6:8].astype(float)
    df["good"] = df["good"].astype(int)
    return df


# train
df = getDataSet()
X, y = df[df.columns[:-1]], df["good"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

gbm = LGBMClassifier(
    num_leaves=5,
    max_depth=2,
    learning_rate=0.05,
    min_data_in_leaf=3,
    n_estimators=5,
    max_bin=5,
    min_data_in_bin=2,
    subsample_for_bin=17,
)
# gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5)
gbm.fit(X_train, y_train)

gbm.save_model("model.txt")
