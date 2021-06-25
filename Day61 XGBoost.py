# 本文用到的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve


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


xgboost.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    silent=True,
    objective="binary:logistic",
    booster="gbtree",
    n_jobs=1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=1,
    colsample_bytree=1,
    colsample_bylevel=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,
    missing=None,
    **kwargs
)

# 训练模型
df = getDataSet()
X, y = df[df.columns[:-1]], df["good"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
sklearn_model_new = XGBClassifier(
    n_estimators=2,
    max_depth=5,
    learning_rate=0.1,
    verbosity=1,
    objective="binary:logistic",
    random_state=1,
)
sklearn_model_new.fit(X_train, y_train)
model.fit(X_train, y_train)


def ceate_feature_map(features):
    outfile = open("xgb.fmap", "w")
    i = 0
    for feat in features:
        outfile.write("{0}\t{1}\tq\n".format(i, feat))
        i = i + 1
    outfile.close()


ceate_feature_map(df.columns)

plot_tree(fmap="xgb.fmap")
plot_tree(sklearn_model_new, fmap="xgb.fmap", num_trees=0)
fig = plt.gcf()
fig.set_size_inches(150, 100)
plt.show()


plot_tree(sklearn_model_new, fmap="xgb.fmap", num_trees=1)
fig = plt.gcf()
fig.set_size_inches(150, 100)
plt.show()


plot_importance(sklearn_model_new)


gsCv = GridSearchCV(
    sklearn_model_new,
    {
        "max_depth": [4, 5, 6],
        "n_estimators": [5, 10, 20],
        "learning_rate ": [0.05, 0.1, 0.3, 0.5, 0.7],
        "min_child_weight": [0.1, 0.2, 0.5, 1],
    },
)
gsCv.fit(X_train, y_train)
print(gsCv.best_params_)
