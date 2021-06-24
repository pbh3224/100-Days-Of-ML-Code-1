import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("creditcard.csv")
data = df.iloc[:, 1:31]

X = data.loc[:, data.columns != "Class"]
y = data.loc[:, data.columns == "Class"]

number_records_fraud = len(data[data.Class == 1])  # class=1的样本函数
fraud_indices = np.array(data[data.Class == 1].index)  # 样本等于1的索引值
normal_indices = data[data.Class == 0].index  # 样本等于0的索引值
random_normal_indices = np.random.choice(
    normal_indices, number_records_fraud, replace=False
)
random_normal_indices = np.array(random_normal_indices)
under_sample_indices = np.concatenate(
    [fraud_indices, random_normal_indices]
)  # Appending the 2 indices
under_sample_data = data.iloc[under_sample_indices, :]  # Under sample dataset
X_undersample = under_sample_data.loc[:, under_sample_data.columns != "Class"]
y_undersample = under_sample_data.loc[:, under_sample_data.columns == "Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X_undersample, y_undersample, test_size=0.3, random_state=0
)


rf0 = RandomForestClassifier(oob_score=True, random_state=666)
rf0.fit(X_train, y_train)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X_test)[:, 1]
print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob))


param_test1 = {"n_estimators": range(10, 101, 10)}
gsearch1 = GridSearchCV(
    estimator=RandomForestClassifier(oob_score=True, random_state=666, n_jobs=2),
    param_grid=param_test1,
    scoring="roc_auc",
    cv=5,
)
gsearch1.fit(X_train, y_train)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


param_test2 = {"max_depth": range(2, 12, 2)}
gsearch2 = GridSearchCV(
    estimator=RandomForestClassifier(
        n_estimators=50, oob_score=True, random_state=666, n_jobs=2
    ),
    param_grid=param_test2,
    scoring="roc_auc",
    cv=5,
)
gsearch2.fit(X_train, y_train)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


param_test2 = {"min_samples_split": range(2, 8, 1)}
gsearch2 = GridSearchCV(
    estimator=RandomForestClassifier(
        n_estimators=50, max_depth=6, oob_score=True, random_state=666, n_jobs=2
    ),
    param_grid=param_test2,
    scoring="roc_auc",
    cv=5,
)
gsearch2.fit(X_train, y_train)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


rf1 = RandomForestClassifier(
    n_estimators=50,
    max_depth=6,
    min_samples_split=5,
    oob_score=True,
    random_state=666,
    n_jobs=2,
)
rf1.fit(X_train, y_train)
print(rf1.oob_score_)
y_predprob1 = rf1.predict_proba(X_test)[:, 1]
print("AUC Score (Train): %f" % roc_auc_score(y_test, y_predprob1))