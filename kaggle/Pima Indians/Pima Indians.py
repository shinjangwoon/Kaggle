# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# %%
df = pd.read_csv('C:/Users/wkddn/OneDrive/문서/GitHub/Kaggle/kaggle/Pima Indians/diabetes.csv')


# %%
df.shape


# %%
df.head()


# %%
split_count = int(df.shape[0]*0.8)
split_count


# %%
train = df[:split_count].copy()
train.shape


# %%
test = df[split_count:].copy()
test.shape


# %%
feature_names = train.columns[:-1].tolist()
feature_names


# %%
label_name = train.columns[-1]
label_name


# %%
X_train = train[feature_names]
print(X_train.shape)
X_train.head()


# %%
y_train = train[label_name]
print(y_train.shape)
y_train.head()


# %%
X_test = test[feature_names]
print(X_test.shape)
X_test.head()


# %%
y_test = test[label_name]
print(y_test.shape)
y_test.head()


# %%
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model


# %%
model.fit(X_train, y_train)


# %%
y_predict = model.predict(X_test)
y_predict[:5]


# %%
from sklearn.tree import plot_tree

plot_tree(model, feature_names=feature_names)


# %%
plt.figure(figsize=(20,20))
tree = plot_tree(model, feature_names=feature_names,
                filled=True,
                fontsize=10)


# %%
import graphviz
from sklearn.tree import export_graphviz

dot_tree = export_graphviz(model, feature_names= feature_names,filled=True)
graphviz.Source(dot_tree)


# %%
model.feature_importances_


# %%
sns.barplot(x=model.feature_importances_, y=feature_names)


# %%
diff_count = abs(y_test-y_predict).sum()
diff_count


# %%
# 전체에서 29%를 틀렸음을 알 수 있음
abs(y_test-y_predict).sum() / len(y_test) * 100


# %%
# 예측도
(len(y_test)-diff_count) / len(y_test) * 100


# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)


# %%
model.score(X_test, y_test) * 100


