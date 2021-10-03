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
df


# %%
df_null = df.isnull()
df_null.head()


# %%
df_null.sum()


# %%
df.describe()


# %%
feature_columns = df.columns[0:-1].tolist()
feature_columns


# %%
cols = feature_columns[1:]
cols


# %%
df_null = df[cols].replace(0, np.nan)
df_null = df_null.isnull()
df_null.sum()


# %%
# 결측치 수치를 시각화
df_null.sum().plot.barh()


# %%
df_null.mean() * 100


# %%
plt.figure(figsize=(15, 4))
sns.heatmap(df_null, cmap="Greys_r")


# %%
df['Outcome']


# %%
# 1 - 발병하는 케이스, 0 - 발병하지 않는 케이스
df['Outcome'].value_counts()


# %%
df['Outcome'].value_counts(normalize=True)


# %%
df.groupby(['Pregnancies'])['Outcome'].mean()


# %%
df_po = df.groupby(['Pregnancies'])['Outcome'].agg(['mean','count']).reset_index()
df_po


# %%
df_po.plot()


# %%
sns.countplot(data=df, x='Outcome')


# %%
sns.countplot(data=df, x='Pregnancies', hue='Outcome')


# %%
# 6번 이상 임신하면 True, 미만은 False
df['Pregnancies_high'] = df['Pregnancies'] > 6
df[['Pregnancies', 'Pregnancies_high']].head()


# %%
sns.countplot(data=df, x='Pregnancies_high')


# %%
sns.countplot(data=df, x='Pregnancies_high', hue='Outcome')


# %%
sns.barplot(data=df, x='Outcome', y='BMI')


# %%
sns.barplot(data=df, x='Outcome', y='Insulin')


# %%
sns.barplot(data=df, x='Pregnancies', y='Outcome')


# %%
# 당뇨병 발병여부에 따라 글루코스 수치 차이가 있음을 알 수 있음
sns.barplot(data=df, x='Pregnancies', y='Glucose', hue='Outcome')


# %%
# 당뇨병 발병한 사람들이 BMI 수치가 더 높음
sns.barplot(data=df, x='Pregnancies', y='BMI', hue='Outcome')


# %%
# 당뇨병 발병한 사람들이 인슐린 수치가 더 높음
# 평균 값이 y축으로 나타나고, 신뢰구간의 차이가 꽤 남
sns.barplot(data=df, x='Pregnancies', y='Insulin', hue='Outcome')


# %%
sns.barplot(data=df[df['Insulin']>0], x='Pregnancies', y='Insulin', hue='Outcome')


# %%
plt.figure(figsize=(15,4))
sns.violinplot(data=df[df['Insulin']>0], x='Pregnancies', y='Insulin', hue='Outcome', split=True)


# %%
# 임신횟수는 당뇨병 발병률과 관련이 있는 것으로 보임
plt.figure(figsize=(15,4))
sns.swarmplot(data=df[df['Insulin']>0], x='Pregnancies', y='Insulin', hue='Outcome')


# %%
sns.displot(df['Pregnancies'])


# %%
df_0 = df[df['Outcome']==0]
df_1 = df[df['Outcome']==1]
df_0.shape, df_1.shape


# %%
sns.displot(df_0['Pregnancies'])
sns.displot(df_1['Pregnancies'])


# %%
df['Pregnancies_high'] = df['Pregnancies_high'].astype(int)
h = df.hist(figsize=(15, 15), bins=20)


# %%
col_num = df.columns.shape
col_num


# %%
cols = df.columns[:-1].tolist()
cols


# %%
import warnings
warnings.filterwarnings(action='ignore')


# %%
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
for i, col_name in enumerate(cols):
    row = i // 3
    col = i % 3
    print(i, col_name, row, col)
    sns.distplot(df[col_name], ax=axes[row][col])


# %%
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))

for i, col_name in enumerate(cols[:-1]):
    row = i // 2
    col = i % 2
    sns.distplot(df_0[col_name], ax=axes[row][col])
    sns.distplot(df_1[col_name], ax=axes[row][col])


# %%
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))

for i, col_name in enumerate(cols[:-1]):
    row = i // 2
    col = i % 2
    sns.violinplot(data=df, x="Outcome", y=col_name, ax=axes[row][col])


# %%
sns.regplot(data=df, x='Glucose', y='Insulin')


# %%
sns.lmplot(data=df, x='Glucose', y='Insulin', hue='Outcome')


# %%
sns.lmplot(data=df[df['Insulin']>0], x='Glucose', y='Insulin', hue='Outcome')


# %%
sns.pairplot(df)


# %%
g = sns.PairGrid(df, hue='Outcome')
g.map(plt.scatter)


# %%
df_corr = df.corr()
df_corr.style.background_gradient()


# %%
sns.heatmap(df_corr, vmax=1, vmin=-1)


# %%
# 글루코스와 Outcome 값이 상관관계가 높은 것으로 나타나고
# 인슐린과 Outcome은 상관관계가 높지 않은 것으로 보임
plt.figure(figsize=(15,8))
sns.heatmap(df_corr, annot=True, vmax=1, vmin=-1, cmap='coolwarm')


# %%
dF_matrix = df.iloc[:,:-2].replace(0,np.nan)
dF_matrix['Outcome'] = df['Outcome']
dF_matrix.head()


# %%
df_corr['Outcome']


# %%
# 회귀선이 1이 가까울수록 상관계수가 높다고 볼 수 있음
# Insulin의 이상치를 정리해도 더욱 상관관계가 높아질 것
sns.regplot(data=dF_matrix, x='Insulin', y='Glucose')


# %%
# 연령이 증가함에 따라 임신 횟수도 증가함
sns.lmplot(data=df, x='Age', y='Pregnancies', hue='Outcome')


# %%
sns.lmplot(data=df, x='Age', y='Pregnancies', hue='Outcome', col='Outcome')


