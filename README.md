# EXNO:4-DS
```
Name-Dhanus karthi
Reg no-24005701
```
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
![439179299-dcbe66db-91ed-40f4-b2a7-462c1e31954f](https://github.com/user-attachments/assets/d0e6ca6e-b991-4984-b560-fd0d60f69def)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![439179416-f99bf383-a8d1-440b-99a3-569422757cd2](https://github.com/user-attachments/assets/642c250e-107a-4d76-91d3-fdf9756374dd)
```
df.dropna()
```
![439179555-d41cb620-f7f3-4211-ba5e-f624103caec7](https://github.com/user-attachments/assets/70c9248e-46a1-4622-b0fb-a3abe35477a0)
```
max_values = np.max(np.abs(df[['Height','Weight']]),axis=0)
max_values
```
![439179697-0992fbcb-974a-4b0a-b2e2-4be281004451](https://github.com/user-attachments/assets/5b28f6a3-2b99-4d60-b7b4-26efd5f2e59b)
```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![439179931-35187da7-957c-45c4-a410-c2fa7f7fb695](https://github.com/user-attachments/assets/a1859fb4-64c8-4182-8be6-52d5b7aa5017)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![439180036-7eef4b4c-2214-4d6b-af34-b7817985ed8e](https://github.com/user-attachments/assets/f97ae87b-8f3f-47e5-b165-2516af538016)
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![439180507-71434b46-fba0-4f02-9488-cc0039956a92](https://github.com/user-attachments/assets/a71851ca-a547-4fc2-a430-a6a553553cdd)
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
![439180911-8d6ace3d-f11b-4913-9450-cd3fe709cdf6](https://github.com/user-attachments/assets/82ebe9c4-635a-4cce-bfec-8b465abfd059)
```
df=pd.read_csv("income(1) (1).csv")
df.info()
```
![439181010-c7487c11-7b6a-49cd-aef3-b8af5054112d](https://github.com/user-attachments/assets/7a596228-6b6c-4188-b584-87af2f68b202)
```
df_nullvalues_sum=df.isnull().sum()
df_nullvalues_sum
```
![439181350-ab8cebe8-50fb-4166-b816-ac9032b94141](https://github.com/user-attachments/assets/652e793b-87fe-48d9-a507-4f580a8f6efd)
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![439181468-71affddd-9d0f-4ec4-a312-67a2a94cf40b](https://github.com/user-attachments/assets/fff7b85e-e097-427d-ab95-05087704c9b8)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![439181628-8ce89669-41d4-4519-958b-343c3aea701e](https://github.com/user-attachments/assets/da87ed36-852c-48cc-be3b-d89e6ef38d37)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![439181830-6d32ba0c-5c8a-49c8-8b25-02143676c9bb](https://github.com/user-attachments/assets/a91c0bf2-3e3f-4c1f-acc5-81fa1002a765)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![439181950-43078c26-4974-48f0-b294-ae59cc950ed3](https://github.com/user-attachments/assets/3fa994d5-3049-437c-86c5-904b6727a3f1)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![439182049-1d0f687f-9ab9-4dac-b448-f05c7f222917](https://github.com/user-attachments/assets/63023e45-8180-4696-8672-9bfb1f9c9157)
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![439182339-c7cf770e-7c40-4ae2-9b20-a55e92da8d18](https://github.com/user-attachments/assets/285d2521-d68f-42c9-8cd6-8e141c23e337)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![439182572-6d2b25b2-17c0-4002-8b38-5ed15284c2aa](https://github.com/user-attachments/assets/2fa46a6b-258e-4604-a797-8eef06cebbe3)
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("income(1) (1).csv")
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![439182834-1b352ac8-2ae4-4979-908b-5b41d341cf91](https://github.com/user-attachments/assets/b7b88076-36bb-4f57-b0e9-045b01cce45b)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![439183042-f2d2486b-91d3-449d-8eaa-03fbf19fd46b](https://github.com/user-attachments/assets/3a8ec261-fc89-4c8a-8479-4f7ded23c312)


# RESULT:
 Thus, Feature selection and Feature scaling has been used on thegiven dataset
