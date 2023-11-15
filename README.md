# Ex NO : 07  Feature Selection Techniques

## DATE :  26/10/2023

## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


## CODE:

DEVELOPED BY : YUGENDARAN.G

REGISTER NO.:212221220063
```
# DATA PREPROCESSING BEFORE FEATURE SELECTION:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()

#checking data
df.isnull().sum()

#removing unnecessary data variables
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()

#cleaning data
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

#removing outliers 
plt.title("Dataset with outliers")
df.boxplot()
plt.show()

cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()

from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()

from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()

import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()

import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 

# FEATURE SELECTION:
# FILTER METHOD:
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()

# HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features

# BACKWARD ELIMINATION:
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

# RFE (RECURSIVE FEATURE ELIMINATION):
model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

# OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

# FINAL SET OF FEATURE:
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

# EMBEDDED METHOD:
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```


## OUPUT:
## DATA PREPROCESSING BEFORE FEATURE SELECTION:
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/801b9c66-3d60-4511-9930-5ccb578a76b8)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/b3d96036-a1a1-4406-bce6-63bdb7e8c6ae)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/1714cbf6-b9ae-45b9-82c5-bea269d1d1c6)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/228fd02e-26eb-43c5-80ed-5f64a5cd1cfa)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/57c6dfaa-2470-4199-ba37-4e24c542956b)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/053b6a6a-dad4-4a50-a6e1-ffeec327a553)
# FEATURE SELECTION:
## FILTER METHOD:
The filtering here is done using correlation matrix and it is most commonly done using Pearson correlation.
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/3021e98f-98cb-47d0-8e3c-04201b4333fb)
## CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/3e8c0b36-ddd3-4d2b-91a1-f45b80eeb60d)
## HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/a9fa5229-14e0-46e4-b512-089cf1d1b392)
## WRAPPER METHOD:
Wrapper Method is an iterative and computationally expensive process but it is more accurate than the filter method.
There are different wrapper methods such as Backward Elimination, Forward Selection, Bidirectional Elimination and RFE.
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/6bd1af88-3223-465e-8ae5-6bb18db60ae5)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/e9bbdc53-fe12-4a84-b555-487adc98be0e)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/5d86c1ec-f876-4d4d-a964-bd315bf751a8)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/d31a14a1-f801-4365-bc4d-b17b6df50610)
![image](https://github.com/Yugendaran/ODD2023-Datascience-Ex-07/assets/128135616/acb3f050-27f5-488e-8215-471272aacd0b)

## RESULT :
Thus, the various feature selection techniques have been performed on a given dataset successfully.
