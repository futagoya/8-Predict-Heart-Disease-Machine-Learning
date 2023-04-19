#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


url = 'https://raw.githubusercontent.com/futagoya/8-Predict-Heart-Disease-Machine-Learning/main/heart.csv'
df = pd.read_csv(url)


# # Machine Learning Project: 2-Predict-Heart-Disease

# ## Dataset on: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset 
Credit to Data Thinkers Youtubenote:
age
sex
chest pain type (4 values)
 Value 0: typical angina
 Value 1: atypical angina
 Value 2: non-anginal pain
 Value 3: asymptomatic
trestbps: resting blood pressure (in mm Hg on admission to the hospital)
chol: serum cholestoral in mg/dl
fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg: resting electrocardiographic results
 Value 0: normal
 Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
 Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach: maximum heart rate achieved
exang: exercise induced angina (1 = yes; 0 = no)
oldpeak = ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
 Value 1: upsloping
 Value 2: flat
 Value 3: downsloping
ca: number of major vessels (0-3) colored by flourosopy
thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
target : 0=less chance of heart attack, 1=more chance of heart attack
# In[3]:


df.head()


# # 1. Check the Dataset's information and the null values

# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# # 2. Check and drop duplicated value

# In[6]:


duplicate_data = df.duplicated().any()
duplicate_data


# In[7]:


df = df.drop_duplicates()


# In[8]:


duplicate_data = df.duplicated().any()
duplicate_data


# # 3. Data Processing

# In[9]:


cat_col = []
num_col = []

for column in df.columns:
    if df[column].nunique() <= 5:
        cat_col.append(column)
    else:
        num_col.append(column)


# In[10]:


cat_col


# In[11]:


num_col


# # 4. Encoding the Categorical Columns

# #### To prevent dummy variable trap, we need to drop first column of Cat_Col which has more than 2 values

# #### 'Sex' and 'Target' are not included

# In[12]:


cat_col.remove('sex')
cat_col.remove('target')
df=pd.get_dummies(df,columns=cat_col,drop_first=True)


# In[13]:


df.head()


# # 5. Scale the Num_col (Feature Scaling)

# We need to scale the numerical columns to get the better analysis of machine learning because this dataset is distance based values

# If we do not use Feature Scaling, then the higher numbers will dominate the data

# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


st = StandardScaler()
df[num_col] = st.fit_transform(df[num_col])


# In[16]:


df.head()


# # 6. Split the data into training & test set
Target = dependent variable. 
The others = independent variables.
# ### 1. Split the data

# ### 2. Train the models on training set

# ### 3. Test the models on testing set

# In[17]:


X = df.drop(['target'],axis=1)
y = df['target']


# In[39]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=29)


# # 7. Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression


# In[40]:


log = LogisticRegression()
log.fit(X_train, y_train)


# In[41]:


y_pred1 = log.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score


# In[43]:


res_1 = accuracy_score(y_test, y_pred1)
res_1


# # 8. SVM

# In[45]:


from sklearn import svm


# In[46]:


svm = svm.SVC()
svm.fit(X_train, y_train)


# In[47]:


y_pred2 = svm.predict(X_test)
res_2 = accuracy_score(y_test, y_pred2)
res_2


# # 9. KNeighbors Classifier

# In[48]:


from sklearn.neighbors import KNeighborsClassifier


# In[49]:


knn = KNeighborsClassifier()


# In[50]:


knn.fit(X_train, y_train)


# In[51]:


y_pred3 = knn.predict(X_test)


# In[52]:


y_pred3 = knn.predict(X_test)
res_3 = accuracy_score(y_test, y_pred3)
res_3


# In[53]:


score = []

for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))


# In[55]:


score


# In[54]:


knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(X_train, y_train)
y_pred3 = knn.predict(X_test)
res_3 = accuracy_score(y_test, y_pred3)
res_3


# # Non-Linear ML Algorithms (No-need to pre-process Data)

# In[56]:


url = 'https://raw.githubusercontent.com/futagoya/8-Predict-Heart-Disease-Machine-Learning/main/heart.csv'
df = pd.read_csv(url)
df.head()


# In[57]:


df = df.drop_duplicates()


# In[58]:


X = df.drop(['target'],axis=1)
y = df['target']


# In[59]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=29)


# # 10. Decision Tree Classifier

# In[60]:


from sklearn.tree import DecisionTreeClassifier


# In[61]:


dtc = DecisionTreeClassifier()


# In[62]:


dtc.fit(X_train, y_train)


# In[64]:


y_pred4 = dtc.predict(X_test)
res_4 = accuracy_score(y_test, y_pred4)
res_4


# # 11. Random Forest Classifier

# In[65]:


from sklearn.ensemble import RandomForestClassifier


# In[66]:


rfc = RandomForestClassifier()


# In[67]:


rfc.fit(X_train, y_train)


# In[68]:


y_pred5 = rfc.predict(X_test)
res_5 = accuracy_score(y_test, y_pred5)
res_5


# # 12. Gradient Boosting Classifier

# In[69]:


from sklearn.ensemble import GradientBoostingClassifier


# In[70]:


gbc = GradientBoostingClassifier()


# In[71]:


gbc.fit(X_train, y_train)


# In[72]:


y_pred6 = gbc.predict(X_test)
res_6 = accuracy_score(y_test, y_pred6)
res_6


# In[73]:


final_data = pd.DataFrame({'Models':['LogisticRegression','SVM','KNeighbors','DecisionTree','RFC','GBR'],'Accuracy':
                           [res_1,res_2,res_3,res_4,res_5,res_6]})
final_data


# In[74]:


import seaborn as sns


# In[75]:


sns.barplot(x=final_data['Models'],y=final_data['Accuracy'])


# # 13. Choosing the Random Forrest Classifier

# In[76]:


url = 'https://raw.githubusercontent.com/futagoya/8-Predict-Heart-Disease-Machine-Learning/main/heart.csv'
df = pd.read_csv(url)
df = df.drop_duplicates()
X=df.drop('target',axis=1)
y=df['target']


# In[77]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X,y)


# In[78]:


import joblib


# In[79]:


joblib.dump(rfc,'heart_disease_analysis_rfc')


# In[80]:


model = joblib.load('heart_disease_analysis_rfc')


# # 13. Using GUI

# In[81]:


from tkinter import *
import joblib


# In[85]:


from tkinter import *
import joblib
import numpy as np
from sklearn import *
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('heart_disease_analysis_rfc')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p8,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="Less Chance of Heart Disease").grid(row=20)
    else:
        Label(master, text="More Chance of Heart Disease").grid(row=20)
    
    
master = Tk()
master.title("Heart Disease Prediction System")


label = Label(master, text = "Heart Disease Prediction System"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter Value of CP").grid(row=3)
Label(master, text="Enter Value of trestbps").grid(row=4)
Label(master, text="Enter Value of chol").grid(row=5)
Label(master, text="Enter Value of fbs").grid(row=6)
Label(master, text="Enter Value of restecg").grid(row=7)
Label(master, text="Enter Value of thalach").grid(row=8)
Label(master, text="Enter Value of exang").grid(row=9)
Label(master, text="Enter Value of oldpeak").grid(row=10)
Label(master, text="Enter Value of slope").grid(row=11)
Label(master, text="Enter Value of ca").grid(row=12)
Label(master, text="Enter Value of thal").grid(row=13)



e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)



Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:




