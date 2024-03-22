#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import xgboost
from xgboost import XGBClassifier
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            classification_report, confusion_matrix, cohen_kappa_score, \
                            hamming_loss
import warnings
warnings.filterwarnings("ignore")
pyo.init_notebook_mode()


# In[11]:


df = pd.read_csv("genres_v2.csv")


# In[12]:


df.head()


# In[13]:


df.describe()


# In[14]:


df.shape



# In[15]:


df.info()



# In[16]:


df = df.drop(["type","id","uri","track_href","analysis_url","song_name","Unnamed: 0","title", "duration_ms", "time_signature"
              ,"mode"], axis =1)


# In[18]:


numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(20,10))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f')
plt.show()


# In[19]:


df["genre"].value_counts()


# In[20]:


df["genre"].value_counts()


# In[21]:


ax = sns.histplot(df["genre"])
_ = plt.xticks(rotation=80)
_ = plt.title("Count for each Genre")


# In[22]:


df.nunique()


# In[23]:


df.isnull().sum(axis=0)


# In[24]:


cols = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo']


# In[25]:


# Create subplots
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(13, 30))
axs = np.ravel(axs)

# Loop through the columns and plot the KDE plot
for i, col in enumerate(cols):
    plt.sca(axs[i])
    sns.kdeplot(data=df[col], fill=True, color='red')

# Adjust the layout of the subplots
plt.tight_layout()

# Show the plot
plt.show()


# In[26]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='danceability')
plt.show()


# In[27]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='loudness')
plt.show()


# In[28]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='speechiness')
plt.show()


# In[29]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='energy')
plt.show()


# In[30]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='acousticness')
plt.show()


# In[31]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='instrumentalness')
plt.show()


# In[32]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='liveness')
plt.show()


# In[33]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='valence')
plt.show()


# In[34]:


plt.figure(figsize = (13, 8))
plt.xticks(rotation=25)
sns.barplot(data=df, x='genre', y='tempo')
plt.show()


# In[20]:


import plotly.express as px

# Assuming df is your DataFrame containing the data
fig = px.scatter_3d(df, x='tempo', y='loudness', z='genre', color='genre')
fig.show()


# In[23]:


import matplotlib.pyplot as plt
def find_outliers_IQR(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    return outliers
dance_outliers = find_outliers_IQR(df['danceability'])
print('Number of outliers of danceability: '+ str(len(dance_outliers)))


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.boxplot(column='danceability')
fig.set_title('')
fig.set_ylabel('danceability')

plt.subplot(2, 2, 2)
fig = df.boxplot(column='energy')
fig.set_title('')
fig.set_ylabel('energy')
energy_outliers = find_outliers_IQR(df['energy'])
print('Number of outliers of energy: '+ str(len(energy_outliers)))

plt.subplot(2, 2, 3)
fig = df.boxplot(column='key')
fig.set_title('')
fig.set_ylabel('key')
key_outliers = find_outliers_IQR(df['key'])
print('Number of outliers of key: '+ str(len(key_outliers)))


plt.subplot(2, 2, 4)
fig = df.boxplot(column='loudness')
fig.set_title('')
fig.set_ylabel('loudness')
loud_outliers = find_outliers_IQR(df['loudness'])
print('Number of outliers of loudness: '+ str(len(loud_outliers)))


# In[37]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = df.boxplot(column='speechiness')
fig.set_title('')
fig.set_ylabel('speechiness')
speech_outliers = find_outliers_IQR(df['speechiness'])
print('Number of outliers of speechiness: '+ str(len(speech_outliers)))


plt.subplot(2, 2, 2)
fig = df.boxplot(column='acousticness')
fig.set_title('')
fig.set_ylabel('acousticness')
acousticness_outliers = find_outliers_IQR(df['acousticness'])
print('Number of outliers of acousticness: '+ str(len(acousticness_outliers)))


plt.subplot(2, 2, 3)
fig = df.boxplot(column='instrumentalness')
fig.set_title('')
fig.set_ylabel('instrumentalness')
instrumentalness_outliers = find_outliers_IQR(df['instrumentalness'])
print('Number of outliers of instrumentalness: '+ str(len(instrumentalness_outliers)))


plt.subplot(2, 2, 4)
fig = df.boxplot(column='liveness')
fig.set_title('')
fig.set_ylabel('liveness')
liveness_outliers = find_outliers_IQR(df['liveness'])
print('Number of outliers of liveness: '+ str(len(liveness_outliers)))


# In[38]:


plt.subplot(2, 2, 1)
fig = df.boxplot(column='valence')
fig.set_title('')
fig.set_ylabel('valence')
valence_outliers = find_outliers_IQR(df['valence'])
print('Number of outliers of valence: '+ str(len(valence_outliers)))


plt.subplot(2, 2, 2)
fig = df.boxplot(column='tempo')
fig.set_title('')
fig.set_ylabel('tempo')
tempo_outliers = find_outliers_IQR(df['tempo'])
print('Number of outliers of tempo: '+ str(len(tempo_outliers)))


# In[39]:


# Removing outliers
Q1 = df['acousticness'].quantile(0.25)
Q3 = df['acousticness'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds of the IQR range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers from the column
df = df[(df['acousticness'] >= lower_bound) & (df['acousticness'] <= upper_bound)]


# In[40]:


Q1 = df['energy'].quantile(0.25)
Q3 = df['energy'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['energy'] >= lower_bound) & (df['energy'] <= upper_bound)]


# In[41]:


Q1 = df['loudness'].quantile(0.25)
Q3 = df['loudness'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['loudness'] >= lower_bound) & (df['loudness'] <= upper_bound)]


# In[42]:


Q1 = df['danceability'].quantile(0.25)
Q3 = df['danceability'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['danceability'] >= lower_bound) & (df['danceability'] <= upper_bound)]


# In[43]:


Q1 = df['liveness'].quantile(0.25)
Q3 = df['liveness'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['liveness'] >= lower_bound) & (df['liveness'] <= upper_bound)]


# In[44]:


Q1 = df['speechiness'].quantile(0.25)
Q3 = df['speechiness'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['speechiness'] >= lower_bound) & (df['speechiness'] <= upper_bound)]


# In[45]:


Q1 = df['tempo'].quantile(0.25)
Q3 = df['tempo'].quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['tempo'] >= lower_bound) & (df['tempo'] <= upper_bound)]


# In[46]:


df.shape


# In[47]:


df["genre"].value_counts()


# In[48]:


ax = sns.histplot(df["genre"])
_ = plt.xticks(rotation=80)
_ = plt.title("Count of Each Genres After Removing Outliers")


# In[49]:


# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['genre']= label_encoder.fit_transform(df['genre'])
  
df['genre'].unique()
print(df.head())


# In[50]:


#Count for each genre after label encoder
df["genre"].value_counts()


# In[51]:


#Split data into feature and target
x = df.drop(['genre'], axis=1)
y = df['genre']
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 42)
#Show the number of data in train and test data
x_train.shape, x_test.shape


# In[52]:


x_train.describe()


# In[53]:


x_test.describe()


# In[54]:


cols = x_train.columns
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = pd.DataFrame(x_train, columns=[cols])
x_test = pd.DataFrame(x_test, columns=[cols])


# In[55]:


x_train.describe()


# In[56]:


x_test.describe()


# In[57]:


# Define the parameter space for hyperparameter tuning
param_dist = {'alpha': uniform(0.1, 1.0),
              'fit_prior': [True, False],
              'class_prior': [None, uniform(0.1, 1.0)]
              }

# Create a Multinomial Naive Bayes classifier
mnb = MultinomialNB()

# Perform hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=mnb, param_distributions=param_dist, n_iter=100, cv=5)
random_search.fit(x_train, y_train)

# Print the best hyperparameters and their corresponding score
print("Best Hyperparameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)


# In[58]:


#Develop mnb using best parameter to perform classification
mnb = MultinomialNB(alpha=0.1724, fit_prior=True, class_prior=None)
mnb.fit(x_train,y_train)
mnb_pred = mnb.predict(x_test)


# In[59]:


#Classfication Report of MNB
print(classification_report(y_test,mnb_pred))


# In[60]:


#Confusion Metrics of MNB

mnb_cm = confusion_matrix(y_test, mnb_pred)
fig = plt.subplots(figsize=(12, 10))
ax = sns.heatmap(mnb_cm, annot=True, fmt="d", cmap="Blues")


# In[61]:


#Performance Metrics of MNB
mnb_accuracy = accuracy_score(y_test, mnb_pred)
mnb_precision_w = precision_score(y_test, mnb_pred, average='weighted')
mnb_recall_w = recall_score(y_test, mnb_pred, average='weighted')
mnb_f1_w = f1_score(y_test, mnb_pred, average='weighted')
mnb_precision_m = precision_score(y_test, mnb_pred, average='macro')
mnb_recall_m = recall_score(y_test, mnb_pred, average='macro')
mnb_f1_m = f1_score(y_test, mnb_pred, average='macro')
mnb_kappa = cohen_kappa_score(y_test, mnb_pred)
mnb_hamming = hamming_loss(y_test, mnb_pred)

print("Accuracy: %.4f"%mnb_accuracy)
print("Precision (Weighted): %.4f"%mnb_precision_w)
print("Recall (Weighted): %.4f"%mnb_recall_w)
print("F1 Score (Weighted): %.4f"%mnb_f1_w)
print("Precision (Macro): %.4f"%mnb_precision_m)
print("Recall (Macro): %.4f"%mnb_recall_m)
print("F1 Score (Macro): %.4f"%mnb_f1_m)
print("Cohen's Kappa: %.4f"%mnb_kappa)
print("Hamming Loss: %.4f"%mnb_hamming)


# In[62]:


# Define the parameter space for hyperparameter tuning
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# Create a Random Forest classifier
rfc = RandomForestClassifier()

# Perform hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist, n_iter=10, cv=5, verbose=1)
random_search.fit(x_train, y_train)

# Print the best hyperparameters and their corresponding score
print("Best Hyperparameters: ", random_search.best_params_)
print("Best Score: ", random_search.best_score_)


# In[67]:


rfc = RandomForestClassifier(criterion='gini', n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_features=None, max_depth=10, bootstrap=False)
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)


# In[68]:


print(classification_report(y_test,rfc_pred))


# In[69]:


cm = confusion_matrix(y_test,rfc_pred)
fig = plt.subplots(figsize=(12, 10))
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")


# In[70]:


# Performance Metrics
rfc_accuracy = accuracy_score(y_test, rfc_pred)
rfc_precision_w = precision_score(y_test, rfc_pred, average='weighted')
rfc_recall_w = recall_score(y_test, rfc_pred, average='weighted')
rfc_f1_w = f1_score(y_test, rfc_pred, average='weighted')
rfc_precision_m = precision_score(y_test, rfc_pred, average='macro')
rfc_recall_m = recall_score(y_test, rfc_pred, average='macro')
rfc_f1_m = f1_score(y_test, rfc_pred, average='macro')
rfc_kappa = cohen_kappa_score(y_test, rfc_pred)
rfc_hamming = hamming_loss(y_test, rfc_pred)

print("Accuracy: %.4f"% rfc_accuracy)
print("Precision (Weighted): %.4f"% rfc_precision_w)
print("Recall (Weighted): %.4f"% rfc_recall_w)
print("F1 Score (Weighted): %.4f"% rfc_f1_w)
print("Precision (Macro): %.4f"% rfc_precision_m)
print("Recall (Macro): %.4f"% rfc_recall_m)
print("F1 Score (Macro): %.4f"% rfc_f1_m)
print("Cohen's Kappa: %.4f"% rfc_kappa)
print("Hamming Loss: %.4f"% rfc_hamming)

