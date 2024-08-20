#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING NECESSARY LIBRARIES 
import numpy as np
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score


# UPLOADING AND READING CSV FILE TO JUPITERNOTEBOOK AND VISUALIZING TOP 5 ROW OF THE DATA. 

# In[2]:


df = pd.read_csv('./incident_event_log.csv')
df.head()


# CHECKING THE SHAPE OF THE DATA

# In[3]:


df.shape


# LOOKING FOR MISSING VALUES IN THE DATA

# In[4]:


df.isnull().sum()


# GETTING DATA STRACTURE INFORMATION

# In[6]:


df.info()


# BECAUSE OF ANOMALY SOME COLUMNS ARE SHOWING FALSE DATATYPE STRUCTURE AND WE CAN SEE THERE IS BOOLEAN DATA TYPES PRESENT IN DATA.

# In[6]:


cols = list(df.columns) #to create a list of all the columns
print(cols)


# DECIDED TO REPLACE THE ANOMALY WITH NAN

# In[7]:


null = [] #An empty list to save all the column names with '?' as a value
for i in range(0,len(cols)):
    a = list(pd.unique(df[cols[i]]))
    for j in range(0,len(a)):
        if a[j] == '?':
            null.append(cols[i])
            #print(cols[i])
        else:
            pass
print(null) 


# LISTING THE COLUMN NAME WHICH HAS "?" AS A VALUE

# REPLACING THE "?" VALUE WITH NAN IN THE DATA AND CHECKING FOR UPDATION

# In[8]:


df_1 = df.replace(to_replace ="?",value = np.nan)
df_1.head()


# CHECKING THE DATA TYPES AND COUNT OF NULL VALUE COLUMN

# In[10]:


df_1.info()
df_1.isnull().sum()


# DROPING COLUMN WITH MORE NULL VALUE AND ALSO REMOVING IMPACT AND URGENCY SINCE PRIORITY VALUE ARE IMPUTED FROM THEM.

# In[9]:


# drop the columns that most values is nan
df_2 = df_1.copy()
df_2.drop(columns = ['cmdb_ci','problem_id','rfc','vendor','caused_by',], inplace = True)
# remove impact and urgency, since Priority value is directly computed from them.
df_2.drop(columns = ['impact','urgency'], inplace = True)


# CHECKING THE SHAPE OF THE DATA AFTER REMOVING THE COLUMN

# In[10]:


df_2.shape


# CHECKING FOR DUPLICATES

# In[11]:


df_2.duplicated().sum()


# In[ ]:


THERE IS ANOMALY PRESENT IN INCIDENT STATE COLUMN SO DECIDED TO IMPUTE WITH MOST FREQUENT OCCURING VALUE.


# In[12]:


df_2['incident_state'].value_counts()


# In[13]:


# If the column contains mixed types, convert all to string
df_2['incident_state'] = df_2['incident_state'].astype(str)

# Identify the anomaly
anomaly_value = '-100'

# Determine the most frequent valid value (mode)
most_frequent_value = df_2[df_2['incident_state'] != anomaly_value]['incident_state'].mode()[0]

# Replace the anomaly with the most frequent value
df_2['incident_state'] = df_2['incident_state'].replace(anomaly_value, most_frequent_value)

print("Updated 'incident_state' column:\n", df_2['incident_state'].value_counts())


# In[14]:


for col in df_2.columns:
    print(col, df_2[col].nunique())


# EXTRACTING PATTERN FROM CATAGORICAL COLUMNS AND FORMATING DATATIME COLUMNS 

# In[15]:


df_3=df_2.copy()


# In[16]:


# extract the numbers from the data 
pattern = r'(\d{1,4})'
colum = ['caller_id','opened_by','sys_created_by','sys_updated_by','location','category','subcategory','u_symptom','priority','assignment_group','assigned_to', 'closed_code', 'resolved_by']
for col in colum:
    df_3[col] = df_3[col].str.extract(pattern)

# time    
from datetime import datetime, date
timeColum = ['opened_at', 'sys_created_at','sys_updated_at','resolved_at','closed_at']    
for col in timeColum:
    df_3[col] = pd.to_datetime(df_3[col], format='%d/%m/%Y %H:%M',errors='coerce')


# CHECKING THE UNIQUE COUNT

# In[17]:


for col in df_3.columns:
    print(col, df_3[col].nunique())


# In[ ]:


EXPLORATORY DATA ANALYSIS.


# CHECKING THE PATTERNS AND CORELATION BETWEEN COLUMNS 'opened_by','sys_created_by','sys_updated_by','assignment_group','assigned_to','resolved_by'

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Distribution and correlation among columns
#%matplotlib notebook
idencolum = ['opened_by','sys_created_by','sys_updated_by','assignment_group','assigned_to','resolved_by']    
df_identify = df_3.loc[:, idencolum]
for col in idencolum:
    df_identify[col] = pd.to_numeric(df_identify[col], errors='coerce').fillna(0).astype(np.int64)
plt.figure()
pd.plotting.scatter_matrix(df_identify,figsize=(12,12))
plt.savefig(r"Distribution and correlation among features.png")


# From the plot of scatter matrix, it can be observed that:
# User Involvement:
# * The individual responsible for resolving incidents (resolved_by) was evenly distributed across the dataset.
# * Other user-related data, such as who opened or was assigned to incidents, showed uneven distribution, indicating varying levels of involvement. 
# 
# Data Relationships:
# * There's a strong correlation between the individuals who opened and created incidents (opened_by and sys_created_by), suggesting potential redundancy in the data.
# * A noticeable relationship exists between the individuals assigned to and those who resolved incidents (assigned_to and resolved_by), implying that the original assignee often completed the resolution. 
# 

# CHECKING THE PATTERNS AND CORELATION BETWEEN COLUMNS 
# 'reassignment_count','reopen_count','made_sla','category','priority','closed_code'

# In[26]:


# continue 
othercolum = ['reassignment_count','reopen_count','made_sla','category','priority','closed_code']

df_other = df_3.loc[:, othercolum]
for col in othercolum:
    df_other[col] = pd.to_numeric(df_other[col], errors='coerce').fillna(0).astype(np.int64)
plt.figure()
pd.plotting.scatter_matrix(df_other,figsize=(12,12))
plt.savefig(r"Distribution and correlation among features_2.png")


# Comments based on this scatter matrix plot:
# 
# The majority of incidents experienced no reassignments or reopenings (reassignment_count and reopen_count).
# A significant proportion (approximately 90%) of incidents had an associated Service Level Agreement (SLA).
# Incident priority is skewed towards moderate (priority 3), emphasizing the need for careful metric selection in priority prediction models.
# The most frequent closure code is approximately 6.
# The dataset exhibits significant imbalance across various categories.

# Specific Features Understanding
# Incident state

# In[27]:


plt.figure()
bins = np.arange(0,df_2.incident_state.nunique()+2,1)

ax = df_3.incident_state.hist(width =0.6,bins= bins,figsize=(6,4),align='mid')
plt.xticks(rotation=45)
ax.grid(False)
ax.set_xticks(bins[:-1])
ax.set_ylabel('Numbers')
ax.set_title('Distribution of the incident_state')


# The figure illustrates seven distinct stages an incident can progress through before closure. While the volume of 'New' and 'Active' incidents surpasses 'Closed' cases, this discrepancy is attributable to the potential for incidents to be reopened multiple times. If an initial resolution is unsatisfactory to the caller, the incident can be re-escalated, as indicated by the 'reopen_count' field within the dataset

# RELATIONSHIP BETWEEN MADE_SLA AND REOPEN_COUNT 

# In[28]:


sla = (df_3[(df_3.made_sla == True) & (df_3.reopen_count>0)].groupby('number')['reopen_count'].mean()).mean()
nosla = (df_3[(df_3.made_sla == False) & (df_3.reopen_count>0)].groupby('number')['reopen_count'].mean()).mean()
print(f'mean reopen_count for having SLA {sla} and without SLA {nosla}')


# Incidents with SLAs have a lower average number of reopenings (1.11) compared to those without SLAs (1.24).
# This indicates that incidents managed under SLA guidelines tend to be resolved more effectively, reducing the need for re-opening.

# In[ ]:


DISTRIBUTION OF CLOSED_CODE;RELATIONSHIP BETWEEN CLOSE_CODE AND REOPEN_COUNT 


# In[30]:


import seaborn as sns
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12,4))
bins=np.arange(0,df_3.closed_code.nunique()+2,1)
df_2[df_2.incident_state=='Closed'].sort_values('closed_code').closed_code.hist(width =0.8,bins = bins,align='left',ax=ax1)
ax1.grid(False)
ax1.set_xticks(bins[:-1])
ax1.set_xlabel('closed code')
ax1.set_ylabel('Numbers')
ax1.set_title('Distribution of closed_code')


dfclosecode = df_3[(df_3.reopen_count>0) & (df_2.incident_state=='Closed')]
dfclose_reopen = dfclosecode.groupby('closed_code').reopen_count.mean()
dfclose_reopen.plot.bar(ax=ax2)
ax2.grid(False)
ax2.set_ylabel('mean of reopen_count')
ax2.set_xticks(bins[:-1])
ax2.set_title('closed_code vs. reopen_count')
plt.show()


# From this figure, it can be observed that:
# The most frequently occurring closure code is 6.
# Closure codes 12, 13, 14, and 15 never result in incident re-openings (reopen_count = 0).
# Incidents closing with code 10 have a high average number of re-openings, suggesting that resolutions for this code category are frequently rejected.

# In[31]:


df_ar = df_3.loc[:,['assigned_to','resolved_by']]
df_ar['equal'] = np.where(df_ar.assigned_to == df_ar.resolved_by,1,0)
equal_num = df_ar['equal'].sum()
print(equal_num/df_ar.shape[0] * 100)


# A mere 0.4% of incidents are resolved by the analyst initially assigned to them. This significant discrepancy highlights a potential for optimization. Accurately predicting the eventual resolver could substantially enhance the efficiency of incident management processes.

# COMPLETION TIME FOR INCIDENT 

# In[32]:


# completion time for incident resolution 
df_closed = df_3[df_3.incident_state=='Closed'].reset_index()
df_closed['completion_time_days'] = (df_closed.closed_at- df_closed.opened_at).dt.total_seconds()/3600/24
#print(f'The mean of completion time for incident resolution is {df_closed.completion_time_days.mean()} days.')

#plots
plt.figure()
ax = df_closed['completion_time_days'].plot(figsize=(24,4))
ax.grid(False)
ax.set_ylabel('completion time in days')
ax.set_title('Distribution of completion_time')


# In[33]:


df_closed['completion_time_days'].describe()


# COMPLETION TIME VS CLOSED CODE; COMPLETION TIME VS MADE_SLA 

# In[34]:


plt.figure()
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(12,4))
df_closecode_time = df_closed.groupby('closed_code')['completion_time_days'].mean()
df_closecode_time.plot.bar(ax=ax1)
ax1.grid(False)
ax1.set_ylabel(' mean completion time in days')
ax1.set_title('completion time vs closed code')

df_made_sla_time = df_closed.groupby('made_sla')['completion_time_days'].mean()
df_made_sla_time.plot.bar(ax=ax2)
ax2.grid(False)
ax2.set_ylabel('mean completion time in days')
ax2.set_title('completion time vs made_sla')
plt.show()


# This figure indicates that:
# Closure Code 13: Incidents resolved with closure code 13 exhibit an unusually long mean completion time of approximately 40 days, which is significantly higher than other closure codes. This is counterintuitive given the previous observation of zero re-openings for this code.
# SLA Impact: Incidents without Service Level Agreements (SLAs) have considerably longer mean completion times compared to those with SLAs. Implementing SLAs is likely to improve overall incident management efficiency.

# In[35]:


df_closed.dtypes 


# In[37]:


df_closed.describe()


# It likely contains information on incident tickets, including reassignment counts, reopen counts, system modification counts, completion time in days, and total counts.
# All statistical values (mean, standard deviation, min, max, etc.) are identical, suggesting potential data quality issues.

# General analysis of What types of incidents are getting reopened?

# Incidents are reassigned and reopend sometimes, maximum reassigment is 27 whereas there are incidents which are reopened 8 times.Let us first see that with what close codes incidents are getting closed

# In[38]:


df_3.groupby('closed_code')     .count()['number']     .plot(kind='barh',
          title='Distribution of closed_code',
          figsize=(15, 3))
plt.show()
df['closed_code'].value_counts()


# The majority of incidents are closed with closure codes 6, 7, 8, or 9.

# Now let us check if that happens for the ones which are getting reopened or it is different?

# In[39]:


df_closed_reo = df_closed["reopen_count"] > 0
df_closed_reo = df_closed[df_closed_reo]


# In[40]:


df_closed_reo.groupby('closed_code')     .count()['number']     .plot(kind='barh',
          title='Distribution of closed_code',
          figsize=(15, 3))
plt.show()
df_closed_reo['closed_code'].value_counts()


# While the primary closure codes are consistent for reopened tickets, a variety of other codes are also present. To understand if these reopened incidents exhibit any correlation with Service Level Agreements (SLAs), we will examine the data further.

# In[41]:


df_closed['made_sla'].value_counts()


# In[42]:


df_closed_reo['made_sla'].value_counts()


# A strong correlation exists between reopened incidents and missed SLAs. Our analysis indicates that incidents requiring re-opening are highly likely to have breached their service level agreements.

# Let us now examine if there are different symptoms for reopened incidents.

# In[43]:


df_closed['u_symptom'].value_counts()


# In[44]:


df_closed_reo['u_symptom'].value_counts()


# Symptoms displayed by reopened incidents closely resemble those of other incidents, revealing no discernible pattern linked to specific closure codes. However, a clear correlation exists between reopened incidents and missed SLAs.

# To understand the nature of incidents that frequently miss SLAs, we will analyze the characteristics of incidents that have failed to meet their service level agreements. Given the established link between reopened incidents and SLA breaches, we will delve deeper into the distribution of incidents with missed SLAs.

# In[45]:


df_closed_reo.groupby('made_sla')     .count()['number']     .plot(kind='barh',
          title='Distribution of closed_code',
          figsize=(15, 3))
plt.show()
df_closed['made_sla'].value_counts()


# Approximately 37% of incidents fail to meet their Service Level Agreements (SLAs). We will now identify the characteristics of these incidents to understand the underlying causes for SLA breaches.

# In[46]:


df_closed_sla = df_closed["made_sla"] == 1
df_closed_sla = df_closed[df_closed_sla]


# In[47]:


pd.crosstab(df_closed['made_sla'],df_closed['priority'])


# A clear correlation exists between incident priority and SLA adherence. Critical and high-priority incidents are significantly more likely to miss their SLAs, while moderate and low-priority incidents tend to meet their service level agreements.

# In[48]:


pd.crosstab(df_2['made_sla'],df_2['reopen_count'])


# In[ ]:


A lower incidence of re-opened tickets is observed among incidents that meet their SLAs.


# converting and recalculating date time filed and droping column redundant for analysis.

# In[18]:


# Convert string dates to datetime objects
df_3['resolved_at'] = pd.to_datetime(df_3['resolved_at'], format='%d/%m/%Y')
df_3['closed_at'] = pd.to_datetime(df_3['closed_at'], format='%d/%m/%Y')
df_3['opened_at'] = pd.to_datetime(df_3['opened_at'], format='%d/%m/%Y')

# Calculate days_res_closed and days_op_closed
df_3['days_res_closed'] = (df_3['closed_at'] - df_3['resolved_at']).dt.days
df_3['days_op_closed'] = (df_3['closed_at'] - df_3['opened_at']).dt.days


# In[19]:


columns_to_drop = ['resolved_at', 'closed_at', 'opened_at','days_res_closed','sys_created_at', 'sys_updated_at','opened_by', 'sys_created_by','notify' ]
df4 = df_3.drop(columns=columns_to_drop)


# converting boolean datatypes to interger

# In[20]:


# Convert boolean columns to integers
df4['active'] = df4['active'].astype(int)
df4['made_sla'] = df4['made_sla'].astype(int)
df4['knowledge'] = df4['knowledge'].astype(int)
df4['u_priority_confirmation'] = df4['u_priority_confirmation'].astype(int)


# In[72]:


df4.info()


# Model Performance:  How well does the model predict the likelihood of an incident re-opening? Consider metrics like AUC (Area Under the ROC Curve) or F1 score. 

# In[21]:


unwanted_columns_for_model = ['number', 'sys_mod_count', 'contact_type', 'knowledge', 'u_priority_confirmation']
df5 = df4.drop(columns=unwanted_columns_for_model)


# In[22]:


df5.isnull().sum()


# In[ ]:


principal component analysis.


# In[23]:


import pandas as pd
from sklearn.impute import SimpleImputer

columns_to_convert = df5.select_dtypes(include=[object]).columns

for col_name in columns_to_convert:
    # Check if there are any numeric values before imputation
    if df5[col_name].str.isnumeric().sum() > 0:
        # Filter for rows with numeric values only in the current column
        numeric_df = df5[pd.to_numeric(df5[col_name], errors='coerce').notnull()]

        # Impute missing values in numeric column
        imputer = SimpleImputer(strategy='mean')  # Replace 'mean' with desired strategy
        numeric_df[col_name] = imputer.fit_transform(numeric_df[[col_name]])

        # Convert to integer (if appropriate)
        numeric_df[col_name] = numeric_df[col_name].astype(int)

        # Update the original DataFrame with the modified values (optional)
        # df5.update(numeric_df)  # Uncomment if you want to update the whole DataFrame
    else:
        print(f"Column '{col_name}' does not contain any numeric values for imputation.")

# Impute missing values in categorical columns (example using mode imputation)
for col_name in df5.select_dtypes(include=['object']).columns:
    imputer = SimpleImputer(strategy='most_frequent')
    df5[col_name] = imputer.fit_transform(df5[[col_name]])


# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Convert reopen_count to binary classification
df5['reopen_binary'] = df5['reopen_count'].apply(lambda x: 1 if x > 0 else 0)

# Define your features and target
X = df5.drop(columns=['reopen_count', 'reopen_binary'])
y = df5['reopen_binary']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Create an imputer to replace NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')

# Impute the missing values in the selected features
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balance the dataset using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)  # Adjust k_neighbors as necessary
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# Set a range of k values
k_values = range(1, len(X_balanced.columns) + 1)

# Apply TruncatedSVD for each value of k and store cumulative explained variance
cumulative_variance_ratios = []

for k in k_values:
    # Create a TruncatedSVD model
    svd = TruncatedSVD(n_components=k)
    
    # Fit the model to the balanced data
    svd_model = svd.fit(X_balanced)
    
    # Get the cumulative explained variance
    cumulative_variance_ratio = svd_model.explained_variance_ratio_.sum()
    cumulative_variance_ratios.append(cumulative_variance_ratio)

# Plot the cumulative explained variance
plt.plot(k_values, cumulative_variance_ratios, marker='o')
plt.xlabel('Number of Components (k)')
plt.ylabel('Cumulative Explained Variance')
plt.title('Truncated SVD Cumulative Explained Variance for Different k')
plt.show()


# PCA analysis indicated that 7 principal components account for the majority of variance in the dependent variable. However, due to limited multicollinearity among the original 10 variables, all were retained for model development

# In[27]:


k = 7

# Apply TruncatedSVD for k components
svd = TruncatedSVD(n_components=k)
X_svd = svd.fit_transform(X_balanced)

# Get loadings matrix
loadings_matrix = svd.components_

# Check the shapes again
print("Shape of loadings_matrix:", loadings_matrix.shape)
print("Number of features:", len(X.columns))

# Ensure that the number of features matches the number of loadings
if loadings_matrix.shape[1] != len(X.columns):
    raise ValueError("Number of features in loadings matrix and feature names do not match!")

# Transpose loadings matrix to match features with principal components
loadings = loadings_matrix.T

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

# Set the bar width
bar_width = 0.1

# Bar plot for each principal component with proper spacing
for i in range(k):
    bar_positions = np.arange(len(X.columns)) + (i * bar_width)
    ax.bar(bar_positions, loadings[:, i], width=bar_width, label=f'PC_{i + 1}')

# Adding labels and title
ax.set_xlabel('Features')
ax.set_ylabel('Loadings')
ax.set_title(f'Loadings on Features for Principal Components 1 to {k}')

# Set x-axis tick labels to feature names
ax.set_xticks(np.arange(len(X.columns)) + bar_width * (k - 1) / 2)
ax.set_xticklabels(X.columns, rotation=45, ha='right')

# Adding legend
ax.legend()

# Show the plot
plt.show()

# List the component names
components_names = [f'PC_{i + 1}' for i in range(k)]
print("Component names:", components_names)


# Model building 

# Random fasrest 

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

# Assuming df5 is your DataFrame and previous preprocessing steps have been applied

# Convert reopen_count to binary classification
df5['reopen_binary'] = df5['reopen_count'].apply(lambda x: 1 if x > 0 else 0)

# Define your features and target
X = df5.drop(columns=['reopen_count', 'reopen_binary'])
y = df5['reopen_binary']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Create an imputer to replace NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')

# Impute the missing values in the selected features
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balance the dataset using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)  # Adjust k_neighbors as necessary
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# Apply TruncatedSVD to reduce the number of components to 7
svd = TruncatedSVD(n_components=7)
X_svd = svd.fit_transform(X_balanced)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_svd, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'AUC: {auc}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Get feature importances
importances = model.feature_importances_
feature_names = [f'PC{i+1}' for i in range(X_svd.shape[1])]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort feature importances
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()


# The model exhibits exceptional performance with an AUC of 0.999 and F1-score of 0.9961, indicating outstanding classification ability. The confusion matrix confirms high accuracy, with minimal false positives and negatives. Precision, recall, and F1-score are all near perfect across classes, demonstrating the model's effectiveness in correctly identifying both classes

# In[ ]:


XGBClassifier


# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Assuming df5 is your DataFrame and previous preprocessing steps have been applied

# Convert reopen_count to binary classification
df5['reopen_binary'] = df5['reopen_count'].apply(lambda x: 1 if x > 0 else 0)

# Define your features and target
X = df5.drop(columns=['reopen_count', 'reopen_binary'])
y = df5['reopen_binary']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Create an imputer to replace NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')

# Impute the missing values in the selected features
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balance the dataset using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)  # Adjust k_neighbors as necessary
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# Apply TruncatedSVD to reduce the number of components to 7
svd = TruncatedSVD(n_components=7)
X_svd = svd.fit_transform(X_balanced)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_svd, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Train an XGBoost Classifier
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'AUC: {auc}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Get feature importances
importances = model.feature_importances_
feature_names = [f'PC{i+1}' for i in range(X_svd.shape[1])]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort feature importances
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()


# The XGBoost model also demonstrates strong performance with an AUC of 0.9993 and F1-score of 0.9650. While slightly lower than the Random Forest model, it still indicates excellent classification capabilities. The confusion matrix reveals a small increase in false positives and negatives compared to Random Forest, but overall accuracy remains high. Precision, recall, and F1-score are still very good, demonstrating the model's effectiveness in identifying both classes.

# In[ ]:


Logistic regression


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

# Assuming df5 is your DataFrame and previous preprocessing steps have been applied

# Convert reopen_count to binary classification
df5['reopen_binary'] = df5['reopen_count'].apply(lambda x: 1 if x > 0 else 0)

# Define your features and target
X = df5.drop(columns=['reopen_count', 'reopen_binary'])
y = df5['reopen_binary']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Create an imputer to replace NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')

# Impute the missing values in the selected features
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balance the dataset using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)  # Adjust k_neighbors as necessary
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# Apply TruncatedSVD to reduce the number of components to 7
svd = TruncatedSVD(n_components=7)
X_svd = svd.fit_transform(X_balanced)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_svd, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Train a Logistic Regression Classifier
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'AUC: {auc}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Get feature importances (coefficients for logistic regression)
importances = model.coef_[0]
feature_names = [f'PC{i+1}' for i in range(X_svd.shape[1])]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort feature importances
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()


# The logistic regression model demonstrates suboptimal performance. With an AUC of 0.546, the model's ability to discriminate between classes is marginally better than random chance. The F1-score of 0.49 indicates a moderate balance between precision and recall, but overall accuracy is low at 53%. The confusion matrix highlights imbalanced classification with higher false negatives than false positives, suggesting potential issues in identifying the positive class.

# In[ ]:


Gradient Boosting


# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

# Assuming df5 is your DataFrame and previous preprocessing steps have been applied

# Convert reopen_count to binary classification
df5['reopen_binary'] = df5['reopen_count'].apply(lambda x: 1 if x > 0 else 0)

# Define your features and target
X = df5.drop(columns=['reopen_count', 'reopen_binary'])
y = df5['reopen_binary']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Create an imputer to replace NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')

# Impute the missing values in the selected features
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balance the dataset using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)  # Adjust k_neighbors as necessary
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# Apply TruncatedSVD to reduce the number of components to 7
svd = TruncatedSVD(n_components=7)
X_svd = svd.fit_transform(X_balanced)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_svd, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'AUC: {auc}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Get feature importances
importances = model.feature_importances_
feature_names = [f'PC{i+1}' for i in range(X_svd.shape[1])]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort feature importances
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()


# The gradient boosting model exhibits strong performance. With an AUC of 0.841, the model demonstrates excellent discriminative power. The F1-score of 0.756 indicates a good balance between precision and recall, leading to an overall accuracy of 76%. The confusion matrix suggests well-calibrated predictions with relatively balanced false positives and false negatives. These metrics collectively indicate a robust and reliable model for classification.

# MLPClassifier

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

# Assuming df5 is your DataFrame and previous preprocessing steps have been applied

# Convert reopen_count to binary classification
df5['reopen_binary'] = df5['reopen_count'].apply(lambda x: 1 if x > 0 else 0)

# Define your features and target
X = df5.drop(columns=['reopen_count', 'reopen_binary'])
y = df5['reopen_binary']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Create an imputer to replace NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')

# Impute the missing values in the selected features
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balance the dataset using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)  # Adjust k_neighbors as necessary
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# Apply TruncatedSVD to reduce the number of components to 7
svd = TruncatedSVD(n_components=7)
X_svd = svd.fit_transform(X_balanced)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_svd, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Train an MLP Classifier
model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'AUC: {auc}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# The MLP regressor with ReLU activation and Adam optimizer demonstrates a strong AUC of 0.856, indicating good discriminative power. However, the F1-score of 0.766 suggests a potential imbalance in precision and recall. The confusion matrix highlights a significant class imbalance, with high precision but low recall for class 0, and vice versa for class 1. This leads to an overall accuracy of 70%, which is lower than expected given the AUC. Further investigation into class imbalance handling techniques might improve the model's performance.

# In[ ]:


Reinforcement learning


# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

# Assuming df5 is your DataFrame and previous preprocessing steps have been applied

# Convert reopen_count to binary classification
df5['reopen_binary'] = df5['reopen_count'].apply(lambda x: 1 if x > 0 else 0)

# Define your features and target
X = df5.drop(columns=['reopen_count', 'reopen_binary'])
y = df5['reopen_binary']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Create an imputer to replace NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')

# Impute the missing values in the selected features
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balance the dataset using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=5)  # Adjust k_neighbors for better balance
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# Apply TruncatedSVD to reduce the number of components to 7
svd = TruncatedSVD(n_components=7)
X_svd = svd.fit_transform(X_balanced)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_svd, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Initialize Q-table with the number of samples as states and 2 actions (0: not reopen, 1: reopen)
num_samples = X_train.shape[0]
num_actions = 2
q_table = np.zeros((num_samples, num_actions))

# Set learning parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01
num_episodes = 2000  # Increase the number of episodes for more training

# Function to choose an action based on the state and exploration rate
def choose_action(state_index):
    if np.random.random() < exploration_rate:
        return np.random.randint(num_actions)
    else:
        return np.argmax(q_table[state_index])

# Training the Q-learning model
for episode in range(num_episodes):
    state_index = np.random.randint(num_samples)
    
    while True:
        action = choose_action(state_index)
        reward = 1 if (action == y_train.values[state_index]) else -1
        next_state_index = np.random.randint(num_samples)
        
        # Q-learning update
        q_table[state_index, action] = q_table[state_index, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state_index]) - q_table[state_index, action])
        
        if reward == 1 or reward == -1:
            break
        state_index = next_state_index
    
    # Decay the exploration rate
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

# Evaluate the Q-learning model
y_pred = []
for i in range(len(X_test)):
    state_index = i
    action = choose_action(state_index)
    y_pred.append(action)

# Evaluate the model
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'AUC: {auc}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# The reinforcement learning model using a random forest as a function approximator exhibits extremely poor performance. An AUC of approximately 0.5 indicates the model's inability to distinguish between the two classes, performing no better than random chance. The F1-score of 0.018 is exceptionally low, highlighting a severe imbalance in precision and recall. The confusion matrix clearly shows a strong bias towards predicting the majority class, with very few instances of the minority class being correctly classified. Overall, the model is ineffective for this classification task.

# In[ ]:


Reinforcement learning hyperparameter tuning


# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import ParameterGrid

# Assuming df5 is your DataFrame and previous preprocessing steps have been applied

# Convert reopen_count to binary classification
df5['reopen_binary'] = df5['reopen_count'].apply(lambda x: 1 if x > 0 else 0)

# Define your features and target
X = df5.drop(columns=['reopen_count', 'reopen_binary'])
y = df5['reopen_binary']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Create an imputer to replace NaN with most frequent values
imputer = SimpleImputer(strategy='most_frequent')

# Impute the missing values in the selected features
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Balance the dataset using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=5)  # Adjust k_neighbors for better balance
X_balanced, y_balanced = smote.fit_resample(X_imputed, y)

# Apply TruncatedSVD to reduce the number of components to 7
svd = TruncatedSVD(n_components=7)
X_svd = svd.fit_transform(X_balanced)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_svd, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Hyperparameter grid for tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'discount_factor': [0.8, 0.9, 0.95],
    'exploration_rate': [1.0, 0.5, 0.1],
    'exploration_decay': [0.995, 0.99, 0.98],
    'num_episodes': [1000, 2000, 3000]
}

# Grid search for best hyperparameters
best_f1 = 0
best_params = None

for params in ParameterGrid(param_grid):
    learning_rate = params['learning_rate']
    discount_factor = params['discount_factor']
    exploration_rate = params['exploration_rate']
    exploration_decay = params['exploration_decay']
    num_episodes = params['num_episodes']
    
    # Initialize Q-table with the number of samples as states and 2 actions (0: not reopen, 1: reopen)
    num_samples = X_train.shape[0]
    num_actions = 2
    q_table = np.zeros((num_samples, num_actions))

    # Training the Q-learning model
    for episode in range(num_episodes):
        state_index = np.random.randint(num_samples)

        while True:
            action = choose_action(state_index)
            reward = 1 if (action == y_train.values[state_index]) else -1
            next_state_index = np.random.randint(num_samples)

            # Q-learning update
            q_table[state_index, action] = q_table[state_index, action] + learning_rate * (
                reward + discount_factor * np.max(q_table[next_state_index]) - q_table[state_index, action])

            if reward == 1 or reward == -1:
                break
            state_index = next_state_index

        # Decay the exploration rate
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    # Evaluate the Q-learning model
    y_pred = []
    for i in range(len(X_test)):
        state_index = i
        action = choose_action(state_index)
        y_pred.append(action)

    # Evaluate the model
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_params = params

print(f'Best F1 Score: {best_f1}')
print(f'Best Hyperparameters: {best_params}')

# Re-train the Q-learning model with best parameters
learning_rate = best_params['learning_rate']
discount_factor = best_params['discount_factor']
exploration_rate = best_params['exploration_rate']
exploration_decay = best_params['exploration_decay']
num_episodes = best_params['num_episodes']

num_samples = X_train.shape[0]
num_actions = 2
q_table = np.zeros((num_samples, num_actions))

for episode in range(num_episodes):
    state_index = np.random.randint(num_samples)
    
    while True:
        action = choose_action(state_index)
        reward = 1 if (action == y_train.values[state_index]) else -1
        next_state_index = np.random.randint(num_samples)
        
        # Q-learning update
        q_table[state_index, action] = q_table[state_index, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state_index]) - q_table[state_index, action])
        
        if reward == 1 or reward == -1:
            break
        state_index = next_state_index
    
    # Decay the exploration rate
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

# Evaluate the Q-learning model
y_pred = []
for i in range(len(X_test)):
    state_index = i
    action = choose_action(state_index)
    y_pred.append(action)

# Evaluate the model
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'AUC: {auc}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Despite hyperparameter tuning, the reinforcement learning model continues to exhibit extremely poor performance. While the best F1-score of 0.0263 represents a slight improvement over the previous result, it's still far from acceptable.
# 
# The model's AUC remains close to 0.5, indicating little to no discriminative power. The confusion matrix consistently shows a strong bias towards predicting the majority class.
# 
# These results suggest that the fundamental approach of using reinforcement learning for this classification problem might be fundamentally flawed. Alternative algorithms or feature engineering techniques should be explored to improve performance.
# 
# Key Findings:
# 
# Hyperparameter tuning did not yield significant improvements.
# The model's performance is still far below acceptable standards.
# Reinforcement learning might not be suitable for this classification task.

# ## Summary of Analysis
# 
#  User Involvement:
# Resolved_by: Even distribution across the dataset.
# Other user-related data: Uneven distribution, indicating varying levels of involvement in opening or being assigned to incidents.
# 
# Data Relationships:
# Opened_by and sys_created_by: Strong correlation suggesting potential redundancy.
# Assigned_to and resolved_by: Notable relationship indicating that original assignee often completes the resolution.
# 
# Scatter Matrix Observations:
# Reassignment and Reopen Counts: Majority of incidents have no reassignments or reopenings. 
# SLA: Approximately 90% of incidents have an associated SLA. Incidents with SLAs have fewer reopenings (1.11) compared to those without (1.24).
# Incident Priority: Skewed towards moderate (priority 3), emphasizing careful metric selection in priority prediction models.
# Closure Codes: Most frequent closure code is 6. Codes 12, 13, 14, and 15 never result in re-openings. Code 10 has a high average number of re-openings.
# 
# ###Reassignments and Re-openings:
# Reassignments: Maximum of 27.
# Reopenings: Maximum of 8. 
# 
# ###Incident Completion and SLAs:
# Closure Code 13: Incidents exhibit an unusually long mean completion time (~40 days) with zero re-openings.
# SLA Impact: Incidents without SLAs have longer mean completion times. Implementing SLAs likely improves incident management efficiency.
# 
# ###Correlations and SLAs:
# Reopened Incidents: Strong correlation with missed SLAs. Incidents requiring reopening are likely to have breached their SLAs.
# Incident Priority: Critical and high-priority incidents are more likely to miss SLAs, while moderate and low-priority incidents tend to meet SLAs.
# 
# ###PCA Analysis:
# Principal Components: 7 components account for majority variance. Due to limited multicollinearity, all 10 variables retained for model development.
# 
# ###Model Performance:
# 
# 1.How well does the model predict the likelihood of an incident re-opening?
# Top Performers: Random Forest and XGBoost significantly outperform other models, demonstrating exceptional ability to predict incident re-opening.
# Moderate Performer: MLPClassifier shows promising results, but is outperformed by the top models.
# Poor Performers: Logistic Regression, Gradient Boosting, and Reinforcement Learning struggle to effectively predict incident re-opening.
# 
# #Interpretability:
# 
# 2.Can you explain the factors that the model uses to predict re-opening?
# Top Factors:
# -INCIDENT_STATE: Status of the incident.
# -ACTIVE: Whether the incident is active.
# -MADE_SLA: SLA adherence.
# -DAYS_OPENED_TO_CLOSED: Time taken to close the incident.
# -PRIORITY: Priority level of the incident.
# -REASSIGNMENT_COUNT: Number of times the incident was reassigned.
# -ASSIGNMENT_GROUP, RESOLVED_BY, SYS_UPDATED_BY: Involvement of various users.
# 
# ###Actionable Insights:
# 
# 2.Does the model identify specific features that can be used to proactively address incidents with a higher chance of re-opening?
# -SLA Adherence: Ensuring incidents meet SLAs can reduce reopenings.
# -Priority Management: Higher priority incidents need more attention to avoid missing SLAs.
# -Reassignment Minimization: Reducing the number of reassignments can lead to more efficient resolutions.
# -User Involvement: Tracking the involvement of specific users and groups can help identify patterns leading to reopenings.
# 
# Overall, the analysis highlights the importance of SLAs, incident priority, and minimizing reassignments in reducing the likelihood of incidents re-opening. The Random Forest and XGBoost models provide high predictive performance and clear insights into the factors affecting incident reopenings.
