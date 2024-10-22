#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Supress unnecessary warnings

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Import the NumPy and Pandas packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[5]:


# Read the dataset
leads = pd.read_csv(r'C:\Users\swana\Downloads\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv')
leads.head()


# In[6]:


# Look at the first few entries
leads.head()


# In[7]:


# Inspect the shape of the dataset
leads.shape


# In[8]:


# Inspect the different columsn in the dataset
print(leads.columns)
# columns attribute


# As you can see, the feature variables are quite intuitive. If you don't understand them completely, please refer to the data dictionary.

# In[10]:


# Check the summary of the dataset
leads.info()
# describe
print(leads.describe())


# In[12]:


# Check the info to see the types of the feature variables and the null values present
missing_values = leads.isnull().sum()
print(missing_values[missing_values > 0])
# info


# Looks like there are quite a few categorical variables present in this dataset for which we will need to create dummy variables. Also, there are a lot of null values present as well, so we will need to treat them accordingly.

# ## Step 1: Data Cleaning and Preparation

# In[15]:


# sort_values descending order
missing_values_sorted = leads.isnull().sum().sort_values(ascending=False)
print(missing_values_sorted)


# As you can see there are a lot of columns which have high number of missing values. Clearly, these columns are not useful. Since, there are 9000 datapoints in our dataframe, let's eliminate the columns having greater than 3000 missing values as they are of no use to us.

# In[16]:


# Drop all the columns in which greater than 3000 missing values are present
# This is just my approach, you can choose to other approaches as well: dropping columns with missing values greater than a
#  certain threshold percentage, imputing the missing values, etc.

for col in leads.columns:
    if leads[col].isna().sum() > 3000:
        leads.drop(col, axis=1, inplace=True)


# In[17]:


# Check the number of null values again
missing_values_sorted_after = leads.isnull().sum().sort_values(ascending=False)
print(missing_values_sorted_after)


# As you might be able to interpret, the variable `City` won't be of any use in our analysis. So it's best that we drop it.

# In[18]:


# drop City
leads.drop('City', axis=1, inplace=True)


# In[19]:


# Same goes for the variable 'Country'
leads.drop('Country', axis=1, inplace=True)
# drop Country


# In[22]:


# Let's now check the percentage of missing values in each column
missing_percentage = (leads.isnull().sum() / leads.shape[0]) * 100
missing_percentage_rounded = missing_percentage.round(2)
missing_percentage_sorted = missing_percentage_rounded.sort_values(ascending=False)
print(missing_percentage_sorted)


# In[23]:


# Check the number of null values again

missing_values_sorted_after = leads.isnull().sum().sort_values(ascending=False)
print(missing_values_sorted_after)


# Now recall that there are a few columns in which there is a level called 'Select' which basically means that the student had not selected the option for that particular column which is why it shows 'Select'. These values are as good as missing values and hence we need to identify the value counts of the level 'Select' in all the columns that it is present.

# In[25]:


# Get the value counts of all the columns
for column in leads.columns:
    print(f"\nValue counts for '{column}':")
    print(leads[column].value_counts())


# The following three columns now have the level 'Select'. Let's check them once again.

# In[28]:


# apply v_c() on Lead Profile col
def v_c(series):
    return series.value_counts()

lead_profile_counts = v_c(leads['Lead Profile'])
print(lead_profile_counts)


# In[49]:


# drop Lead Profile and How did you hear about X Education cols
leads.drop('Lead Profile', axis=1, inplace=True)


# In[50]:


print(leads.columns.tolist())


# Clearly the levels `Lead Profile` and `How did you hear about X Education` have a lot of rows which have the value `Select` which is of no use to the analysis so it's best that we drop them.

# In[ ]:


leads.drop('How did you hear about X Education', axis=1, inplace=True)


# In[57]:


print(leads.columns.tolist())


# Also notice that when you got the value counts of all the columns, there were a few columns in which only one value was majorly present for all the data points. These include `Do Not Call`, `Search`, `Magazine`, `Newspaper Article`, `X Education Forums`, `Newspaper`, `Digital Advertisement`, `Through Recommendations`, `Receive More Updates About Our Courses`, `Update me on Supply Chain Content`, `Get updates on DM Content`, `I agree to pay the amount through cheque`. Since practically all of the values for these variables are `No`, it's best that we drop these columns as they won't help with our analysis.

# In[71]:


# drop the above mentioned columns
# Define columns to drop
columns_to_drop = [
    'Lead Profile',
    'How did you hear about X Education',
    'Do Not Call',
    'Search',
    'Magazine',
    'Newspaper Article',
    'X Education Forums',
    'Newspaper',
    'Digital Advertisement',
    'Through Recommendations',
    'Receive More Updates About Our Courses',
    'Update me on Supply Chain Content',
    'Get updates on DM Content',
    'I agree to pay the amount through cheque'
]


# In[74]:


# Check for existing columns in the DataFrame before dropping
existing_columns_to_drop = [col for col in columns_to_drop if col in leads.columns]
# Drop the existing columns
if existing_columns_to_drop:
    leads.drop(existing_columns_to_drop, axis=1, inplace=True)
    print("\nDropped columns:", existing_columns_to_drop)
else:
    print("\nNo columns were dropped. Check if the names are correct.")
    
print(leads.columns.tolist())


# In[ ]:





# In[ ]:





# In[ ]:





# Also, the variable `What matters most to you in choosing a course` has the level `Better Career Prospects` `6528` times while the other two levels appear once twice and once respectively. So we should drop this column as well.

# In[80]:


# v_c() on "Name: What matters most to you in choosing a course"
column_name = "What matters most to you in choosing a course"
if column_name in leads.columns:
    value_counts = v_c(leads[column_name])
    print(f"\nValue counts for '{column_name}':\n", value_counts)
else:
    print(f"\nColumn '{column_name}' does not exist in the DataFrame.")
    print(leads.columns.tolist())


# In[86]:


# Drop the null value rows present in the variable 'What matters most to you in choosing a course'
column_name = "What matters most to you in choosing a course"
leads= leads.dropna(subset=[column_name])
# Check the shape of the DataFrame after dropping null values
print(f"\nShape of DataFrame after dropping null values in '{column_name}':", leads.shape)


# In[91]:


# Check the number of null values again

leads.isnull().sum().sort_values(ascending=False)


# Now, there's the column `What is your current occupation` which has a lot of null values. Now you can drop the entire row but since we have already lost so many feature variables, we choose not to drop it as it might turn out to be significant in the analysis. So let's just drop the null rows for the column `What is you current occupation`.

# In[90]:


leads = leads[ ~pd.isnull(leads['What is your current occupation']) ]


# In[94]:


# Check the number of null values again
null_values = leads.isnull().sum().sort_values(ascending=False)


# Since now the number of null values present in the columns are quite small we can simply drop the rows in which these null values are present.

# In[95]:


# Drop the null value rows in the column 'TotalVisits'
leads = leads[ ~pd.isnull(leads['TotalVisits']) ]


# In[96]:


# Check the null values again

leads.isnull().sum().sort_values(ascending=False)


# In[101]:


# Drop the null values rows in the column 'Lead Source'
leads = leads.dropna(subset=['Lead Source'])
print(f"\nShape of DataFrame after dropping null values in 'Lead Source':", leads.shape)
null_values = leads.isnull().sum().sort_values(ascending=False)
# Display the result
print(null_values)


# In[103]:


# Check the number of null values again

leads.isnull().sum().sort_values(ascending=False)


# In[31]:





# In[ ]:





# Now your data doesn't have any null values. Let's now check the percentage of rows that we have retained.

# In[105]:


print(len(leads.index))
print(len(leads.index)/9240)


# We still have around 69% of the rows which seems good enough.

# In[106]:


# Let's look at the dataset again

leads.head()


# Now, clearly the variables `Prospect ID` and `Lead Number` won't be of any use in the analysis, so it's best that we drop these two variables.

# In[109]:


# Drop the columns 'Prospect ID' and 'Lead Number'
leads.drop(['Prospect ID', 'Lead Number'], axis=1, inplace=True)

# Check the shape of the DataFrame after dropping the columns
print(f"\nShape of DataFrame after dropping 'Prospect ID' and 'Lead Number':", leads.shape)

# Display the remaining columns
print("\nRemaining columns in the dataset:", leads.columns.tolist())


# In[110]:


leads.head()


# ### Dummy variable creation
# 
# The next step is to deal with the categorical variables present in the dataset. So first take a look at which variables are actually categorical variables.

# In[111]:


# Check the columns which are of type 'object'
temp = leads.loc[:, leads.dtypes == 'object']
temp.columns


# In[112]:


# Demo Cell
df = pd.DataFrame({'P': ['p', 'q', 'p']})
df


# In[113]:


pd.get_dummies(df)


# In[114]:


pd.get_dummies(df, prefix=['col1'])


# In[ ]:


# Create dummy variables using the 'get_dummies' command
dummy = pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                              'What is your current occupation','A free copy of Mastering The Interview', 
                              'Last Notable Activity']], drop_first=True, dtype=int)

# Add the results to the master dataframe
leads = pd.concat([leads, dummy], axis=1)


# In[ ]:


# Creating dummy variable separately for the variable 'Specialization' since it has the level 'Select' 
# which is useless so we
# drop that level by specifying it explicitly

dummy_spl = pd.get_dummies(leads['Specialization'], dtype=int)
dummy_spl = dummy_spl.drop(['Specialization'], 1)
leads = pd.concat([leads, dummy_spl], axis = 1)


# In[ ]:


# Drop the variables for which the dummy variables have been created

leads = leads.drop(['Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
                   'Specialization', 'What is your current occupation',
                   'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)


# In[122]:


# Let's take a look at the dataset again

leads.head()


# ### Test-Train Split
# 
# The next step is to split the dataset into training an testing sets.

# In[124]:


# Import the required library

from sklearn.model_selection import train_test_split


# In[125]:


# Put all the feature variables in X

X = leads.drop('Converted', axis=1)
X.head()


# In[126]:


# Put the target variable in y

y = leads['Converted']
y.head()


# In[127]:


# Split the dataset into 70% train and 30% test, and set the random state to 100

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# Check the shape of the train dataset and the test dataset
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Scaling
# 
# Now there are a few numeric variables present in the dataset which have different scales. So let's go ahead and scale these variables.

# In[128]:


# Import MinMax scaler

from sklearn.preprocessing import MinMaxScaler


# In[132]:


# Scale the three numeric features present in the dataset
scaler = MinMaxScaler()
num_vars = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
# Apply it on columns: TotalVisits,	Total Time Spent on Website,	Page Views Per Visit
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.transform(X_test[num_vars])
# Check the scaled data
print(X_train.head())
print(X_test.head())


# ### Looking at the correlations
# 
# Let's now look at the correlations. Since the number of variables are pretty high, it's better that we look at the table instead of plotting a heatmap

# In[139]:


# Looking at the correlation table
# Compute the correlation matrix
correlation_matrix = X_train.corr()

# Display the correlation matrix
print(correlation_matrix)


# In[156]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


# ## Step 2: Model Building
# 
# Let's now move to model building. As you can see that there are a lot of variables present in the dataset which we cannot deal with. So the best way to approach this is to select a small set of features from this pool of variables using RFE.

# In[149]:


# Import 'LogisticRegression' and create a LogisticRegression object "logreg"
# Create a Logistic Regression object
print(X_train.columns)  # This will show you all columns in X_train


# In[152]:


print(leads.columns)


# In[166]:


categorical_columns = [
    'Lead Origin', 'Lead Source', 'Do Not Email', 'Last Activity',
    'What is your current occupation', 'A free copy of Mastering The Interview',
    'Last Notable Activity', 'Specialization', 'What matters most to you in choosing a course'
]

# Check which columns are present in the DataFrame
categorical_columns = [col for col in categorical_columns if col in leads.columns]


# In[170]:


# Print the categorical columns to ensure there are valid columns
print("Categorical columns present in the DataFrame:", categorical_columns)

# Proceed only if there are categorical columns left
if categorical_columns:
    # Create dummy variables for the existing categorical columns
    dummy_vars = pd.get_dummies(leads[categorical_columns], drop_first=True, dtype=int)

    # Add dummy variables to the leads DataFrame
    leads = pd.concat([leads, dummy_vars], axis=1)

    # Drop the original categorical columns
    leads.drop(columns=categorical_columns, axis=1, inplace=True)
else:
    print("No categorical columns to process.")


# In[202]:


# Now, define X and y
X = leads.drop('Converted', axis=1)
y = leads['Converted']

# Split the dataset into training and testing sets again
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

# Use RFE to select top 15 features
logreg = LogisticRegression(solver='liblinear', random_state=100)
rfe = RFE(estimator=logreg, n_features_to_select=15)
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfe.support_]
print("Selected features by RFE:", selected_features)


# In[203]:


# Let's take a look at which features have been selected by RFE
selected_features_info = list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[ ]:





# Now you have all the variables selected by RFE and since we care about the statistics part, i.e. the p-values and the VIFs, let's use these variables to create a logistic regression model using statsmodels.

# In[204]:


X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Check the shape of the new X_train_selected and X_test_selected
print("Shape of X_train_selected:", X_train_selected.shape)
print("Shape of X_test_selected:", X_test_selected.shape)


# In[205]:


# Import statsmodels

import statsmodels.api as sm


# In[206]:


# Fit a logistic Regression model on X_train after adding a constant and output the summary

X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# There are quite a few variable which have a p-value greater than `0.05`. We will need to take care of them. But first, let's also look at the VIFs.

# In[207]:


# Import 'variance_inflation_factor'

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[209]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train_selected.columns  # Use the selected features
vif['VIF'] = [variance_inflation_factor(X_train_selected.values, i) for i in range(X_train_selected.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)  # Round VIF values to 2 decimal places
vif = vif.sort_values(by="VIF", ascending=False)  # Sort VIF in descending order

# Display the VIF dataframe
print(vif)


# VIFs seem to be in a decent range except for three variables. 
# 
# Let's first drop the variable `Lead Origin_Lead Add Form` since it has a high p-value as well as a high VIF.

# In[210]:


selected_columns = X_train.columns[rfe.support_]

X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]

# Now, drop the 'Lead Source_Reference' column if it exists
if 'Lead Source_Reference' in X_train_selected.columns:
    X_train_selected = X_train_selected.drop('Lead Source_Reference', axis=1)
    X_test_selected = X_test_selected.drop('Lead Source_Reference', axis=1)

# Check the shape after dropping the column
print(X_train_selected.shape)
print(X_test_selected.shape)


# In[211]:


# Refit the model with the new set of features

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# The variable `Lead Profile_Dual Specialization Student	` also needs to be dropped.

# In[212]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train_selected.columns  # Use the columns from the updated DataFrame
vif['VIF'] = [variance_inflation_factor(X_train_selected.values, i) for i in range(X_train_selected.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)  # Round VIF values to 2 decimal places
vif = vif.sort_values(by="VIF", ascending=False)  # Sort VIF in descending order

# Display the VIF dataframe
print(vif)


# The VIFs are now all less than 5. So let's drop the ones with the high p-values beginning with `Last Notable Activity_Had a Phone Conversation`.

# In[213]:


X_train.drop('Last Notable Activity_Had a Phone Conversation', axis = 1, inplace = True)


# In[214]:


# Refit the model with the new set of features
logm1_updated = sm.GLM(y_train, sm.add_constant(X_train_selected), family=sm.families.Binomial())
result_updated = logm1_updated.fit()

# Display the summary of the updated model
print(result_updated.summary())


# Drop `What is your current occupation_Housewife`.

# In[215]:


X_train_selected = X_train_selected.drop('What is your current occupation_Housewife', axis=1)


# In[216]:


# Refit the model with the updated set of features
logm1_updated = sm.GLM(y_train, sm.add_constant(X_train_selected), family=sm.families.Binomial())
result_updated = logm1_updated.fit()

# Display the summary of the updated model
print(result_updated.summary())


# Drop `What is your current occupation_Working Professional`.

# In[217]:


X_train_selected = X_train_selected.drop('What is your current occupation_Working Professional', axis=1)


# In[218]:


# Refit the model with the updated set of features
logm1_updated = sm.GLM(y_train, sm.add_constant(X_train_selected), family=sm.families.Binomial())
result_updated = logm1_updated.fit()

# Display the summary of the updated model
print(result_updated.summary())


# All the p-values are now in the appropriate range. Let's also check the VIFs again in case we had missed something.

# In[219]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train_selected.columns
vif['VIF'] = [variance_inflation_factor(X_train_selected.values, i) for i in range(X_train_selected.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending=False)

# Display the VIF DataFrame
print(vif)


# We are good to go!

# ## Step 3: Model Evaluation
# 
# Now, both the p-values and VIFs seem decent enough for all the variables. So let's go ahead and make predictions using this final set of features.

# In[226]:


# Use 'predict' to predict the probabilities on the train set
logm1 = sm.GLM(y_train, sm.add_constant(X_train_selected), family=sm.families.Binomial())
res = logm1.fit()


# In[227]:


y_train_pred = res.predict(sm.add_constant(X_train_selected))

# Display the first 10 predicted probabilities
print("Predicted probabilities for the first 10 instances in the training set:")
print(y_train_pred[:10])


# #### Creating a dataframe with the actual conversion flag and the predicted probabilities

# In[228]:


# Create a new dataframe containing the actual conversion flag and the probabilities predicted by the model

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# #### Creating new column 'Predicted' with 1 if Paid_Prob > 0.5 else 0

# In[229]:


# Creating new column 'Predicted' with 1 if Paid_Prob > 0.5 else 0
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# Now that you have the probabilities and have also made conversion predictions using them, it's time to evaluate the model.

# In[231]:


# Import metrics from sklearn for evaluation

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[245]:


# Create confusion matrix 
confusion = confusion_matrix(y_train_pred_final['Converted'], y_train_pred_final['Predicted'])

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion)


# In[77]:


# Predicted     not_churn    churn
# Actual
# not_churn        2543      463
# churn            692       1652  


# In[246]:


# Let's check the overall accuracy

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[247]:


# Let's evaluate the other metrics as well

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[248]:


# Calculate the sensitivity

TP/(TP+FN)


# In[249]:


# Calculate the specificity

TN/(TN+FP)


# ### Finding the Optimal Cutoff
# 
# Now 0.5 was just arbitrary to loosely check the model performace. But in order to get good results, you need to optimise the threshold. So first let's plot an ROC curve to see what AUC we get.

# In[250]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[252]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, 
                                         y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[253]:


# Import matplotlib to plot the ROC curve

import matplotlib.pyplot as plt


# In[254]:


# Call the ROC function

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# The area under the curve of the ROC is 0.86 which is quite good. So we seem to have a good model. Let's also check the sensitivity and specificity tradeoff to find the optimal cutoff point.

# In[255]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[256]:


# Let's create a dataframe to see the values of accuracy, sensitivity, and specificity at 
# different values of probabiity cutoffs

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[257]:


# Let's plot it as well

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# As you can see that around 0.42, you get the optimal values of the three metrics. So let's choose 0.42 as our cutoff now.

# In[258]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.42 else 0)

y_train_pred_final.head()


# In[259]:


# Let's check the accuracy now

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[262]:


# Let's create the confusion matrix once again
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
print("Confusion Matrix:\n", confusion2)


# In[263]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[264]:


# Calculate Sensitivity

TP/(TP+FN)


# In[265]:


# Calculate Specificity

TN/(TN+FP)


# This cutoff point seems good to go!

# ## Step 4: Making Predictions on the Test Set
# 
# Let's now make predicitons on the test set.

# In[ ]:



# Scale the test set using just 'transform'
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = (
    scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
)


# In[344]:


# Select the columns in X_train for X_test as well
 
print("Columns in X_test:", X_test.columns.tolist())
selected_columns = [col for col in selected_columns if col in X_test.columns]
X_test_selected = X_test[selected_columns]

# Define X and y
X = leads.drop('Converted', axis=1)
y = leads['Converted']

columns_to_drop = ['Lead Source_Reference', 'Last Notable Activity_Had a Phone Conversation']
X = X.drop(columns=columns_to_drop, errors='ignore') 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
logreg = LogisticRegression(solver='liblinear', random_state=100)
rfe = RFE(estimator=logreg, n_features_to_select=15)
rfe.fit(X_train, y_train)

# Get the selected features
selected_columns = X_train.columns[rfe.support_]
print("Selected features by RFE:", selected_columns)

selected_columns = [col for col in selected_columns if col in X_test.columns]

X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns] 

print("Shape of X_train_selected:", X_train_selected.shape)
print("Shape of X_test_selected:", X_test_selected.shape)

X_test_selected.head()


# In[350]:


# Add a constant to X_test

columns_to_drop = ['Lead Source_Reference', 'Last Notable Activity_Had a Phone Conversation']
X = X.drop(columns=columns_to_drop, errors='ignore') 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

logreg = LogisticRegression(solver='liblinear', random_state=100)
rfe = RFE(estimator=logreg, n_features_to_select=15)
rfe.fit(X_train, y_train)

# Get the selected features
selected_columns = X_train.columns[rfe.support_]
print("Selected features by RFE:", selected_columns)

# Select the columns in X_test for consistency
selected_columns = [col for col in selected_columns if col in X_test.columns]
X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]

# Check the shapes of the selected data
print("Shape of X_train_selected:", X_train_selected.shape)
print("Shape of X_test_selected:", X_test_selected.shape)

# Add a constant to the selected X_test
X_test_sm = sm.add_constant(X_test_selected)

# Display the first few rows of the prepared test set
print(X_test_sm.head())


# In[351]:


# Check X_test_sm

X_test_sm


# In[ ]:


# Drop the required columns from X_test as well

X_test.drop(['Lead Source_Reference', 'What is your current occupation_Housewife', 
             'What is your current occupation_Working Professional', 
                     'Last Notable Activity_Had a Phone Conversation'], 1, 
                                inplace = True)


# In[360]:


# Make predictions on the test set and store it in the variable 'y_test_pred'

# Fit the model with selected features
logreg = LogisticRegression(solver='liblinear', random_state=100)
logreg.fit(X_train_selected, y_train)

# Store the fitted model in 'res'
res = logreg

# Check the shape of the training and test sets before adding a constant
print("Shape of X_train_selected:", X_train_selected.shape)
print("Shape of X_test_selected:", X_test_selected.shape)

# Add a constant to X_test_selected
X_test_selected_sm = sm.add_constant(X_test_selected)

# Check the shape after adding the constant
print("Shape of X_test_selected_sm after adding constant:", X_test_selected_sm.shape)





# In[365]:


# Make predictions using the selected test set with the constant
y_test_pred = res.predict(X_test_selected_sm)


# In[ ]:


y_test_pred[:10]


# In[102]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[103]:


# Let's see the head

y_pred_1.head()


# In[367]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[105]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[106]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[107]:


# Check 'y_pred_final'

y_pred_final.head()


# In[108]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[109]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[110]:


# Make predictions on the test set using 0.45 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)


# In[111]:


# Check y_pred_final

y_pred_final.head()


# In[112]:


# Let's check the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[113]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[114]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[115]:


# Calculate sensitivity
TP / float(TP+FN)


# In[116]:


# Calculate specificity
TN / float(TN+FP)


#  

#  

#  

#  

#  

# ## Precision-Recall View
# 
# Let's now also build the training model using the precision-recall view

# In[117]:


#Looking at the confusion matrix again


# In[118]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# ##### Precision
# TP / TP + FP

# In[119]:


confusion[1,1]/(confusion[0,1]+confusion[1,1])


# ##### Recall
# TP / TP + FN

# In[120]:


confusion[1,1]/(confusion[1,0]+confusion[1,1])


# ### Precision and recall tradeoff

# In[121]:


from sklearn.metrics import precision_recall_curve


# In[122]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[123]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[124]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[125]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.44 else 0)

y_train_pred_final.head()


# In[126]:


# Let's check the accuracy now

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[368]:


# Let's create the confusion matrix once again

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[128]:


# Let's evaluate the other metrics as well

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[129]:


# Calculate Precision

TP/(TP+FP)


# In[130]:


# Calculate Recall

TP/(TP+FN)


# This cutoff point seems good to go!

# ## Step 4: Making Predictions on the Test Set
# 
# Let's now make predicitons on the test set.

# In[314]:


# Check if X_test has the same features used in the training set before adding the constant
print("Columns in X_test:", X_test.columns)

# Ensure X_test has been prepared similarly to X_train
X_test_selected = X_test[selected_features]  # Use the same features selected during training
# Add a constant to X_test_selected
X_test_sm = sm.add_constant(X_test_selected)
# Make predictions on the test set and store it in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)


# In[132]:


y_test_pred[:10]


# In[133]:


# Converting y_pred to a dataframe

y_pred_1 = pd.DataFrame(y_test_pred)


# In[134]:


# Let's see the head

y_pred_1.head()


# In[135]:


# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)


# In[136]:


# Remove index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[137]:


# Append y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[138]:


# Check 'y_pred_final'

y_pred_final.head()


# In[139]:


# Rename the column 

y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})


# In[140]:


# Let's see the head of y_pred_final

y_pred_final.head()


# In[141]:


# Make predictions on the test set using 0.44 as the cutoff

y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.44 else 0)


# In[142]:


# Check y_pred_final

y_pred_final.head()


# In[143]:


# Let's check the overall accuracy

metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[144]:


confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[145]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[146]:


# Calculate Precision

TP/(TP+FP)


# In[147]:


# Calculate Recall

TP/(TP+FN)


# # Happy Learning

# In[ ]:


# 1.	Which are the top three variables in your model which contribute most towards the probability of a lead getting converted?

# >TotalVisits, Total Time spent on the website, Lead Origin_Lead Add Form	 

# 2.	What are the top 3 categorical/dummy variables in the model which should be focused the most on in order 
# to increase the probability of lead conversion?
# Lead Origin_Lead Add Form, Last Activity_Had a phone conversation, Lead Score_Wellingak Website
# Last Notable Activity_Unrechable -> no business sense as it seems like a small flaw in the model.

# 3. 3.	X Education has a period of 2 months every year during which they hire some interns. 
# The sales team, in particular, has around 10 interns allotted to them. 
# So during this phase, they wish to make the lead conversion more aggressive. So they want almost all of
#  the potential leads (i.e. the customers who have been predicted as 1 by the model) to be converted and hence, 
# want to make phone calls to as much of such people as possible. Suggest a good strategy they should employ at this stage.


# ![image.png](attachment:image.png)

# In[ ]:


#1.	Similarly, at times, the company reaches its target for a quarter before the deadline. 
# During this time, the company wants the sales team to focus on some new work as well. So during this t
# ime, the company’s aim is to not make phone calls unless it’s extremely necessary, i.e. they want to
#  minimize the rate of useless phone calls. Suggest a strategy they should employ at this stage.

