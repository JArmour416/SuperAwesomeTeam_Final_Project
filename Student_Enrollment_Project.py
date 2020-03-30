import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

# Obtain and combine database
def read_csv():

    global data

    leads = pd.read_csv('data/leads.csv')

    opps = pd.read_csv('data/opps.csv')

    opps['Opportunity'].fillna(value=1, inplace=True)

    student = pd.concat([opps, leads])

    student.drop_duplicates(subset=['Id'])
    
    data = student.iloc[:10000]
    
    return None

# clean the dataset

def cleandata():
    # Convert the SFDC Campaigns to yes or no
    data['SFDC Campaigns'].fillna(value=0, inplace=True)
    data['From SFDC Campaigns'] = np.where(data['SFDC Campaigns'] == 0, 0, 1)
    # Convert 'City of Event' to yes or no
    data['City of Event'].fillna(value=0, inplace=True)
    data['Attended Event'] = np.where(data['City of Event'] == 0, 0, 1)
    # convert birth date to age
    time_value = pd.to_datetime(data['Birth Date'], format='%Y-%m-%d')
    time_value = pd.DatetimeIndex(time_value)
    data['Age'] = 2020 - time_value.year
    data['Age'].fillna(value=data['Age'].mean(), inplace=True)
    # clearn all the features we need for machine learening
    data['Unsubscribed'].fillna(value=0, inplace=True)
    data['Person Score'].fillna(value=0, inplace=True)
    data['Behavior Score'].fillna(value=0, inplace=True)
    data['Media SubGroup'].fillna(value=0, inplace=True)
    data['Address Country'].fillna(value=0, inplace=True)
    data['Primary Program'].fillna(value=0, inplace=True)
    data['Engagement'].fillna(value=0, inplace=True)
    data['Opportunity'].fillna(value=0, inplace=True)
    return None

# set up data preprocessing funtions
def im(x):

    im = SimpleImputer(missing_values=np.nan, strategy='mean')

    data = im.fit_transform(x)

    print(data)

    return None

def pca(x):

    pca = PCA(n_components=0.9)

    data = pca.fit_transform(x)

    print(data)

    return None

def var(x):
    var = VarianceThreshold(threshold=0.0)

    data = var.fit_transform()

    print(data)

    return None


def mm(x):
    mm = MinMaxScaler()

    data = mm.fit_transform(x)

    print(data)

    return None

def stand(x):

    std = StandardScaler()

    data = std.fit_transform(x)

    print(data)

    return None

def dict(x):
    dict = DictVectorizer(sparse=False) 

    data = dict.fit_transform(x)

    print(data)

    return None

# Prepare the dataframe for machine learning (we could try different combination, and we will use the following one to cut the calculation waitting time)
# Data Preprocessing

def traintest_split():
    global x_train, x_test, y_train, y_test
    df = data[['Media SubGroup', 'Primary Program', 'Unsubscribed', 'Attended Event', 'Opportunity']]
    y = df['Opportunity']
    x = df.drop(axis=1, columns=['Opportunity'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return None

# use knn model
def knn():

    knn = KNeighborsClassifier()

    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))

    x_test = dict.transform(x_test.to_dict(orient='records'))

    knn.fit(x_train, y_train)

    score = knn.score(x_test, y_test)

    # using GridSearchCV 
    param = {'n_neighbors': [5, 10, 50, 100, 500]}

    gc = GridSearchCV(knn, param_grid=param, cv=2)

    gc.fit(x_train, y_train)

    gcscore = gc.score(x_test, y_test)

    parameter = gc.best_params_

    # y_predict = knn.predict(x_test)
    # y_predict

    print(f'the best score for knn model is {gcscore} and the best parameter is {parameter}')

    return None

# Using decision tree

def dec():

    dec = DecisionTreeClassifier()

    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))

    x_test = dict.transform(x_test.to_dict(orient='records'))

    dec.fit(x_train, y_train)

    score = dec.score(x_test, y_test)

    export_graphviz(dec, out_file='tree.dot')

    print(f'the score for dec model is {score}')

    return None

# Using Random Forest

def rf():

    rf = RandomForestClassifier()

    dict = DictVectorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient='records'))

    x_test = dict.transform(x_test.to_dict(orient='records'))

    rf.fit(x_train, y_train)

    score = rf.score(x_test, y_test)

    # using GridSearchCV to evalue the result
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 25, 30]}

    GC = GridSearchCV(rf, param_grid=param, cv=2)

    GC.fit(x_train, y_train)

    GCscore = GC.score(x_test, y_test)

    parameter = GC.best_params_

    print(f'the best score Random Forest model is {GCscore} and the best parameter is {parameter}')

    return None

if __name__ == "__main__":
    cleandata()
    traintest_split()
    knn()
    dec()
    rf()
