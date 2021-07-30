import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("online_shoppers_intention.csv")
data = data.dropna(subset=['Administrative',
'Administrative_Duration',
'Informational',
'Informational_Duration',
'ProductRelated',
'ProductRelated_Duration',
'BounceRates',
'ExitRates'  ])

le = preprocessing.LabelEncoder()
Month = le.fit_transform(list(data['Month']))
VisitorType = le.fit_transform(list(data['VisitorType']))
Weekend = le.fit_transform(list(data['Weekend']))
Revenue = le.fit_transform(list(data['Revenue']))

dropped_columns = ['OperatingSystems', 'Browser', 'Region', 'Administrative_Duration', 'Informational_Duration','Revenue','Month','Weekend','TrafficType','ProductRelated_Duration','BounceRates','ExitRates']

data.drop(columns = 'Revenue')
data['Revenue'] = Revenue

X = data.drop(columns = dropped_columns)
X['Month'] = Month
X['VisitorType'] = VisitorType

y = data['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state= 42)

tree = RandomForestClassifier(criterion = 'entropy')
tree.fit(X_train, y_train)
acc = tree.score(X_test,y_test)
predictions = tree.predict(X_test)



