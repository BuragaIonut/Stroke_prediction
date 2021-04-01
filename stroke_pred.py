import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)

data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# print(data.describe())
# print(data.isnull().sum())

data['bmi'].fillna(data['bmi'].mean(), inplace=True)

# print(data['bmi'].isna().sum())

label_encoder = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category')
        data[col] = label_encoder.fit_transform(data[col])

# print(data.info())

label = data['stroke']
features = data.drop('stroke', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.33)

clf = LogisticRegression()

clf.fit(X_train, y_train)

predicted = pd.DataFrame(clf.predict(X_test))
y_test = pd.DataFrame(y_test)

print(f'The accuracy:{100*accuracy_score(y_test, predicted):.2f}%')


y_test.reset_index(drop=True, inplace=True)
# print(len(y_test))
# print(len(predicted))
# for i in range(len(y_test)):
#     if y_test.iloc[i, 0] != predicted.iloc[i, 0]:
#         print(f'{y_test.iloc[i, 0]} != {predicted.iloc[i, 0]}')
