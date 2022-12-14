import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_excel("data/N3C_full_data.xlsx")

df.drop(columns=[1, '1.1'], inplace=True)
np_arr = df.values
np_arr1 = np.reshape(np_arr, (-1))
df = pd.DataFrame(np_arr1)

df[['conditions', 'age' ,'severity_covid_death', 'outcome', 'zip', 'ethnicity_concept_id', 'gender_concept_id', 'race_concept_id','trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine', 'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']] = df[0].apply(lambda x: pd.Series(str(x).split('+')))
df.drop(columns=0, inplace=True)

df.conditions = df.conditions.apply(lambda x: str(x).split(','))
df.conditions = df.conditions.apply(lambda x: list(map(int, x)))


def convert_to_int(x):
    try:
        return int(float(x))
    except ValueError:
        return 0
for col in df.columns[1:]:
    df.loc[:,col] = df.loc[:,col].apply(convert_to_int)

df.drop(columns=['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine', 'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline'], inplace=True)
# df

# Loading concept id to snomed mapping.

df_sn2con = pd.read_excel("data/snomed_to_concept_id.xlsx")
df_sn2con.drop(columns=['Unnamed: 0'], inplace=True)

np_arr = df_sn2con.values
np_arr1 = np.reshape(np_arr, (-1))
df_sn2con = pd.DataFrame(np_arr1[:-30])

# df_sn2con

df_sn2con[['snomed_id', 'condition_concept_id']] = df_sn2con[0].apply(lambda x: pd.Series(str(x).split('+')))
df_sn2con.drop(columns=0, inplace=True)
df_sn2con.snomed_id = df_sn2con.snomed_id.astype(int)
df_sn2con.condition_concept_id = df_sn2con.condition_concept_id.apply(lambda x: int(float(x)))

df_sn2con_dict = dict(zip(df_sn2con.condition_concept_id, df_sn2con.snomed_id))
# df_sn2con_dict[4262443]
# [df_sn2con_dict[x] for x in df.conditions[0]]
df['snomed_conditions'] = df.conditions.apply(lambda con: [df_sn2con_dict[x] for x in con])
# df
with open('/data1/mrahman/snomed_embeddings/line.json') as f:
    line_emb = json.load(f)
# line_emb[14265004]
line_emb = {int(k):v for k,v in line_emb.items()}

def mean_emb(snmd_con):
    p = np.array([line_emb.get(x,[0]*128) for x in snmd_con])
    p = p[~np.all(p == 0, axis=1)]
    p = p.mean(axis=0)
    return p

embaded_snmd = df.snomed_conditions.apply(mean_emb)
embaded_snmd
embaded_snmd.isna().sum()
embaded_snmd_line_arr = np.array(embaded_snmd.values.tolist())
embaded_snmd_line_arr = np.nan_to_num(embaded_snmd_line_arr)
df
embaded_snmd_n2v_arr

#random forest

X = embaded_snmd_line_arr
Y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

embed_model_rf=RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=4)

embed_model_rf.fit(X_train,y_train)

y_pred=embed_model_rf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#logistic regression

embed_model_lr = LogisticRegression(max_iter=1000 ,C=1, verbose=1, n_jobs=-1).fit(X_train,y_train)

y_pred=embed_model_lr.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#SVM
embed_model_svm= svm.SVC(kernel='linear', verbose=1, max_iter=20000)
embed_model_svm.fit(X_train,y_train)

y_pred=embed_model_svm.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Neural Network

model = Sequential()
model.add(Dense(12, input_dim=128, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=64, batch_size=1000)
model.evaluate(X_test, y_test)
y_pred=model.predict(X_test).round()

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",acc)
precision = metrics.precision_score(y_test, y_pred)
print("Precision:",precision)
recall = metrics.recall_score(y_test, y_pred)
print("Recall:",recall)