import pandas as pd
import numpy as np
import itertools
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import transformers
from tqdm.auto import tqdm
print(transformers.__version__)

from transformers import BertModel

df = pd.read_excel("data/sample_embed.xlsx")
df.drop(columns=['Unnamed: 0'], inplace=True)

np_arr = df.values
np_arr1 = np.reshape(np_arr, (-1))

df1 = pd.DataFrame(np_arr1)
cols = ['person_id', 'conditions']

df1[cols] = df1[0].apply(lambda x: pd.Series(str(x).split('+')))
df1.drop(columns=0, inplace=True)
df1.to_csv("sample_embed.csv")

df1.person_id = df1.person_id.apply(lambda x: int(x))
df1.conditions = df1.conditions.apply(lambda x: str(x).split(','))
df1.conditions = df1.conditions.apply(lambda x: list(map(int, x)))

token_conditions = list(itertools.chain.from_iterable(df1.conditions.values))
token_conditions_str = [str(x) for x in token_conditions]
# token_conditions_str = ' '.join(token_conditions_str)

df1

tkn = Tokenizer()
tkn.fit_on_texts(token_conditions_str)


def c2tk(cell):
    tk = [str(x) for x in cell]
    tk = tkn.texts_to_sequences(tk)
    tk = list(np.array(tk).reshape(-1))
    return tk

df1["condition_tokens"] = df1.conditions.apply(c2tk)
df1["visit_occ"] = df1.condition_tokens.apply(lambda x: np.ones(len(x),dtype=int))
df1["condition_tokens"]
unique_person = df1.person_id.unique()


for id in unique_person:
    per_visit_occ = df1.loc[df1['person_id']==id, 'visit_occ']
    df1.loc[df1['person_id']==id, 'visit_occ'] = per_visit_occ + np.arange(len(per_visit_occ))


df1.drop(columns='conditions', inplace=True)
df1['condition_tokens'] = df1['condition_tokens'].apply(lambda x: list(map(int, x)))
df1['visit_occ'] = df1['visit_occ'].apply(lambda x: list(map(int, x)))

per_condition_token =df1.groupby(by='person_id').agg({'condition_tokens':'sum' ,'visit_occ':'sum'})

per_condition_token 

train_test_split(per_condition_token, shuffle=True)

model_name = "bert-base-uncased"

df1















