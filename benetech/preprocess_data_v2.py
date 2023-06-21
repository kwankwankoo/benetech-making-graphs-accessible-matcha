import pandas as pd
from glob import  glob
import json
from tqdm import trange, tqdm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

label_path_500k = '/home/gumingjun/benetech-making-graphs-accessible-matcha/benetech/datasets/500k/metadata.csv'
df_500k = pd.read_csv(label_path_500k)
# print(df_500k)

relative_path = './datasets/500k/'
df_500k['file_name'] = df_500k['file_name'].apply(lambda x: os.path.join(relative_path, x))
df_500k= df_500k.drop('count', axis=1)
df_500k = df_500k.rename(columns={'chart_type': 'types'})
df_500k = df_500k.rename(columns={'file_name': 'image_path'})
df_500k = df_500k.rename(columns={'text': 'label'})
# print(df_500k)

df_500k = df_500k.drop('validation', axis=1)
# print(df_500k)

def transform_labels(label):
    parts = label.split("<0x0A>")
    numerical_values = []
    text_values = []
    for part in parts:
        numerical_value = part.split("|")[0].strip()
        text_value = part.split("|")[1].strip()
        numerical_values.append(numerical_value)
        text_values.append(text_value)
    numerical_values_str = ",".join(numerical_values)
    text_values_str = ",".join(text_values)
    return f"<x_start>{numerical_values_str}<x_end><y_start>{text_values_str}<y_end>"

# 对DataFrame中的labels列应用转换函数
df_500k['label'] = df_500k['label'].apply(transform_labels)
# print(df_500k["label"][2])

type_counts = df_500k['types'].value_counts()
sample_size = type_counts // 10
sample_size
df_500k_1 = pd.DataFrame(columns=df_500k.columns)
for type_, size in sample_size.items():
    samples = df_500k[df_500k['types'] == type_].sample(n=size)
    df_500k_1 = df_500k_1.append(samples)
# print(df_500k_1)

type_counts = df_500k_1['types'].value_counts()
# print(type_counts)

label_path = glob('./datasets/default/train/annotations/*')
# print(label_path)

print(len(label_path))
image_path = glob('./datasets/default/train/images/*')
print(len(label_path))

image_path = []
gts = []
types = []
for i in  tqdm(label_path):
    with open(i) as f:
        data = json.loads(f.read())
    image_path.append(i.replace('annotations', 'images').replace('json', 'jpg'))
    gts.append(data['data-series'])
    types.append(data['chart-type'])

df = pd.DataFrame({'image_path':image_path,
                   'gts':gts,
                   'types':types
                   })
# print(df)

import re
def preprocess_gts(input):
    output = '<x_start>'
    xs = []
    ys = []
    for i in input:
        xs.append(str(i['x']))
        ys.append(str(i['y']))
    output += ';'.join(xs) + '<x_end> <y_start>' + ';'.join(ys) + '<y_end>'
    
#     return output
    x_pattern = r"<x_start>(.*?)<x_end>"
    y_pattern = r"<y_start>(.*?)<y_end>"

    x_match = re.search(x_pattern, output)
    y_match = re.search(y_pattern, output)
    
    
    deplot_output = ""
    
    if x_match and y_match:
        x_values = x_match.group(1).split(";")
        y_values = y_match.group(1).split(";")

        
        for x, y in zip(x_values, y_values):
            deplot_output += f"{x.strip()} | {y.strip()} <0x0A> "
        
        deplot_output = deplot_output.rstrip("<0x0A> ")
    return deplot_output

df['label'] = df['gts'].apply(lambda x:preprocess_gts(x))
# print(df['gts'].values[1])
# print(df['label'].values[1])
# print(df)

df = df.drop('gts', axis=1)
# print(df)

df_large = pd.concat([df, df_500k_1], axis=0)
df_large.reset_index(drop=True, inplace=True)  # 重置索引
# print(df_large)
print(df_large.head(10))
print(df_large.columns)
print(df['label'].head(10))

# df_large.to_csv('./datasets/train_with_df_large_cwq.csv', index=None)
# xf = pd.read_csv('./datasets/train_with_df_large_cwq.csv')
# print(xf)






