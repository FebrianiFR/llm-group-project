import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter
import openai
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix



with open(r'../config.json') as config_file:
    config = json.load(config_file)

data_path = config['data_path']
output_path = config['output_path']

# Initialize API Key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", 
                                         config['api_key']))  # Securely store API key

#Read Data
df_raw = pd.read_csv(data_path)
print(df_raw.head())


## Section for Title Detection

#Insert different prompt type:

def prompt_selector(number, type, insert_input):
    '''
    this function is to generate different prompting selection
    '''
    if number==1:
        prompt=f"""You are fake news detector, 
        you need to do binary classification whether this news is fake from different type of inputs depend on what i give you, this time i will give you {type}.
        Input: {insert_input}
        answer in a single word real or fake (must be binary, can't be something else)
        """
    elif number==2:
        prompt=f"""Classify this {type}: {insert_input} as fake or real news (must be binary, can't be something else), 
        answer in a single word real or fake, input can be title, url, text
        """
    else:
        prompt=f"""
        I will give you a news {type}, then you need to classify whether it is real or fake in a single word (must be binary, can't be something else)
        Title: {insert_input}
        """
    return prompt


def classify_news_title(input_prompt):
    ''' This function is to classify news based on title.
    The input is the prompt
    '''
    prompt=input_prompt
    try:
        response = client.chat.completions.create(model='gpt-4o',
                                                    messages=[{"role": "user","content": prompt}],
                                                    temperature=0
                                                    )
        result= response.choices[0].message.content
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        return result, token_usage
    except Exception as e:
        return "Unknown", "Unknown"


#run
df_raw_used=df_raw.copy()

results_detection = []
results_token=[]
detection_type=['title','news_url','article_text']
for i in detection_type:
    temp_df=df_raw_used[df_raw_used[i].notna()]
    input=temp_df[i].tolist()
    id_list=temp_df.id.tolist()
    for j,k in zip(input[:10],id_list):
        for number in range(1,4):
            prompt=prompt_selector(number, i, j)
            result, token_usage=classify_news_title(prompt)
            row_to_append={'id':k,
                        'result':result.lower() if result is not None else "Unknown",
                        'prompt_type':i,
                        'prompt_number':number}
            results_detection.append(row_to_append)
            results_token.append({'id':k, 'prompt_type':i,'prompt_number':number}|token_usage)


#convert to df then pivot
#convert to df then pivot
df_prompt=pd.DataFrame(results_detection)
df_pivot_prompt=pd.pivot(df_prompt,index=['id'],columns=['prompt_type','prompt_number'],values=['result']).reset_index()
df_pivot_prompt.columns=[a if b=="" else 'prompt'+'_'+str(b)+'_'+str(c) for a,b,c in df_pivot_prompt.columns]


df_token=pd.DataFrame(results_token)
df_token['prompt_type_number']='prompt_'+df_token['prompt_type']+'_'+df_token['prompt_number'].astype(str)
df_token_agg=df_token.groupby(['prompt_type_number']).agg({'prompt_tokens':'mean','completion_tokens':'mean'}).reset_index()
df_token_agg=df_token_agg.rename(columns={'prompt_tokens':'avg_prompt_tokens_per_article',
                                          'completion_tokens':'avg_completion_tokens_per_article'
                                          })
df_token_agg['avg_total_tokens_per_article']=df_token_agg['avg_prompt_tokens_per_article']+df_token_agg['avg_completion_tokens_per_article']


#merge
df_final=(df_raw_used[['id','label']]).merge(df_pivot_prompt,on='id',how='inner')


accuracy_list=[]
#get accuracy score
for i in (df_final.drop(columns=['id','label'])).columns:
    accuracy_df_temp=df_final[df_final[i].notna()].copy()
    accuracy_df_temp['label']=(np.where(accuracy_df_temp['label']=='real',1,0))
    accuracy_df_temp[i]=(np.where(accuracy_df_temp[i]=='real',1,0))
    accuracy_temp=accuracy_score(accuracy_df_temp['label'],accuracy_df_temp[i])

    rows_accuracy={'prompt_type_number':i,'accuracy':accuracy_temp}
    accuracy_list.append(rows_accuracy)
df_accuracy=pd.DataFrame(accuracy_list)

#merge with token 
df_output=df_accuracy.merge(df_token_agg, on=['prompt_type_number'], how='inner')
df_output['token_efficiency']=df_output['accuracy']/df_output['avg_total_tokens_per_article']
df_output['token_efficiency_normalized']=df_output['accuracy']/np.log1p(df_output['avg_total_tokens_per_article'])

df_output.to_csv(output_path)