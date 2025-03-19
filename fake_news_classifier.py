import re
import os
import openai
import pandas as pd
import time
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import json
import tiktoken

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class FakeNewsLLM:
    def __init__(self, api_key, prompt_file):
        self.client = openai.OpenAI(api_key=api_key)
        self.stop_words = set(stopwords.words("english"))
      


        
        # Load predefined prompt templates from external JSON file
        with open(prompt_file, 'r') as f:
            self.prompts = json.load(f)
    
    def clean_text(self, text):
        text = re.sub(r"\n+", "", text).strip().lower()
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        text = " ".join(filtered_words)
        text = re.sub(r"[^\w\s]", "", text)
        return text
    
    def truncate_article_tokens(self,text, max_tokens=3000, openai_engine=None):
        enc = tiktoken.encoding_for_model(openai_engine)  
        tokens = enc.encode(text)
        truncated_tokens = tokens[:max_tokens]
        truncated_article=enc.decode(truncated_tokens)
        return truncated_article
    
    def preprocess_data(self, df,openai_engine):
        print("Preprocessing data: Cleaning text...")
        df['clean_title'] = df['title'].apply(self.clean_text)
        
        #df_text = df.dropna(subset=['article_text']) if 'article_text' in df.columns else None
        df_text = df.copy() if 'article_text' in df.columns else None
        if df_text is not None:
            df_text['article_text'] = (df_text['article_text']).astype(str).apply(self.clean_text)
            df_text['article_text'] = (df['article_text'].astype(str)).apply(lambda row: self.truncate_article_tokens(row, openai_engine=openai_engine))
            print("Preprocessing completed.")
            return df_text
        else:
            print("Preprocessing completed.")
            return df
        

    
    def classify_text(self, method, prompt_type, title=None, article_text=None, news_url=None, openai_engine=None,dataset=None):
        
        
        #validate input
        if prompt_type not in self.prompts:
            raise ValueError(f"Invalid prompt type: {prompt_type}. Must be 'short' or 'long'.")
        if method not in self.prompts[prompt_type]:
            raise ValueError(f"Invalid method: {method}. Available methods: {list(self.prompts[prompt_type].keys())}")
        
        # retrieve template and create message
        prompt_template = self.prompts[prompt_type][method]

        message = prompt_template.format(
            title=title if title else "N/A",
            article_text=article_text if article_text else "N/A",
            news_url=news_url if news_url else "N/A"
        )
        
        start_time_detection = time.time()
        response = self.client.chat.completions.create(
            model=openai_engine, 
            messages=[{"role": "user", "content": message}],
            temperature=0
        )
        end_time_detection= time.time()
        detection_time=end_time_detection - start_time_detection
        
        result = response.choices[0].message.content.strip().lower()
        token_usage = {
            "Data":dataset,
            "Engine": openai_engine,
            "Method": method,
            "Prompt Type": prompt_type,
            "Prompt Tokens": response.usage.prompt_tokens,
            "Completion Tokens": response.usage.completion_tokens,
            "Total Tokens": response.usage.total_tokens,
            "Detection Time per Article":detection_time
        }
        return result, token_usage
    
    def evaluate_metrics(self, df,prompt_type, label_column, results_list, processing_time, openai_engine=None,dataset=None):
        print(f"Evaluating metrics for {label_column}...")
        
        valid_labels = ['real', 'fake']
        invalid_classifications = df[~df[label_column].isin(valid_labels)]
        num_invalid = len(invalid_classifications)
        df = df[df[label_column].isin(valid_labels)]
        
        y_true = df['label'].map({'real': 1, 'fake': 0})
        y_pred = df[label_column].map({'real': 1, 'fake': 0})
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        print(f"Metrics evaluation completed for {label_column}. {num_invalid} entries were dropped due to unrecognized classification.")
        
        results_list.append({
            'Data':dataset,
            'Engine': openai_engine,
            'Prompt Type':prompt_type,
            'Method': str(label_column).replace("_label", ""),
            'Total Processing Time (s)': processing_time,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Invalid Classifications': num_invalid
        })
    
    def create_summary(self, token_usage_list, results_list):
        engine_cost_dict_per_token={
            'gpt-4o':{'input':0.0000025,'output':0.00001},
            'gpt-3.5-turbo':{'input':0.0000005,'output':0.0000015}


        }
        df_cost=pd.DataFrame.from_dict(engine_cost_dict_per_token, orient='index').reset_index()
        df_cost=df_cost.rename(columns={'index':'Engine'})
        results_df_temp=pd.DataFrame(results_list)
        token_df_temp=pd.DataFrame(token_usage_list)

        token_df_temp_agg=token_df_temp.groupby(['Data','Engine','Prompt Type','Method']).agg({'Prompt Tokens':'mean',
                                                                                      'Completion Tokens':'mean',
                                                                                      'Total Tokens':'mean',
                                                                                      'Detection Time per Article':'mean'
                                                                                      }).reset_index()

        token_df_temp_agg=token_df_temp_agg.rename(columns={'Prompt Tokens':'Avg Prompt Tokens per Article',
                                                            'Completion Tokens':'Avg Completion Tokens per Article',
                                                            'Total Tokens':'Avg Total Tokens per Article',
                                                            'Detection Time per Article':'Avg Detection Time per Article'
                                                            })
        final_df=results_df_temp.merge(token_df_temp_agg, on=['Data', 'Engine','Prompt Type','Method'], how='inner')

        final_df=final_df.merge(df_cost,on='Engine',how='inner')
        final_df['Average OpenAI API Cost (USD) per Article']=(final_df['Avg Prompt Tokens per Article']*final_df['input'])+(final_df['Avg Completion Tokens per Article']*final_df['output'])
        final_df=final_df.drop(columns=['input','output'])
        return final_df
    
    def csv_export(self, df, filename):
        df.to_csv('output_data/'+filename+'.csv', index=False)
        print(f"Results saved to {filename}")

    def run_pipeline(self, df, sample_size=None, methods=['title', 'text', 'url'], remap_labels=False,dataset=None):
        print("Starting pipeline...")
        
        if remap_labels:
            df['label'] = df['label'].map({'pants-fire': 'fake', 'barely-true': 'fake', 'false': 'fake', 'half-true': 'real', 'mostly-true': 'real', 'true': 'real'})
        if sample_size:
            print(f"Sampling {sample_size} random entries from the dataset...")
            df = df.sample(n=sample_size, random_state=42)
        
        
        results_list = []
        token_usage_list = []
        engine_choices=['gpt-4o','gpt-3.5-turbo']
        prompt_types = ['short','long']
        for engine in engine_choices:
            for prompt_type in prompt_types:
                for method in methods:
                    print(f"Evaluating classification using {method} with {prompt_type} prompt...")
                    df = self.preprocess_data(df,openai_engine=engine)
                    start_time = time.time()
                    df[method + '_label'], token_usage = zip(*df.apply(lambda row: self.classify_text(method, prompt_type, row.get('title'), row.get('article_text'), row.get('news_url'), openai_engine=engine,dataset=dataset), axis=1))
                    processing_time = time.time() - start_time
                    self.evaluate_metrics(df, prompt_type, method + '_label', results_list, processing_time,openai_engine=engine, dataset=dataset)
                    token_usage_list.extend(token_usage)
        

        

        final_df=self.create_summary(token_usage_list, results_list)
        print(f"Pipeline completed successfully for {dataset}")
        return final_df
    



