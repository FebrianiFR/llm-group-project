import re
import os
import openai
import pandas as pd
import matplotlib.pyplot as plt
import time
from nltk.corpus import stopwords
import json
from hybrid_news_classifier import HybridNewsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class FakeNewsLLM:
    def __init__(self, api_key, prompt_file):
        self.client = openai.OpenAI(api_key=api_key)
        self.stop_words = set(stopwords.words("english"))
        self.models = 'gpt-4o'

        self.hybrid_classifier = HybridNewsClassifier(self.client)
        
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
    
    def preprocess_data(self, df):
        print("Preprocessing data: Cleaning text...")
        df['clean_title'] = df['title'].apply(self.clean_text)
        
        df_text = df.dropna(subset=['article_text']) if 'article_text' in df.columns else None
        if df_text is not None:
            df_text['clean_article'] = df_text['article_text'].apply(self.clean_text)
        print("Preprocessing completed.")
        
        return df, df_text
    
    def classify_text(self, method, prompt_type, title=None, article_text=None, news_url=None):
        if method == "hybrid":
            classification, token_usage = self.hybrid_classifier.classify_news(title, news_url, article_text)
            return classification, token_usage
        
        if prompt_type not in self.prompts:
            raise ValueError(f"Invalid prompt type: {prompt_type}. Must be 'short' or 'long'.")
        if method not in self.prompts[prompt_type]:
            raise ValueError(f"Invalid method: {method}. Available methods: {list(self.prompts[prompt_type].keys())}")
        
        prompt_template = self.prompts[prompt_type][method]
        
        if method == "combination":
            message = prompt_template.format(
                title=title if title else "N/A",
                article_text=article_text if article_text else "N/A",
                news_url=news_url if news_url else "N/A"
            )
        else:
            text_input = title if method == "title" else article_text if method == "text" else news_url
            message = prompt_template.format(
                title=title if title else "N/A",
                text=text_input if text_input else "N/A",
                news_url=news_url if news_url else "N/A"
            )
        
        response = self.client.chat.completions.create(
            model='gpt-4o',
            messages=[{"role": "user", "content": message}],
            temperature=0
        )
        
        result = response.choices[0].message.content.strip().lower()
        token_usage = {
            "Method": method,
            "Prompt Type": prompt_type,
            "Prompt Tokens": response.usage.prompt_tokens,
            "Completion Tokens": response.usage.completion_tokens,
            "Total Tokens": response.usage.total_tokens
        }
        return result, token_usage
    
    def evaluate_metrics(self, df,prompt_type, label_column, results_list, processing_time):
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
            'Prompt Type':prompt_type,
            'Method': label_column,
            'Processing Time (s)': processing_time,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Invalid Classifications': num_invalid
        })
    
    def save_results(self, results_list, filename="output_data/classification_results.csv"):
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


    def run_pipeline(self, df, sample_size=None, methods=['title', 'text', 'url', 'combination'], remap_labels=False):
        print("Starting pipeline...")
        
        if remap_labels:
            df['label'] = df['label'].map({'pants-fire': 'fake', 'barely-true': 'fake', 'false': 'fake', 'half-true': 'real', 'mostly-true': 'real', 'true': 'real'})
        if sample_size:
            print(f"Sampling {sample_size} random entries from the dataset...")
            df = df.sample(n=sample_size, random_state=42)
        df, df_text = self.preprocess_data(df)
        
        results_list = []
        token_usage_list = []

        prompt_types = ['short','long']
        for prompt_type in prompt_types:
            for method in methods:
                print(f"Evaluating classification using {method} with {prompt_type} prompt...")
                start_time = time.time()
                df[method + '_label'], token_usage = zip(*df.apply(lambda row: self.classify_text(method, prompt_type, row.get('title'), row.get('article_text'), row.get('news_url')), axis=1))
                processing_time = time.time() - start_time
                self.evaluate_metrics(df, prompt_type, method + '_label', results_list, processing_time)
                token_usage_list.extend(token_usage)
        
        print("Evaluating classification using hybrid method...")
        start_time = time.time()
        df['hybrid_label'], hybrid_token_usage = zip(*df.apply(lambda row: self.hybrid_classifier.classify_news(row.get('title'), row.get('news_url'), row.get('article_text')), axis=1))
        processing_time = time.time() - start_time
        self.evaluate_metrics(df, 'hybrid', 'hybrid_label', results_list, processing_time)
        token_usage_list.extend(hybrid_token_usage)
        
        self.save_results(results_list)
        self.save_token_usage(token_usage_list)
        print("Pipeline completed successfully.")
        return df, df_text
    
    def save_token_usage(self, token_usage_list, filename="output_data/token_usage.csv"):
        token_df = pd.DataFrame(token_usage_list)
        token_df.to_csv(filename, index=False)
        print(f"Token usage saved to {filename}")


