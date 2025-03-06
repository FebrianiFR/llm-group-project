import pandas as pd
import json
import os
from hybrid_news_classifier import HybridNewsClassifier
from fake_news_classifier import FakeNewsLLM


with open("config.json") as f:
    config = json.load(f)

api_key = config["OPENAI_API_KEY"]
fake_news_detector = FakeNewsLLM(api_key, prompt_file="prompts.json")
df1 = pd.read_csv('input_data/test.tsv', delimiter='\t', header=None)
df1.columns = ['json_id', 'label', 'title','subject', 'speaker', 'speaker_job','state', 'party', 'barely_true','fasle','half_true', 'mostly_true', 'pants_on_fire','context']
df2 = pd.read_csv("input_data/text_fakenews_politic.csv")

#df1_result, df1_text_result = fake_news_detector.run_pipeline(df1, sample_size=5, remap_labels=True, methods=['title'], prompt_type="long")
df2_result, df2_text_result = fake_news_detector.run_pipeline(df2, sample_size=1, remap_labels=False, methods=['title', 'text', 'url'])
