import pandas as pd
import json
import os
from hybrid_news_classifier import HybridNewsClassifier
from fake_news_classifier import FakeNewsLLM


with open("config.json") as f:
    config = json.load(f)

api_key = config["OPENAI_API_KEY"]
fake_news_detector = FakeNewsLLM(api_key, prompt_file="prompts.json")

datasets=['liar_test_dataset','fake_news_dataset.csv']
df_export=pd.DataFrame()

sample_size_used=config["sample_size_used"]

for i in datasets:
    dataset_config=config.get(i)
    temp_df=pd.read_csv(dataset_config['file_path'], delimiter=dataset_config['delimiter'])
    if dataset_config['columns']:
        temp_df.columns = dataset_config['columns']


    summary_df_temp=fake_news_detector.run_pipeline(temp_df, 
                                                    sample_size=sample_size_used,
                                                    remap_labels=dataset_config['remap_labels'], 
                                                    methods=dataset_config['methods'], 
                                                    dataset=i)
    df_export=pd.concat([df_export, summary_df_temp])

fake_news_detector.csv_export(df_export, 'output_summary')
