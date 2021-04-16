import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import json

data_dir= f"data"
conn = sqlite3.connect(f'{data_dir}/ws.db')


############################################################
# Get the data and merge it
############################################################

df_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';", conn)

df_chapter = pd.read_sql('select * from chapter', conn)
df_chapter.sort_values(['chapter_number'], inplace=True)
print(df_chapter.columns)

df_section = pd.read_sql('select * from section', conn)
df_section.sort_values(['chapter_number', 'section_number'], inplace=True)
print(df_section.columns)

df_text = pd.read_sql('select * from text', conn)
df_text.sort_values(['chapter_number', 'section_number', 'id'], inplace=True)
print(df_text.columns)


df_text = pd.merge(df_text, df_chapter.drop(['id'], axis=1), how='left', on='chapter_number')
df_text = pd.merge(df_text, df_section.drop(['id'], axis=1), how='left', on=['chapter_number', 'section_number'])



############################################################
# Build the generic dataset
############################################################

# get the good labels so only use those
df_bow = pd.read_sql('select * from tradition_bow', conn)

df_discrim_output = df_text[df_text['source_tradition'].isin(df_bow['tradition'].unique().tolist())].copy()
df_discrim_output.shape
df_text.shape

# In case of generic, the dataset is expected to be a TSBV file with structure: class \\t text"
df_discrim_output['chapter_name_label'] = df_discrim_output['chapter_name'].str.lower().replace('[^\w]+', '_', regex=True)

# go through each row - if it's > 100 length then split into multipe rows -
df_discrim_output = df_discrim_output[['chapter_name_label', 'source_text']].copy()
df_discrim_output.to_csv('data/discrim_train_data.tsv', sep='\t', index=False)

# test that outputted correctly
# test_input = pd.read_csv('data/discrim_train_data.tsv', sep='\t')
# print(df_discrim_output.shape)
# print(test_input.shape)



############################################################
# Train the discreminator
############################################################

from PPLM.run_pplm_discrim_train import train_discriminator


train_discriminator(
    dataset='generic',
    dataset_fp='data/discrim_train_data.tsv',
    pretrained_model="gpt2-medium",
    epochs=2,
    learning_rate=0.0001,
    batch_size=64,
    log_interval=10,
    save_model=True,
    cached=False,
    no_cuda=False,
    output_fp='data/weights/')