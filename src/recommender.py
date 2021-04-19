"""
Based on this:
https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6
"""
from sentence_transformers import SentenceTransformer, util
import sqlite3
import pandas as pd
import torch

class Recommender():

    def __init__(
            self,
            db_path,
            pretrained_model='stsb-roberta-large',
            no_cuda=True
    ):

        self.device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        # self.device = 'cpu'
        self.db_path = db_path
        self.pretrained_model = pretrained_model

        self.load_model()
        self.load_db()

    def load_model(self):
        """
        Load the SentenceTransformer model
        base ond
        :return:
        """

        print(f"SentenceTransformer for model {self.pretrained_model}")
        self.model = SentenceTransformer(self.pretrained_model, device=self.device)

    def load_db(self):
        self.conn = sqlite3.connect(self.db_path)

        df_chapter = pd.read_sql('select * from chapter', self.conn)
        df_chapter.sort_values(['chapter_number'], inplace=True)

        df_section = pd.read_sql('select * from section', self.conn)
        df_section.sort_values(['chapter_number', 'section_number'], inplace=True)

        df_text = pd.read_sql('select * from text', self.conn)
        df_text.sort_values(['chapter_number', 'section_number', 'id'], inplace=True)

        df_text = pd.merge(df_text, df_chapter.drop(['id'], axis=1), how='left', on='chapter_number')
        self.df_text = pd.merge(df_text, df_section.drop(['id'], axis=1), how='left', on=['chapter_number', 'section_number'])

        print(f"DB data text table loaded with shape {self.df_text.shape}")


    # predict method from run_pplm_discrim_train.py
    def match(self, input_text, source_tradition, top_labels):

        print(f"Finding closest passage for {input_text}")

        # get the candidate sources
        candidate_text = self.df_text[self.df_text['chapter_name'].isin(top_labels)
                                 & self.df_text['source_tradition'].isin(source_tradition)]['source_text'].tolist()

        embedding1 = self.model.encode(input_text, convert_to_tensor=True)
        embedding2 = self.model.encode(candidate_text, convert_to_tensor=True)

        # compute similarity scores of two embeddings
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
        max_match_text = ''
        max_sim = 0.0
        for i in range(len(input_text)):
            for j in range(len(candidate_text)):
                sim = cosine_scores[i][j].item()
                if sim>max_sim:
                    max_match_text = candidate_text[j]
                    max_sim = sim
                    # print(f"New best match: {sim}: {max_match_text}")

        source = self.df_text[self.df_text['source_text'] == max_match_text]['source_location'].item()

        return f"{source_tradition[0]}, {source}: {max_match_text}"
