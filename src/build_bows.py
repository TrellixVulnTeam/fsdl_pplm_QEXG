"""
Borrowed from here:
https://github.com/kavgan/nlp-in-practice/tree/master/tf-idf
"""
import argparse
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import json


def pre_process(text):
    # lowercase
    text=text.lower()
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)

    return text

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

# remove stopwords
def get_stop_words(stop_file_path):
    """load stop words """

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)



def run_all(data_dir, resource_dir):

    conn = sqlite3.connect(f'{data_dir}/ws.db')


    ############################################################
    # Get the data and merge it
    ############################################################

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
    # Process text
    ############################################################

    #load a set of stop words
    stopwords = get_stop_words(f"{resource_dir}/stopwords.txt")

    # process text on the entire
    print(f"top counts:\n{df_text['source_tradition'].value_counts()}")
    traditions = df_text['source_tradition'].unique()

    keywords = {}
    # testing: tradition = 'Buddhism'
    for tradition in traditions:
        # print(tradition)

        corpus_df = df_text[df_text['source_tradition'] == tradition].copy()
        corpus_df['source_text'] = corpus_df['source_text'].apply(lambda x:pre_process(x))

        corpus = corpus_df['source_text'].tolist()

        if len(corpus)>20:

            cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
            word_count_vector=cv.fit_transform(corpus)
            feature_names=cv.get_feature_names()

            tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
            tf_idf_vector = tfidf_transformer.fit_transform(word_count_vector)

            #sort the tf-idf vectors by descending order of scores
            sorted_items=sort_coo(tf_idf_vector.tocoo())

            #extract only the top n; n here is 10
            keywords_dict = extract_topn_from_vector(feature_names,sorted_items,200)

            print("="*80)
            print(f"tradition: {tradition}")
            print(f"keywords: {list(keywords_dict.items())[:10]}")

            keywords[tradition] = list(keywords_dict.keys())

    key_df = pd.DataFrame({'tradition': list(keywords.keys()), 'keywords': list(keywords.values())})
    key_df['keywords'] = key_df['keywords'].apply(lambda x: json.dumps(x))
    key_df.to_sql('tradition_bow', if_exists='replace', con=conn, index=False)


    # # test read them back in as list
    # test = pd.read_sql('select * from tradition_bow', conn)
    # keywords = test['keywords'].apply(lambda x: json.loads(x)).tolist()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="directory to write data to")
    parser.add_argument("--resource_dir", type=str, default="resources",
                        help="directory to write data to")

    args = parser.parse_args()
    print(f"Args: {vars(args)}")

    run_all(**(vars(args)))
