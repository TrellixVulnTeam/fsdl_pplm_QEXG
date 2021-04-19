"""
Parts borrowed from
https://towardsdatascience.com/rick-and-morty-story-generation-with-gpt2-using-transformers-and-streamlit-in-57-lines-of-code-8f81a8f92692
"""

import streamlit as st
import pandas as pd

from src.predictor import Predictor
from src.generator import Generator
from src.recommender import Recommender

############################################
# Common paramas
############################################

bow_fps = {
    'Custom Keywords (fill in below)': 'temp_bow.tsv',
    'Buddhism': 'model/bow_buddhism.tsv',
    'Hinduism': 'model/bow_hinduism.tsv',
    'Christianity': 'model/bow_christianity.tsv',
    'Judaism': 'model/bow_judaism.tsv',
    'Confucianism': 'model/bow_confucianism.tsv',
    'Sikhism': 'model/bow_sikhism.tsv',
    'Taoism': 'model/bow_taoism.tsv',
    'Jainism': 'model/bow_jainism.tsv',
    'African Traditional Religions': 'model/bow_african_traditional_religions.tsv'
}

attr_labels = {
    "None": None,
    "Divine Law Truth and Cosmic Principle": "divine_law_truth_and_cosmic_principle",
    "Eschatology and Messianic Hope":"eschatology_and_messianic_hope",
    "Faith": "faith",
    "Fall and Deviaiton": "fall_and_deviation",
    "Good Government and Welfare of Society": "good_government_and_the_welfare_of_society",
    "Life Beyond Death and Spiritual World": "life_beyond_death_and_the_spiritual_world",
    "Live for Others": "live_for_others",
    "Offering and Sacrifice": "offering_and_sacrifice",
    "Responsibility and Predestination": "responsibility_and_predestination",
    "Salvation Liberation and Enlightenment": "salvation_liberation_enlightenment",
    "Self Cultivation and Spiritual Growth": "self_cultivation_and_spiritual_growth",
    "Self Denial and Renunciation": "self_denial_and_renunciation",
    "The Founder": "the_founder",
    "The Human Condition": "the_human_condition",
    "The Major Sins": "the_major_sins",
    "The Purpose of Life for the Individual": "the_purpose_of_life_for_the_individual",
    "The Purpose of Life in the Family and Society": "the_purpose_of_life_in_the_family_and_society",
    "The Purpose of Life in the Natural World":"the_purpose_of_life_in_the_natural_world",
    "Ultimate Reality": "ultimate_reality",
    "Wisdom": "wisdom",
    "Worship": "worship"}

weights_path = 'model/weights/best_loss.pt'
meta_path = 'model/weights/generic_classifier_head_meta.json'
db_path = 'model/ws.db'


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(weights_path, meta_path, db_path):
    pred = Predictor(weights_path, meta_path)
    gen = Generator(weights_path, meta_path)
    rec = Recommender(db_path, 'stsb-roberta-large')

    return pred, gen, rec

predictor, generator, recommender = load_model(weights_path, meta_path, db_path)
print(f"Pred: {predictor}")
print(f"Gen: {generator}")

############################################
# Sidebar panel
############################################

st.sidebar.markdown('## Configure Style and Content')
bow = st.sidebar.selectbox('Choose Style', tuple(bow_fps.keys()))
bow_custom = st.sidebar.text_input('Custom Keywords (comma separated)', 'Love, Peace, Joy, God')
attr_model_label = st.sidebar.selectbox('Choose Topic', tuple(attr_labels.keys()))

st.sidebar.markdown('## Generate Options')
num_samples = st.sidebar.number_input("Number of samples:", value=1, min_value=1, max_value=10, step=1)
length = st.sidebar.number_input("Length of output text:", value=25, min_value=10, max_value=100, step=1)
stepsize = st.sidebar.number_input("Strength of conditioning:", value=0.04, min_value=0.01, max_value=.2, step=.01)
num_iterations = st.sidebar.number_input("Number of iterations:", value=1, min_value=1, max_value=10, step=1)

st.sidebar.markdown('## Recommender Options')
num_topics = st.sidebar.number_input("Number of topics to search:", value=1, min_value=1, max_value=10, step=1)


# if pick a custom BoW, then write a temporary file to pull that in
if bow == 'Custom Keywords (fill in below)':
    # write a file of keywords
    # bow_custom = 'Existential, Transcendent, Mystic'
    pd.DataFrame({'list': [x.strip() for x in bow_custom.split(',')]}).to_csv('temp_bow.tsv', header=False, index=False)

bow_fp = bow_fps[bow]
class_label = attr_labels[attr_model_label]

print(f"bow_fp: {bow_fp}")
print(f"class_label: {class_label}")

############################################
# Main panel
############################################
output_text_location = st.markdown('## Input:')
context = st.text_area('What do you want hear more about?', '', height=200, max_chars=1000)
gen_button = st.button('Generate new text')
rec_button = st.button('Recommend similar passages')

st.markdown('\n')
st.markdown('## Output:')
output_text_location = st.markdown('<- enter text and click button to begin ->')

st.markdown('\n')

from utils import st_stdout


if gen_button:
    output_text_location.markdown('<- computing in progress ->')

    with st_stdout("code"):
        output_text = generator.generate(
            cond_text=context,
            bag_of_words=bow_fp,
            class_label=class_label,
            num_samples=num_samples,
            length=length,
            stepsize=stepsize,
            num_iterations=num_iterations
        )

    output_text_location.markdown(output_text)

def get_key(val):
    for key, value in attr_labels.items():
        if val == value:
            return key

if rec_button:
    output_text_location.markdown('<- computing in progress ->')

    if bow == 'Custom Keywords (fill in below)':
        output_text_location.markdown("Currently you need to select an existing "
                                      "Keyword list in the dropdown instead of the Custom Keywords")
    else:
        top_labels = predictor.predict_top_k(context, top_k_n=num_topics)
        source_tradition = [bow]
        top_labels_text = [get_key(lab) for lab in top_labels]
        match = recommender.match([context], source_tradition, top_labels_text)
        output_text_location.markdown(match)

