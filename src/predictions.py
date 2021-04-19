
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from src.predictor import Predictor
from src.generator import Generator
from src.recommender import Recommender



############################################
# Common paramas
############################################

weights_path = 'data/weights/best_loss.pt'
meta_path = 'data/weights/generic_classifier_head_meta.json'
db_path = 'data/ws.db'

############################################
# Get predictions from the attribute model
############################################


pred = Predictor(weights_path, meta_path)
type(pred)
pred.predict('This is incredible! I love it, this is the best chicken I have ever had.')
pred.predict('god is love')

top_labels = pred.predict_top_k('god is love', top_k_n=5)


############################################
# Get generated text similar to as above
############################################

gen = Generator(weights_path, meta_path)

gen.generate(
    cond_text="In the beginning",
    bag_of_words='data/bow_confucianism.tsv',
    class_label=None,
)



############################################
# Recommendations
############################################

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

def get_key(val):
    for key, value in attr_labels.items():
        if val == value:
            return key

source_tradition = ['Hinduism']
input_text = ["I like God"]
top_labels = pred.predict_top_k(input_text, top_k_n=5)
top_labels_text = [get_key(lab) for lab in top_labels]

rec = Recommender('data/ws.db')
match = rec.match(input_text, source_tradition, top_labels_text)