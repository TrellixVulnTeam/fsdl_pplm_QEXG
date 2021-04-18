
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from src.predictor import Predictor
from src.generator import Generator


############################################
# Common paramas
############################################

weights_path = 'data/weights/best_loss.pt'
meta_path = 'data/weights/generic_classifier_head_meta.json'

############################################
# Get predictions from the attribute model
############################################


pred = Predictor(weights_path, meta_path)
type(pred)
pred.predict('This is incredible! I love it, this is the best chicken I have ever had.')
pred.predict('god is love')


############################################
# Get generated text similar to as above
############################################

gen = Generator(weights_path, meta_path)

gen.generate(
    cond_text="In the beginning",
    bag_of_words='data/bow_confucianism.tsv',
    class_label=None,
)