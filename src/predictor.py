import json
import os
import sys
import torch
import math
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from PPLM.run_pplm_discrim_train import load_discriminator


class Predictor():

    def __init__(self, weights_path, meta_path):
        self.weights_path = weights_path
        self.meta_path = meta_path
        self.load_model()
        self.device = 'cpu'

    def load_model(self):
        with open(self.meta_path, 'r', encoding="utf8") as f:
            meta_params = json.load(f)
        self.classes = list(meta_params['class_vocab'].keys())
        self.model, self.meta_param = load_discriminator(self.weights_path, self.meta_path)


    # predict method from run_pplm_discrim_train.py
    def predict(self, input_sentence):
        input_t = self.model.tokenizer.encode(input_sentence)
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)

        log_probs = self.model(input_t).data.cpu().numpy()
        log_probs_list = log_probs.flatten().tolist()
        probs_list = [math.exp(log_prob) for log_prob in log_probs_list]
        print("Input sentence:", input_sentence)
        print("Predictions:", ", ".join(
            "{}: {:.4f}".format(c, prob) for c, prob in
            zip(self.classes, probs_list)
        ))

        max_value = max(probs_list)
        max_index = probs_list.index(max_value)
        label = self.classes[max_index]

        print(f"Best prediction: {label}: {max_value}")
        # to get top_n
        # top_k_n = 5
        # top_k = np.array([x.argsort()[-top_k_n:][::-1] for x in log_probs])

        return label