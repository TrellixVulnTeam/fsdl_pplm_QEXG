import json
import os
import sys
import torch
import math
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from PPLM.run_pplm import get_bag_of_words_indices, generate_text_pplm, PPLM_BOW_DISCRIM, PPLM_BOW, PPLM_DISCRIM
from PPLM.pplm_classification_head import ClassificationHead




class Generator():

    def __init__(
            self,
            weights_path,
            meta_path,
            pretrained_model="gpt2-medium",
            seed=0,
            no_cuda=True
        ):

        torch.manual_seed(seed)
        np.random.seed(seed)

        # set the device
        self.device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        # self.device = 'cpu'

        self.weights_path = weights_path
        self.meta_path = meta_path
        self.pretrained_model = pretrained_model

        self.load_lang_model()
        self.load_attribute_model()


    def load_lang_model(self):
        print(f"Loading language model {self.pretrained_model}")
        # load pretrained model
        self.lang_model = GPT2LMHeadModel.from_pretrained(
            self.pretrained_model,
            output_hidden_states=True
        )
        self.lang_model.to(self.device)
        self.lang_model.eval()

        # load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_model)

        # Freeze GPT-2 weights
        for param in self.lang_model.parameters():
            param.requires_grad = False



    def load_attribute_model(self):
        print(f"Loading attribute classifier model")

        with open(self.meta_path, 'r', encoding="utf8") as f:
            self.meta_params = json.load(f)

        print(f"\t{self.meta_params}")
        print(f"\t{self.meta_params}")

        self.classifier = ClassificationHead(
            class_size=self.meta_params['class_size'],
            embed_size=self.meta_params['embed_size']
        ).to(self.device)

        self.classifier.load_state_dict(
            torch.load(self.weights_path, map_location=self.device))
        self.classifier.eval()


    # predict method from run_pplm_discrim_train.py
    def generate(
            self,
            cond_text,
            bag_of_words=None,
            class_label=None,
            num_samples=1,
            length=100,
            stepsize=0.02,
            temperature=1.0,
            top_k=10,
            sample=True,
            num_iterations=1,
            grad_length=10000,
            horizon_length=1,
            window_length=0,
            decay=False,
            gamma=1.5,
            gm_scale=0.9,
            kl_scale=0.01,
        ):

        # get the class label integer if using attribute model
        label_id = None
        temp_classifier = None
        if class_label:
            temp_classifier = self.classifier
            if isinstance(class_label, str):
                if class_label in self.meta_params["class_vocab"]:
                    label_id = self.meta_params["class_vocab"][class_label]
                else:
                    label_id = self.meta_params["default_class"]

            elif isinstance(class_label, int):
                if class_label in set(self.meta_params["class_vocab"].values()):
                    label_id = class_label
                else:
                    label_id = self.meta_params["default_class"]

            else:
                label_id = self.meta_params["default_class"]


        bow_indices = []
        if bag_of_words:
            bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                                   self.tokenizer)

        context = self.tokenizer.encode(
            self.tokenizer.bos_token + cond_text,
            add_special_tokens=False
        )


        if bag_of_words and class_label:
            loss_type = PPLM_BOW_DISCRIM
        elif bag_of_words:
            loss_type = PPLM_BOW
        elif class_label is not None:
            loss_type = PPLM_DISCRIM
        else:
            raise Exception("Specify either a bag of words or a discriminator")

        pert_gen_tok_texts = []

        for i in range(num_samples):
            pert_gen_tok_text, _, _ = generate_text_pplm(
                model=self.lang_model,
                tokenizer=self.tokenizer,
                context=context,
                device=self.device,
                perturb=True,
                bow_indices=bow_indices,
                classifier=temp_classifier,
                class_label=label_id,
                loss_type=loss_type,
                length=length,
                stepsize=stepsize,
                temperature=temperature,
                top_k=top_k,
                sample=sample,
                num_iterations=num_iterations,
                grad_length=grad_length,
                horizon_length=horizon_length,
                window_length=window_length,
                decay=decay,
                gamma=gamma,
                gm_scale=gm_scale,
                kl_scale=kl_scale
            )

            pert_gen_tok_texts.append(pert_gen_tok_text)

            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # iterate through the perturbed texts
        gen_text = []
        for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
            pert_gen_text = self.tokenizer.decode(pert_gen_tok_text.tolist()[0]).replace('<|endoftext|>', '')

            # print("= Perturbed generated text {} =".format(i + 1))
            # print(pert_gen_text)
            # print()

            gen_text.append(pert_gen_text)

        return gen_text