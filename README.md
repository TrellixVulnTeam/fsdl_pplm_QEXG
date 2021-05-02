# Evolving Wisdom

## Repository Scope
This Git repository contains the materials for the final project of the [Full Stack Deep Learning Online 2021](https://fullstackdeeplearning.com/spring2021/)
course.


## Project Summary
The goal of this project was to explore the applicability of existing Natural Language Processing (NLP) tools for extracting
meaningful language reprentations from sacred texts across a range of ancient wisdom traditions (e.g. Buddhism, Taoism,
Hinduism, etc).  Specifically, I tested (1) if text could be generated in the style and topic chosen by a user, given 
a relatively small corpus of ancient texts; and (2) if similar texts to user-provided text could be identified from the corpus of ancient
texts.  Such approaches could be useful across a range of applications, such as providing useful text suggestions to users when journaling or writing on particular
topics.  

To accomplish this objective, a diverse range of ancient texts were extracted from an [online source](http://www.tparents.org/Library/Unification/Books/World-S/0-Toc.htm) 
that provides comparative texts passages across a range of topics and traditions.  Once extracted, those texts were used for
two objectives: (1) to extract keywords from each source tradition, using a TF-IDF approach (representing the tradition 'style' of interest); and (2) to train a discriminator model 
to predict the chapter heading topics provided in the text (representing the content), using the code from the [Plug and Play Language Model](https://github.com/uber-research/PPLM) 
approach developed by the [Uber Team](https://eng.uber.com/pplm/).  A simple Streamlit application was developed to 
take freeform user-supplied text, and give the user the option to either generate new text using their text as a prompt with GPT-2 as a base model; or
find the most similar passages to their text from the corpus.  The application allows the user to configure the style 
they are interested in (based on keywords) and the topic they want to generate text on.  

While the tool was successful at generating text and finding passages based on a similarity measure, the overall approach did not perform
well when tested anecdotally.  More work is needed, particularly with integrating a larger corpus of texts to improve the capacity
to generate text in a particular tradition style.


## Methods 

### Data Extraction

Data was extracted through regex parsing of the downloadable zip file of html pages provided on the website 
([source code](https://github.com/wtcooper/fsdl_pplm/blob/master/src/scrape_world_script.py)). 
Due to imperfect html parsing, the automatic parsing was enhanced by manually editing some of the html files to convert the html tags
on certain pages into the same expected format.  In addition, edits were made to the html text to have all source traditions
the same, as small spelling changes or differences in how the sources were referenced led to extraction issues. 

The code can be run in the repo using the following command:
```bash
   python src/scrape_world_script.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' --resource_dir 'resources'
```

### Extracting Keywords for Style

Keywords for each of the top ten major traditions were extracted from the corpus of sacred texts by performing a TF-IDF analysis
for each tradition and extracting the top 100 words.  The code for the analysis was adapted from [here](https://github.com/kavgan/nlp-in-practice/tree/master/tf-idf) and
can be found in the [source code](https://github.com/wtcooper/fsdl_pplm/blob/master/src/build_bows.py). Note: only the top ten 
traditions were used since the less common traditions had a limited number of text passages available in the corpus.

The keywords can be extracted by running the following command:
```bash
   python src/build_bows.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' --resource_dir 'resources'
```

### Training the Discriminator Model

The discriminator model was trained using the 21 chapter names as the label for each text passage in the corpus.  The 
[PPML codebase](https://github.com/wtcooper/fsdl_pplm/tree/master/PPLM)  was used to train the model with
a few minor updates, and utilized GPT-2 as the base model.  The discriminator model was run for a total of 50 epochs with a learning rate of 0.001.  90% of the 
corpus was used for training while the remaining 10% was used for testing, as per the default in the original Uber code.  Some manual and adhoc hyperparameter
tuning was conducted to improve the accuracy, where increasing the learning rate from 0.0001 to 0.001 improved the accuracy from roughly
25% to a maximum of 39% across the 50 epochs.    

The discriminator model can be trained by running the following command:
```bash
   python src/train_attribute_model.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' --epochs 50 --learning_rate 0.001 
```


### Generating Text

The [PPML codebase](https://github.com/wtcooper/fsdl_pplm/tree/master/PPLM) was used to generate new text using GPT-2 as the base model.  The user
can configure to use either a single or multiple BoW from the keyword extraction, in addition to using the discriminator model
to control the text generation process for a single topic type.  Note that a warning is provided when using the BoW in combination
with the discriminator model, as the authors of the original approach did not optimize the code to use both in tandem.  

Text can be generated by running the following command:
```bash
python PPLM/run_pplm.py -B '/content/drive/MyDrive/Colab Notebooks/FSDL/data/bow_taoism.tsv' -D generic -discrim_weights '/content/drive/MyDrive/Colab Notebooks/FSDL/data/weights/best_loss.pt' --discrim_meta '/content/drive/MyDrive/Colab Notebooks/FSDL/data/weights/generic_classifier_head_meta.json'  --class_label divine_law_truth_and_cosmic_principle --cond_text "In the beginning" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```

In addition, see the Streamlit application code for how to use the [Generator class](https://github.com/wtcooper/fsdl_pplm/blob/master/src/generator.py#L25)
 to handle [custom keywords](https://github.com/wtcooper/fsdl_pplm/blob/master/wisdom.py#L91) as input.

### Finding Similar Text

As an alternative to generating text, the application can also be used to find the most similar text passage from the corpus
relative to the user-provided text.  This was accomplished using the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library.
Only text passages within the users selected source style (e.g. Buddhism, Taoism) were searched.  Due to the large number
of possible text passages to search through, the user-provided text was first run through the discriminator model 
([Predictor class](https://github.com/wtcooper/fsdl_pplm/blob/master/src/predictor.py#L13)) to determine
the most top-five most likely chapter topics with the highest predicted labels, and then each potential candidate text passage
was compared to the user text using the cosine similarity metric built into the sentence-transformers library.  The default 
'stsb-roberta-large' pretrained model was used to extract and compare embeddings between the user-provided and corpus text.

Examples of using the [Recommender class](https://github.com/wtcooper/fsdl_pplm/blob/master/wisdom.py#L140) can be found in the Streamlit source code.  


### Developing the Streamlit Application

Streamlit was used to quickly develop a user interface that could take user input and provide configurable settings
for the user to select the desired style and content; and let user choose to either generate new text or find the most
similar passage in the existing text corpus.  Attempts were made to share the streamlit application through the Git interface, but 
build errors were encounted, likely due to memory and compute constraints due to attempting to load the GPT-2 model into 
the Streamlit servers.

The code can be run locally using the following command after saving the keyword files, model weights, and corpus DB
to a *data* folder from the previous commands:
```bash
streamlit run wisdom.py
```

### Model Training Infrastructure

Google Colab Pro was used for all components of the data extraction and model training, additionally providing a secure and 
robust logging mechanism by reading and writing directly to and from Google Drive.    


## Discussion and Next Steps

When using the application, the model did a relatively poor job at providing text that was anecdotally related to the source
traditions and content topics of interest.  This is most likely due to the esoteric nature of the texts, and the likelihood
that the underlying base model did not incorporate a large collection of sacred wisdom texts in the pre-trained model corpus.  Future 
work should focus on ingesting a large corpus of the sacred texts across traditions, which are often freely available
due to the public availablity of these ancient texts.  Once a new corpus of texts are collected, they should be used to fine-tune
the underlying based models to provide a base-model that has a broader spectrum of language across the styles and content of interest.

