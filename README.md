# Data Source

Data from text:
[World Scripture: A Comparative Anthology of Sacred Texts](http://www.tparents.org/Library/Unification/Books/World-S/0-Toc.htm)

# Steps
1. Download and extract [Zip from website](http://www.tparents.org/Library/Unification/Books/World-S/World-S.zip)
    * note: lots of manual cleaning required to get the extract script to run, not included

2. scrape the data:
```bash
   python src/scrape_world_script.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' --resource_dir 'resources'
```

3. build the BoW keyword list:
   - based on [this tutorial](https://github.com/kavgan/nlp-in-practice/tree/master/tf-idf)
```bash
   python src/build_bows.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' --resource_dir 'resources'
```

4. train the descriminator:
```bash
   python src/train_attribute_model.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' 
```

5. Generate text:
```bash
python PPLM/run_pplm.py -B '/content/drive/MyDrive/Colab Notebooks/FSDL/data/bow_taoism.tsv' -D generic -discrim_weights '/content/drive/MyDrive/Colab Notebooks/FSDL/data/weights/best_loss.pt' --discrim_meta '/content/drive/MyDrive/Colab Notebooks/FSDL/data/weights/generic_classifier_head_meta.json'  --class_label divine_law_truth_and_cosmic_principle --cond_text "In the beginning" --length 50 --gamma 1.0 --num_iterations 10 --num_samples 10 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample
```


python PPLM/run_pplm.py -B 'data/bow_taoism.tsv'  --cond_text "In the beginning" --length 20 --gamma 1.0 --num_iterations 1 --num_samples 3 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample  --verbosity 'regular'
python PPLM/run_pplm.py -B 'data/bow_confucianism.tsv'  --cond_text "In the beginning" --length 20 --gamma 1.0 --num_iterations 1 --num_samples 3 --stepsize 0.04 --kl_scale 0.01 --gm_scale 0.95 --sample --verbosity 'regular'



# Run Streamlit
streamlit run wisdom.py