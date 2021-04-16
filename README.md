# Data Source

Data from text:
[World Scripture: A Comparative Anthology of Sacred Texts](http://www.tparents.org/Library/Unification/Books/World-S/0-Toc.htm)

# Steps
1. Download and extract [Zip from website](http://www.tparents.org/Library/Unification/Books/World-S/World-S.zip)
    * note: lots of manual cleaning required to get the extract script to run, not included

2. scrape the data:
   ```
   python src/scrape_world_script.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' --resource_dir 'resources'
   ```
3. build the BoW keyword list:
   - based on [this tutorial](https://github.com/kavgan/nlp-in-practice/tree/master/tf-idf)
   ```
   python src/build_bows.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' --resource_dir 'resources'
   ```
4. train the descriminator:
   ```
   python src/train_attribute_model.py --data_dir '/content/drive/MyDrive/Colab Notebooks/FSDL/data' 
   ```


