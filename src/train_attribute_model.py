import argparse
import os
import pandas as pd
import sqlite3

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from PPLM.run_pplm_discrim_train import train_discriminator


def run_all(data_dir):

    print('Starting run_all to build descriminator dataset')
    if not os.path.exists(f'{data_dir}/weights/'):
        os.makedirs(f'{data_dir}/weights/')

    conn = sqlite3.connect(f'{data_dir}/ws.db')

    ############################################################
    # Get the data and merge it
    ############################################################

    # df_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%';", conn)

    df_chapter = pd.read_sql('select * from chapter', conn)
    df_chapter.sort_values(['chapter_number'], inplace=True)

    df_section = pd.read_sql('select * from section', conn)
    df_section.sort_values(['chapter_number', 'section_number'], inplace=True)

    df_text = pd.read_sql('select * from text', conn)
    df_text.sort_values(['chapter_number', 'section_number', 'id'], inplace=True)

    df_text = pd.merge(df_text, df_chapter.drop(['id'], axis=1), how='left', on='chapter_number')
    df_text = pd.merge(df_text, df_section.drop(['id'], axis=1), how='left', on=['chapter_number', 'section_number'])



    ############################################################
    # Build the generic dataset
    ############################################################

    # get the good labels so only use those
    df_bow = pd.read_sql('select * from tradition_bow', conn)

    df_discrim_output = df_text[df_text['source_tradition'].isin(df_bow['tradition'].unique().tolist())].copy()
    df_discrim_output.shape
    df_text.shape

    # In case of generic, the dataset is expected to be a TSBV file with structure: class \\t text"
    df_discrim_output['chapter_name_label'] = df_discrim_output['chapter_name'].str.lower().replace('[^\w]+', '_', regex=True)

    # go through each row - if it's > 100 length then split into multipe rows -
    df_discrim_output = df_discrim_output[['chapter_name_label', 'source_text']].copy()
    df_discrim_output.to_csv(f'{data_dir}/discrim_train_data.tsv', sep='\t', index=False)

    print(f'Finished writing to: {data_dir}/discrim_train_data.tsv')
    # test that outputted correctly
    # test_input = pd.read_csv('data/discrim_train_data.tsv', sep='\t')
    # print(df_discrim_output.shape)
    # print(test_input.shape)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="directory to write data to")
    # parser.add_argument("--dataset", type=str, default="SST",
    #                     choices=("SST", "clickbait", "toxic", "generic"),
    #                     help="dataset to train the discriminator on."
    #                          "In case of generic, the dataset is expected"
    #                          "to be a TSBV file with structure: class \\t text")
    # parser.add_argument("--dataset_fp", type=str, default="",
    #                     help="File path of the dataset to use. "
    #                          "Needed only in case of generic datadset")
    parser.add_argument("--pretrained_model", type=str, default="gpt2-medium",
                        help="Pretrained model to use as encoder")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learnign rate")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    # parser.add_argument("--save_model", action="store_true",
    #                     help="whether to save the model")
    # parser.add_argument("--cached", action="store_true",
    #                     help="whether to cache the input representations")
    # parser.add_argument("--no_cuda", action="store_true",
    #                     help="use to turn off cuda")
    # parser.add_argument("--output_fp", default=".",
    #                     help="path to save the output to")
    args = parser.parse_args()

    # build the training data
    run_all(args.data_dir)


    # train the descriminator
    train_discriminator(
        dataset='generic',
        dataset_fp=f'{args.data_dir}/discrim_train_data.tsv',
        pretrained_model=args.pretrained_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        save_model=True,
        cached=False,
        no_cuda=False,
        output_fp=f'{args.data_dir}/weights/')



