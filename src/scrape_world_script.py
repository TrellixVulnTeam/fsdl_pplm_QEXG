import argparse
import os
import re
import sqlite3
import pandas as pd


def write_chapter(conn, chapter):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO chapter(chapter_number, chapter_name)
                  VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, chapter)
    conn.commit()
    return cur.lastrowid

def write_section(conn, section):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO section(chapter_number, section_number, section_name)
                  VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, section)
    conn.commit()
    return cur.lastrowid

def write_text(conn, text):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO text(chapter_number, section_number, source_tradition, source_location, source_text)
                  VALUES(?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, text)
    conn.commit()
    return cur.lastrowid


def run_all(data_dir, resource_dir):

    create_db_tables=True
    write_data=True

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # delete the DB if exists
    if os.path.exists(f'{data_dir}/ws.db'):
        os.remove(f'{data_dir}/ws.db')

    # set up sqllite
    conn = sqlite3.connect(f'{data_dir}/ws.db')

    create_chapter = """
    CREATE TABLE IF NOT EXISTS chapter (
        id integer PRIMARY KEY AUTOINCREMENT,
        chapter_number integer NOT NULL,
        chapter_name text not null
    );
    """

    create_section = """
    CREATE TABLE IF NOT EXISTS section (
        id integer PRIMARY KEY AUTOINCREMENT,
        chapter_number integer NOT NULL,
        section_number integer NOT NULL,
        section_name text not null
    );
    """

    create_text = """
    CREATE TABLE IF NOT EXISTS text (
        id integer PRIMARY KEY AUTOINCREMENT,
        chapter_number integer NOT NULL,
        section_number integer NOT NULL,
        source_tradition text not null,
        source_location text not null,
        source_text text not null
    );
    """

    if create_db_tables:
        c = conn.cursor()
        c.execute(create_chapter)
        c.execute(create_section)
        c.execute(create_text)
        c.close()


    df_chapter = pd.read_sql('select * from chapter', conn)
    df_chapter.sort_values(['chapter_number'], inplace=True)
    print(f"Initial df_chapter.shape: {df_chapter.shape}")

    df_section = pd.read_sql('select * from section', conn)
    df_section.sort_values(['chapter_number', 'section_number'], inplace=True)
    print(f"Initial df_section.shape: {df_section.shape}")

    df_text = pd.read_sql('select * from text', conn)
    df_text.sort_values(['chapter_number', 'section_number', 'id'], inplace=True)
    print(f"Initial df_text.shape: {df_text.shape}")


    #########################################
    # Data
    #########################################

    # get a list of all the files
    files = [f for f in os.listdir(f'{resource_dir}/wc_extract') if re.match(r'^\w\w-\d\d-\d\d.htm$', f)]

    # file_name = files[19]  chapter
    # file_name = files[18]
    for i,file_name in enumerate(files):
        # print(f"i: {i}, file: {file_name}")

        file_name_extract = file_name.lower()
        chapter_number = re.findall('ws-(\d*)-\d*.htm', file_name_extract)
        section_number = re.findall('ws-\d*-(\d*).htm', file_name_extract)[0]

        if len(chapter_number)==0:
            print("\tNot a correct page format")
        else:
            chapter_number = chapter_number[0]
            # print(f"\tChapter number: {chapter_number}")

        # print(f"\tSection number: {section_number}")

        file_path = f'{resource_dir}/wc_extract/{file_name}'

        with open(file_path, 'r') as file:
            data = file.read()


        ####################
        # get the chapter if it's the chapter
        ####################

        # get the chapter name and continue
        if '-00.htm' in file_name:

            main_match = '<p>Chapter [\d\w]+:(.*)</p>'
            matches = re.findall(main_match, data)

            chapter_name = matches[0].strip()
            # print(f"\tChapter name: {chapter_name}")

            if write_data:
                write_chapter(conn, (chapter_number, chapter_name))

            continue


        ####################
        # get the title and description of the section
        ####################

        data_title = data.replace('<b><font FACE="Arial" SIZE="4"><p>&nbsp;<img SRC="GoldGrnLine.GIF" WIDTH="580" HEIGHT="14"></p>', '__title_start__').replace('</b></font><p>', '__title_end__')
        main_match = '__title_start__([\S\s]*)__title_end__'
        matches = re.findall(main_match, data_title)

        if len(matches)==1:
            section_title = matches[0].replace("\n","").replace("<p>", "").replace("</p>", "").strip()
            # section_desc = matches[1].replace("\n","").replace("<p>", "").replace("</p>", "").strip()

            if write_data:
                write_section(conn, (chapter_number, section_number, section_title))

        else:
            print("\tBroken logic in title extraction")
            continue


        ####################
        # get the text and source blocks
        ####################

        data_scrubbed = data.replace("<p><hr></p>", "__marker__").replace("<p>", "").replace("</p>", "")

        main_match = '__marker__([\S\s]*)__marker__'
        matches = re.findall(main_match, data_scrubbed)

        if len(matches)==1:
            data_splits = matches[0].split("__marker__")

            # print(f"\tNumber of data blocks: {len(data_splits)}")
            # data_split = data_splits[0]

            for data_split in data_splits:
                data_block = [x.strip().replace("&quot;", '"') for x in data_split.split("\n") if x != '']

                # only do if greater than one split - the footnotes are single usually
                if len(data_block)>1:
                    # print(data_block)
                    text = '\n'.join(data_block[:-1])
                    source = data_block[-1]

                    # extra religion
                    tradition = source.split(".")[0].strip()
                    location = ' '.join(source.split(".")[1:]).strip()
                    # print(f"\ttradition: {tradition}")
                    # print(f"\tlocation: {location}")

                    if write_data:
                        write_text(conn, (chapter_number, section_number, tradition, location, text))


    df_chapter = pd.read_sql('select * from chapter', conn)
    df_chapter.sort_values(['chapter_number'], inplace=True)
    print(f"Final df_chapter.shape: {df_chapter.shape}")

    df_section = pd.read_sql('select * from section', conn)
    df_section.sort_values(['chapter_number', 'section_number'], inplace=True)
    print(f"Final df_section.shape: {df_section.shape}")

    df_text = pd.read_sql('select * from text', conn)
    df_text.sort_values(['chapter_number', 'section_number', 'id'], inplace=True)
    print(f"Final df_text.shape: {df_text.shape}")

    conn.close()


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


