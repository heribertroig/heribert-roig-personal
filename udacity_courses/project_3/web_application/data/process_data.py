# from udacity_courses.project_3.scripts.process_data import ETL
import sqlite3
import sys

import pandas as pd


class ETL:
    # Code also present in udacity_courses/project_3/scripts/process_data.py
    def __init__(self, messages_filepath: str = "data/messages.csv",
                 categories_filepath: str = "data/categories.csv"):
        self.messages = pd.read_csv(messages_filepath)
        self.categories = pd.read_csv(categories_filepath)
        self.df = pd.DataFrame()

    def run(self):
        self.merge_dataframes()
        self.split_categories()
        self.remove_duplicates()
        self.save_to_db()

    def merge_dataframes(self):
        df = pd.merge(self.messages, self.categories, how="inner", on="id")
        self.df = df
        return df

    def split_categories(self):
        df = self.df
        categories = (df["categories"].apply(lambda x: dict(item.split('-') for item in x.split(';')))).apply(pd.Series)
        categories = categories.astype(int)
        categories = categories.fillna(0)
        df.drop(columns=["categories"], inplace=True)
        df = pd.concat([df, categories], axis=1)
        self.df = df
        return df

    def remove_duplicates(self):
        df = self.df
        subset_columns = [col for col in df.columns if col != "id"]
        df.drop_duplicates(subset=subset_columns, inplace=True)
        self.df = df
        return df

    def save_to_db(self):
        conn = sqlite3.connect('../data/messages_categories.db')
        self.df.to_sql('clean_data', con=conn, if_exists='replace', index=False)

    def load_data(self, messages_filepath, categories_filepath):
        self.messages = pd.read_csv(messages_filepath, encoding='latin-1')
        self.categories = pd.read_csv(categories_filepath)
        df = self.merge_dataframes()
        return df

    def clean_data(self, df):
        df = self.split_categories()
        df = self.remove_duplicates()
        return df

    @staticmethod
    def save_data(df, database_filename):
        conn = sqlite3.connect(database_filename)
        df.to_sql('clean_data', con=conn, if_exists='replace', index=False)


def load_data(messages_filepath, categories_filepath, etl):
    df = etl.load_data(messages_filepath, categories_filepath)
    return df


def clean_data(df, etl):
    df = etl.clean_data(df)
    return df


def save_data(df, database_filename, etl):
    etl.save_data(df, database_filename)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        etl_ = ETL(messages_filepath, categories_filepath)
        df = load_data(messages_filepath, categories_filepath, etl_)

        print('Cleaning data...')
        df = clean_data(df, etl_)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, etl_)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == "__main__":
    main()
