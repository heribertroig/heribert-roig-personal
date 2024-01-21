# from udacity_courses.project_3.scripts.process_data import ETL
import sqlite3
import sys

import pandas as pd


# Code also present in udacity_courses/project_3/scripts/process_data.py. Need to find why it can't be imported by
# udacity reviewers
class ETL:
    """ETL class to load, clean and save data to database"""
    def __init__(self, messages_filepath: str = "data/messages.csv",
                 categories_filepath: str = "data/categories.csv"):
        self.messages = pd.read_csv(messages_filepath)
        self.categories = pd.read_csv(categories_filepath)
        self.df = pd.DataFrame()

    def run(self):
        """
        Run the ETL process
        Returns:
            None
        """
        self.merge_dataframes()
        self.split_categories(self.df)
        self.drop_wrong_values(self.df)
        self.remove_duplicates(self.df)
        self.save_to_db()

    def merge_dataframes(self):
        """
        Merge messages and categories dataframes
        Returns:
            df: merged dataframe
        """
        df = pd.merge(self.messages, self.categories, how="inner", on="id")
        self.df = df
        return df

    def drop_wrong_values(self, df):
        """
        Drop rows with wrong values
        Args:
            df: dataframe to drop rows from
        Returns:
            df: dataframe without wrong values
        """
        df = df.copy()
        for col in df.columns:
            if col not in ["id", "message", "original", "genre"]:
                df = df[df[col].isin([0, 1])]
        self.df = df
        return df

    def split_categories(self, df):
        """
        Split categories column into separate columns
        Args:
            df: dataframe to split categories from
        Returns:
            df: dataframe with categories split into separate columns
        """
        df = df.copy()
        categories = (df["categories"].apply(lambda x: dict(item.split('-') for item in x.split(';')))).apply(pd.Series)
        categories = categories.astype(int)
        categories = categories.fillna(0)
        df.drop(columns=["categories"], inplace=True)
        df = pd.concat([df, categories], axis=1)
        self.df = df
        return df

    def remove_duplicates(self, df: pd.DataFrame):
        """
        Remove duplicates from dataframe
        Args:
            df: dataframe to remove duplicates from
        Returns:
            df: dataframe without duplicates
        """
        df = df.copy()
        subset_columns = [col for col in df.columns if col != "id"]
        df.drop_duplicates(subset=subset_columns, inplace=True)
        self.df = df
        return df

    def save_to_db(self):
        """
        Save dataframe to database
        Returns:
            None
        """
        conn = sqlite3.connect('../data/messages_categories.db')
        self.df.to_sql('clean_data', con=conn, if_exists='replace', index=False)

    def load_data(self, messages_filepath, categories_filepath):
        """
        Load data from csv files
        Args:
            messages_filepath: filepath of the messages csv file
            categories_filepath: filepath of the categories csv file

        Returns:
            df: merged dataframe
        """
        self.messages = pd.read_csv(messages_filepath, encoding='latin-1')
        self.categories = pd.read_csv(categories_filepath)
        df = self.merge_dataframes()
        return df

    def clean_data(self, df):
        """
        Perform data cleaning on dataframe
        Args:
            df: dataframe to clean
        Returns:
            df: cleaned dataframe
        """
        df = self.split_categories(df)
        df = self.drop_wrong_values(df)
        df = self.remove_duplicates(df)
        return df

    @staticmethod
    def save_data(df, database_filename):
        """
        Save dataframe to database
        Args:
            df: dataframe to save
            database_filename: database filename to save dataframe to
        Returns:
            None
        """
        conn = sqlite3.connect(database_filename)
        df.to_sql('clean_data', con=conn, if_exists='replace', index=False)


def main():
    if True:
    # if len(sys.argv) == 4:

        # messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        messages_filepath = "disaster_messages.csv"
        categories_filepath = "disaster_categories.csv"
        database_filepath = "DisasterResponse.db"

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        etl_ = ETL(messages_filepath, categories_filepath)
        df = etl_.load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = etl_.clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        etl_.save_data(df, database_filepath)

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
