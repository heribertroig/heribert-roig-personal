import sys
from udacity_courses.project_3.scripts.process_data import ETL




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
        etl = ETL(messages_filepath, categories_filepath)
        df = load_data(messages_filepath, categories_filepath, etl)

        print('Cleaning data...')
        df = clean_data(df, etl)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, etl)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
