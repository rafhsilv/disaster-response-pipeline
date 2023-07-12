import re
import pandas as pd
import numpy as np
import sys

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.
    
    Args:
    messages_filepath: str, path to the messages dataset
    categories_filepath: str, path to the categories dataset
    
    Returns:
    df: DataFrame, merged dataset containing messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')


def clean_data(df):
    """
    Clean the merged dataset by splitting categories and converting them to numerical values.
    
    Args:
    df: DataFrame, merged dataset containing messages and categories
    
    Returns:
    df: DataFrame, cleaned dataset with categories split and converted to numerical values
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.head(1)
    col_names = row.iloc[0].apply(lambda x: x[:-2])
    categories.columns = col_names
   
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
        categories[column] = categories[column].astype(int)

    df = pd.concat([df.drop('categories', axis=1), categories], axis=1)
    df = df.drop_duplicates()
    df = df.loc[~(df == 2).any(axis=1)] #dropping all non binary rows
    
    return df



def save_data(df, database_filename):
    """
    Save the cleaned dataset to an SQLite database.
    
    Args:
    df: DataFrame, cleaned dataset with categories split and converted to numerical values
    database_filename: str, path to the SQLite database file
    
    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')     


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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