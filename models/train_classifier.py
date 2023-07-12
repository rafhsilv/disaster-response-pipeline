import sys
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Load data from the SQLite database and return the features, labels, and category names.
    Args:
    database_filepath: str, path to the SQLite database file
    Returns:
    X: pandas DataFrame, features (messages)
    y: pandas DataFrame, labels (categories)
    y.columns: Index, category names
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages_categories', engine) 
    X = df['message']
    y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    return X, y, y.columns

def tokenize(text):
    """
    Tokenize and lemmatize the input text.
    Args:
    text: str, input text
    Returns:
    clean_tokens: list of str, tokenized and lemmatized words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline with TfidfVectorizer and MultiOutputClassifier.
    Returns:
    cv: GridSearchCV object, the model with the best parameters found by grid search
    """
    text_pipeline = Pipeline([
        ('tfidfv', TfidfVectorizer())
    ])

    classifier = MultiOutputClassifier(RandomForestClassifier())
    
    pipeline = Pipeline([
        ('text_pipeline', text_pipeline),
        ('clf', classifier)
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'text_pipeline__vect__ngram_range': ((1,1), (1, 2))
    }
  
    cv = GridSearchCV(pipeline, param_grid=parameters)
  
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model by printing the classification report for each category.
    Args:
    model: trained model
    X_test: pandas DataFrame, test features
    Y_test: pandas DataFrame, test labels
    category_names: Index, category names
    """
    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f"Classification report for {category}:")
        print(classification_report(Y_test[category], y_pred[:, i]))
        
def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    Args:
    model: trained model
    model_filepath: str, path to save the pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()