import json
import plotly
import pandas as pd
import joblib
import plotly.express as px

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Pie, Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    genre_counts = df['genre'].value_counts()
    fig = px.pie(genre_counts, names=genre_counts.index, values=genre_counts.values, title='Genre Distribution')

    # Create the pie chart
    fig = px.pie(genre_counts, names=genre_counts.index, values=genre_counts.values, title='Genre Distribution')

    # Extract the necessary information
    genre_labels = genre_counts.index.tolist()
    genre_percentages = [float(value) for value in genre_counts.values / genre_counts.sum() * 100]

    category_columns = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
                         'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 
                         'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death',
                         'other_aid', 'infrastructure_related', 'transport', 'buildings',
                         'electricity', 'tools', 'hospitals', 'shops', 'aid_centers',
                         'other_infrastructure', 'weather_related', 'floods', 'storm',
                         'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    
    category_counts = df[category_columns].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_counts.index,
                    values=genre_counts.values
                )
            ],
            'layout': {
                'title': 'Genre Distribution'
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Category Distribution',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()