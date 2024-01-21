# import libraries
import pickle
import re
import sqlite3
import sys
# from udacity_courses.project_3.scripts.train_classifier import TrainClassifier
from pathlib import Path
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class TrainClassifier:
    # Code also present in udacity_courses/project_3/scripts/train_classifier.py

    def __init__(self, database_filepath: str, sample: bool = True, x_cols: List[str] = ["message"],
                 y_cols: List[str] = [],
                 pipeline: Pipeline = None, grid_search: bool = False, grid_search_params: Dict[str, Any] = {}):

        self.database_filepath = database_filepath
        self.sample = sample
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.X, self.y = pd.DataFrame(), pd.DataFrame()
        self.pipeline = pipeline
        self.grid_search = grid_search
        self.grid_search_params = grid_search_params

        self.data = pd.DataFrame()

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_test_split()
        if not self.pipeline:
            self.build_pipeline()
        if self.grid_search:
            self.tune_model(self.pipeline, self.grid_search_params)
        else:
            self.train_classifier()
        self.evaluate_model()
        self.save_model()

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # This sample is to reduce the size of the data for faster training
        if self.sample:
            data = self.data.sample(frac=0.1, random_state=42)
            print(f"Final data shape: {data.shape}")
        if not self.y_cols:
            self.y_cols = [col for col in data.columns if col not in ["id", "message", "original", "genre"]]

        self.X = data[self.x_cols]
        self.y = data[self.y_cols]
        return self.X, self.y

    def build_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', Pipeline([
                    ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf_transformer', TfidfTransformer())
                ]), 'message'),
                # ('genre', OneHotEncoder(), ['genre'])
            ])
        classifier = MultiOutputClassifier(RandomForestClassifier())
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        self.pipeline = pipeline
        return pipeline

    def train_classifier(self):
        self.pipeline.fit(self.X_train, self.y_train)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)
        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def evaluate_model(self):
        y_predict = self.pipeline.predict(self.X_test)
        y_predict = pd.DataFrame(y_predict, columns=self.y_test.columns)
        display_results(self.y_test, y_predict)

    def tune_model(self, pipeline: Pipeline, parameters: Dict[str, Any]):
        model = grid_search(pipeline, parameters)
        model.fit(self.X_train, self.y_train)
        y_predict = model.predict(self.X_test)
        y_predict = pd.DataFrame(y_predict, columns=self.y_test.columns)
        display_results(self.y_test, y_predict)

        self.model = model

    def save_model(self, model=None, model_filepath: str = '../models/trained_pipeline.pkl'):
        filename = model_filepath
        if not model:
            model = self.pipeline
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, model_filepath: str = '../models/trained_pipeline.pkl'):
        with open(model_filepath, 'rb') as file:
            pipeline = pickle.load(file)
        self.pipeline = pipeline
        return pipeline

    def predict(self, text: str):
        text_df = pd.DataFrame([text], columns=["message"])
        y_predict = self.pipeline.predict(text_df)
        return y_predict

    def load_data(self) -> pd.DataFrame:
        conn = sqlite3.connect(self.database_filepath)
        sqlquery = """SELECT * FROM clean_data"""
        data = pd.read_sql(sqlquery, con=conn)
        self.data = data
        return data


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok in stopwords.words("english"):
            continue
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def display_results(y_test, y_pred):
    accuracies = {}
    precisions = {}
    recalls = {}
    f1_scores = {}

    for label in y_test.columns:
        true_labels = np.array(y_test[label])
        predicted_labels = np.array(y_pred[label])

        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        print(f"Confusion Matrix for Label {label}:\n{conf_matrix}\n")

        # Accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        # Precision, Recall and F1 Score
        precision = precision_score(true_labels, predicted_labels, average="weighted")
        recall = recall_score(true_labels, predicted_labels, average="weighted")
        f1 = f1_score(true_labels, predicted_labels, average="weighted")

        precisions[label] = precision
        recalls[label] = recall
        f1_scores[label] = f1
        accuracies[label] = accuracy

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Accuracy:", accuracy)

    weighted_precision = np.mean(list(precisions.values()))
    weighted_recall = np.mean(list(recalls.values()))
    weighted_f1 = np.mean(list(f1_scores.values()))
    weighted_accuracy = np.mean(list(accuracies.values()))

    print(f"\nWeighted Precision: {weighted_precision}")
    print(f"Weighted Recall: {weighted_recall}")
    print(f"Weighted F1 Score: {weighted_f1}")
    print(f"Weighted Accuracy: {weighted_accuracy}\n")

    return weighted_accuracy, accuracies, precisions, recalls, f1_scores


def grid_search(pipeline, parameters):
    # create grid search object
    cv = GridSearchCV(pipeline, parameters)

    return cv


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('message', Word2VecVectorizer(), ['message']),
            ('genre', OneHotEncoder(), ['genre'])
        ]
    )

    # Combine preprocessor with the classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ("classifier", MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, size=100, window=5, min_count=1, workers=4):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        messages = X.apply(lambda row: tokenize(row["message"]), axis=1)
        self.model = Word2Vec(sentences=messages, vector_size=self.size, window=self.window,
                              min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X):
        vectors = []
        for _, row in X.iterrows():
            message = word_tokenize(row['message'])
            message_vectors = np.array([self.model.wv[word] for word in message if word in self.model.wv])
            if message_vectors.size == 0:
                message_vectors = np.zeros((1, self.size))  # Use zero vectors for messages with no known words
            avg_vector = np.mean(message_vectors, axis=0)
            vectors.append(avg_vector)

        return np.vstack(vectors)


@staticmethod
def build_database_filepath(database_filepath: str) -> str:
    src_directory = Path(__file__).resolve().parent.parent

    # Ensure src_directory includes 'web_application'
    src_directory = src_directory / 'web_application'

    # Join src_directory with the relative path and resolve any '..'
    final_path = (src_directory / database_filepath).resolve()

    return final_path


def main():
    if len(sys.argv) == 3:
        print(sys.argv)
        database_filepath, model_filepath = sys.argv[1:]
        # database_filepath = build_database_filepath(database_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        train_classifier = TrainClassifier(database_filepath=database_filepath, sample=True)

        train_classifier.load_data()

        print('Preprocessing data...')
        train_classifier.preprocess_data()

        print('Splitting data...')
        train_classifier.train_test_split()
        
        print('Building model...')
        train_classifier.build_pipeline()
        
        print('Training model...')
        train_classifier.train_classifier()
        
        print('Evaluating model...')
        train_classifier.evaluate_model()

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        train_classifier.save_model(model_filepath=model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
