import sys
from udacity_courses.project_3.scripts.train_classifier import TrainClassifier
from pathlib import Path


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
        database_filepath = build_database_filepath(database_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        train_classifier = TrainClassifier(database_filepath=database_filepath, sample=True)

        print('Loading data...')
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
