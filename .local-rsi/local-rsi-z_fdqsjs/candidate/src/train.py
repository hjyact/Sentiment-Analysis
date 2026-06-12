from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.features import extract_features
from src.config import MODEL_PATH
import joblib


def main():
    # Load data
    from datasets import load_dataset
    dataset = load_dataset('fancyzhx/amazon_polarity')
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    dev_texts = dataset['dev']['text']
    dev_labels = dataset['dev']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # Extract features
    X_train = extract_features(train_texts)
    X_dev = extract_features(dev_texts)
    X_test = extract_features(test_texts)

    # Train model
    model = LogisticRegression(C=0.1, solver='liblinear', max_iter=1000)
    model.fit(X_train, train_labels)

    # Evaluate model
    train_pred = model.predict(X_train)
    dev_pred = model.predict(X_dev)
    test_pred = model.predict(X_test)

    train_report = classification_report(train_labels, train_pred, output_dict=True)
    dev_report = classification_report(dev_labels, dev_pred, output_dict=True)
    test_report = classification_report(test_labels, test_pred, output_dict=True)

    print('Train accuracy:', train_report['accuracy'])
    print('Train precision:', train_report['macro avg']['precision'])
    print('Train recall:', train_report['macro avg']['recall'])
    print('Train F1 score:', train_report['macro avg']['f1-score'])

    print('Dev accuracy:', dev_report['accuracy'])
    print('Dev precision:', dev_report['macro avg']['precision'])
    print('Dev recall:', dev_report['macro avg']['recall'])
    print('Dev F1 score:', dev_report['macro avg']['f1-score'])

    print('Test accuracy:', test_report['accuracy'])
    print('Test precision:', test_report['macro avg']['precision'])
    print('Test recall:', test_report['macro avg']['recall'])
    print('Test F1 score:', test_report['macro avg']['f1-score'])

    # Save model
    joblib.dump(model, MODEL_PATH)

if __name__ == '__main__':
    main()