import pandas as pd

from sklearn import linear_model, svm
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def train_linear_model(x, y):

    linear_reg_model = linear_model.LinearRegression()
    linear_reg_model.fit(x, y)

    return linear_reg_model


def train_svm_regressor(x, y):

    svm_regressor = svm.LinearSVR()
    svm_regressor.fit(x, y)

    return svm_regressor


def train_svm_classifier(x, y):

    svm_classifier = svm.LinearSVC()
    svm_classifier.fit(x, y)

    return svm_classifier


def train_logistic_reg_classifier(x, y):

    class_weight = {0: 1, 1: 10}

    logistic_reg_classifier = linear_model.LogisticRegression(class_weight=class_weight, n_jobs=8)
    logistic_reg_classifier.fit(x, y)

    return logistic_reg_classifier


def train_gnb_classifier(x, y):

    gnb_classifier = GaussianNB()
    gnb_classifier.fit(x, y)

    return gnb_classifier


def train_gnb_classifier_dense(x, y):

    gnb_classifier = MultinomialNB()
    gnb_classifier.fit(x, y)

    return gnb_classifier


def extract_training_parameters(doc2vec_model, sentiment_scores_dict):
    x_docvecs = list()
    y_scores = list()

    for tag in sentiment_scores_dict.keys():
        x_docvecs.append(doc2vec_model.docvecs[tag])
        y_scores.append(sentiment_scores_dict[tag])

    return x_docvecs, y_scores


def persist_model_to_disk(model, model_path):
    joblib.dump(model, model_path)


def get_model_from_disk(model_path):
    return joblib.load(model_path)


def get_confusion_matrix(y_true, y_predicted):

    y_actu = pd.Series(y_true, name='Actual')
    y_pred = pd.Series(y_predicted, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)

    return df_confusion
