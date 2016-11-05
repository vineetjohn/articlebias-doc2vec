from sklearn import linear_model
from sklearn.externals import joblib


def train_linear_model(x, y):

    linear_reg_model = linear_model.LinearRegression()
    linear_reg_model.fit(x, y)

    return linear_reg_model


def extract_training_parameters(doc2vec_model, sentiment_scores_dict):
    x_docvecs = list()
    y_scores = list()

    for tag in sentiment_scores_dict.keys():
        x_docvecs.append(doc2vec_model.docvecs[tag])
        y_scores.append(sentiment_scores_dict[tag])

    return x_docvecs, y_scores


def persist_model_to_disk(model, model_path):
    joblib.dump(model, model_path)
