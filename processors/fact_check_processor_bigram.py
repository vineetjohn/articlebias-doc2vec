import itertools
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils import scikit_ml_helper

log = log_helper.get_logger("FactCheckProcessorBigram")


class FactCheckProcessorBigram(Processor):

    def __init__(self, labeled_articles_source_file_path, doc2vec_model_file_path, ml_model_file_path,
                 articles_source_file_path, shuffle_count, classification_sources_file_path):
        self.labeled_articles_file_path = labeled_articles_source_file_path
        self.articles_source_file_path = articles_source_file_path
        self.doc2vec_model_file_path = doc2vec_model_file_path
        self.ml_model_file_path = ml_model_file_path
        self.shuffle_count = shuffle_count
        self.classification_sources_file_path = classification_sources_file_path
        self.samples_per_class_train = 680
        self.samples_per_class_test = 50

    def process(self):

        log.info("Commencing execution")

        vectorizer = CountVectorizer(stop_words='english', ngram_range=(2,2))
        x_all = \
            vectorizer.fit_transform(
                itertools.chain(
                    file_helper.get_file_iterable(self.labeled_articles_file_path),
                    file_helper.get_file_iterable(self.articles_source_file_path)
                )
            )

        x_train = x_all[:self.samples_per_class_train * 2]

        y_train = [0] * self.samples_per_class_train
        y_train.extend([1] * self.samples_per_class_train)

        x_test = x_all[self.samples_per_class_train * 2:]

        y_true = [1] * self.samples_per_class_test
        y_true.extend([0] * self.samples_per_class_test)

        ml_model_logreg = scikit_ml_helper.train_logistic_reg_classifier(x_train, y_train)
        y_pred = ml_model_logreg.predict(x_test)
        log.info("Logistic Regression")
        log.info("y_pred: " + str(y_pred))
        log.info("Precision: " + str(metrics.precision_score(y_pred=y_pred, y_true=y_true)))
        log.info("Recall: " + str(metrics.recall_score(y_pred=y_pred, y_true=y_true)))

        ml_model_svm = scikit_ml_helper.train_svm_classifier(x_train, y_train)
        y_pred = ml_model_svm.predict(x_test)
        log.info("SVM")
        log.info("y_pred: " + str(y_pred))
        log.info("Precision: " + str(metrics.precision_score(y_pred=y_pred, y_true=y_true)))
        log.info("Recall: " + str(metrics.recall_score(y_pred=y_pred, y_true=y_true)))

        ml_model_nb = scikit_ml_helper.train_gnb_classifier_dense(x_train, y_train)
        y_pred = ml_model_nb.predict(x_test)
        log.info("Naive Bayes")
        log.info("y_pred: " + str(y_pred))
        log.info("Precision: " + str(metrics.precision_score(y_pred=y_pred, y_true=y_true)))
        log.info("Recall: " + str(metrics.recall_score(y_pred=y_pred, y_true=y_true)))

        log.info("Completed execution")
