from gensim.models.doc2vec import TaggedLineDocument

from processors.processor import Processor
from utils import doc2vec_helper
from utils import log_helper
from utils import scikit_ml_helper
from sklearn import metrics

log = log_helper.get_logger("FactCheckProcessor")


class FactCheckProcessor(Processor):

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

        tagged_docs = TaggedLineDocument(self.labeled_articles_file_path)

        log.info("Training Doc2Vec model")
        doc2vec_model = doc2vec_helper.init_model(tagged_docs)
        log.info("Learnt vocab from training set")

        x_train = list()
        with open(self.labeled_articles_file_path) as training_set:
            for line in training_set:
                x_train.append(doc2vec_model.infer_vector(line))

        y_train = [0] * self.samples_per_class_train
        y_train.extend([1] * self.samples_per_class_train)

        x_test = list()
        with open(self.articles_source_file_path) as test_set:
            for line in test_set:
                x_test.append(doc2vec_model.infer_vector(line))

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

        ml_model_nb = scikit_ml_helper.train_gnb_classifier(x_train, y_train)
        y_pred = ml_model_nb.predict(x_test)
        log.info("Naive Bayes")
        log.info("y_pred: " + str(y_pred))
        log.info("Precision: " + str(metrics.precision_score(y_pred=y_pred, y_true=y_true)))
        log.info("Recall: " + str(metrics.recall_score(y_pred=y_pred, y_true=y_true)))

        log.info("Completed execution")
