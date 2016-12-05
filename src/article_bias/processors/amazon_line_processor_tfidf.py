import json

from sklearn.feature_extraction.text import TfidfVectorizer

from src.article_bias.processors.processor import Processor
from src.article_bias.utils import log_helper
from src.article_bias.utils import scikit_ml_helper

log = log_helper.get_logger("AmazonLineProcessor")


class AmazonLineProcessorTfIdf(Processor):

    def __init__(self, labeled_articles_source_file_path, doc2vec_model_file_path, ml_model_file_path,
                 articles_source_file_path, shuffle_count, classification_sources_file_path):
        self.labeled_articles_file_path = labeled_articles_source_file_path
        self.articles_source_file_path = articles_source_file_path
        self.doc2vec_model_file_path = doc2vec_model_file_path
        self.ml_model_file_path = ml_model_file_path
        self.shuffle_count = shuffle_count
        self.classification_sources_file_path = classification_sources_file_path

    def process(self):

        log.info("Commencing execution")

        with open(self.classification_sources_file_path) as source_cfg:
            sources_dict = json.load(source_cfg)

        X_documents = list()
        y_scores = list()
        for sentiment in sources_dict:
            review_file_path = sources_dict[sentiment]

            with open(review_file_path) as review_file:
                for line in review_file:
                    x_document = line
                    y_score = sentiment

                    X_documents.append(x_document)
                    y_scores.append(y_score)

        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        x_vectors = vectorizer.fit_transform(X_documents)
        log.info("Vectors have been trained")

        log.info("Training the ML models")
        ml_model_logreg = scikit_ml_helper.train_logistic_reg_classifier(x_vectors, y_scores)
        ml_model_nb = scikit_ml_helper.train_gnb_classifier(x_vectors, y_scores)
        ml_model_svm_linear = scikit_ml_helper.train_svm_classifier(x_vectors, y_scores)

        log.info("Saving the ML models to disk")
        scikit_ml_helper.persist_model_to_disk(ml_model_logreg, self.ml_model_file_path + ".tfidf.log_reg")
        scikit_ml_helper.persist_model_to_disk(ml_model_nb, self.ml_model_file_path + ".tfidf.nb")
        scikit_ml_helper.persist_model_to_disk(ml_model_svm_linear, self.ml_model_file_path + ".tfidf.svm_linear")

        log.info("Completed execution")
