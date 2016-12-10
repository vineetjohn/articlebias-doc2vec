import json

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import file_helper
from utils import scikit_ml_helper

from processors.processor import Processor
from utils import log_helper

log = log_helper.get_logger("AmazonLineProcessorTFIDF")


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

        x_documents = list()
        y_scores = list()

        log.info("Parsing documents")
        for sentiment in sources_dict:
            review_file_path = sources_dict[sentiment]

            with open(review_file_path) as review_file:
                for line in review_file:
                    x_document = line
                    y_score = sentiment

                    x_documents.append(x_document)
                    y_scores.append(y_score)

        semeval_classified_articles_file = self.articles_source_file_path
        semeval_classified_articles = file_helper.get_articles_list(semeval_classified_articles_file)

        y_true = list()
        semeval_count = 0
        for semeval_classified_article in semeval_classified_articles:

            # article_text = semeval_classified_article['articleText']
            article_text = semeval_classified_article['title']

            if not article_text:
                continue

            semeval_count += 1
            x_documents.append(article_text)
            y_true.append(semeval_classified_article['label'])

        log.info("Initiating training for document vectors")
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        x_vectors = vectorizer.fit_transform(x_documents)
        labelled_docs = x_vectors[(-1*semeval_count):]

        log.info("Vectors have been trained")

        log.info("Training the ML models")
        ml_model_logreg = scikit_ml_helper.train_logistic_reg_classifier(x_vectors[:(-1 * semeval_count)], y_scores)
        ml_model_nb = scikit_ml_helper.train_gnb_classifier_dense(x_vectors[:(-1 * semeval_count)], y_scores)
        ml_model_svm_linear = scikit_ml_helper.train_svm_classifier(x_vectors[:(-1 * semeval_count)], y_scores)

        log.info("Saving the ML models to disk")
        scikit_ml_helper.persist_model_to_disk(ml_model_logreg, self.ml_model_file_path + ".tfidf.log_reg")
        scikit_ml_helper.persist_model_to_disk(ml_model_nb, self.ml_model_file_path + ".tfidf.nb")
        scikit_ml_helper.persist_model_to_disk(ml_model_svm_linear, self.ml_model_file_path + ".tfidf.svm_linear")

        predictions_logreg = ml_model_logreg.predict(labelled_docs)
        predictions_linearsvm = ml_model_svm_linear.predict(labelled_docs)
        predictions_nb = ml_model_nb.predict(labelled_docs)

        accuracy_logreg = \
            sklearn.metrics.accuracy_score(y_true=y_true, y_pred=predictions_logreg, normalize=True, sample_weight=None)
        accuracy_linearsvm = \
            sklearn.metrics.accuracy_score(y_true=y_true, y_pred=predictions_linearsvm, normalize=True, sample_weight=None)
        accuracy_nb = \
            sklearn.metrics.accuracy_score(y_true=y_true, y_pred=predictions_nb, normalize=True, sample_weight=None)

        log.info("accuracy_logreg: " + str(accuracy_logreg))
        log.info("accuracy_linearsvm: " + str(accuracy_linearsvm))
        log.info("accuracy_nb: " + str(accuracy_nb))

        log.info("\ncm_logreg\n" + str(scikit_ml_helper.get_confusion_matrix(y_true, predictions_logreg)))
        log.info("\ncm_linearsvm\n" + str(scikit_ml_helper.get_confusion_matrix(y_true, predictions_linearsvm)))
        log.info("\ncm_nb\n" + str(scikit_ml_helper.get_confusion_matrix(y_true, predictions_nb)))

        log.info("Completed execution")
