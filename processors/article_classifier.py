from processors.processor import Processor
from utils import log_helper, file_helper, doc2vec_helper, scikit_ml_helper

log = log_helper.get_logger("ArticleClassifier")


class ArticleClassifier(Processor):

    def __init__(self, labeled_articles_source_file_path, doc2vec_model_file_path, ml_model_file_path,
                 articles_source_file_path):
        self.labeled_articles_file_path = labeled_articles_source_file_path
        self.articles_source_file_path = articles_source_file_path
        self.doc2vec_model_file_path = doc2vec_model_file_path
        self.ml_model_file_path = ml_model_file_path
        self.shuffle_count = 1

    def process(self):

        log.info("Commencing execution")

        # Get tagged articles from Semeval
        log.info("Getting Semeval articles ... ")
        semeval_articles_raw = file_helper.get_articles_list(self.labeled_articles_file_path)
        semeval_tagged_articles, document_sentiment_classes = \
            doc2vec_helper.get_tagged_semeval_articles(semeval_articles_raw)

        # model initialization and vocab building
        log.info("Initializing the doc2vec model ...")
        doc2vec_model = doc2vec_helper.init_model(semeval_tagged_articles)

        # shuffling and training the model
        log.info("Training the doc2vec model ...")
        for i in range(self.shuffle_count):
            log.info("Shuffles remaining: " + str(self.shuffle_count - i))
            doc2vec_helper.shuffle_and_train_articles(doc2vec_model, semeval_tagged_articles)

        # saving the doc2vec model to disk
        doc2vec_model.save(self.doc2vec_model_file_path)

        # Extracting parameters for and training the ML model
        x_docvecs, y_scores = scikit_ml_helper.extract_training_parameters(doc2vec_model, document_sentiment_classes)
        log.info("Training the ML model ...")
        # ml_model = scikit_ml_helper.train_linear_model(x_docvecs, y_scores)
        ml_model = scikit_ml_helper.train_gnb_classifier(x_docvecs, y_scores)

        # saving the ml model to disk
        scikit_ml_helper.persist_model_to_disk(ml_model, self.ml_model_file_path)

        log.info("Completed execution")
