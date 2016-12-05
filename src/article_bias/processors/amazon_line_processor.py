from gensim import utils
from gensim.models.doc2vec import TaggedDocument

from src.article_bias.processors.processor import Processor
from src.article_bias.utils import log_helper, file_helper, doc2vec_helper

log = log_helper.get_logger("AmazonLineProcessor")


class AmazonLineProcessor(Processor):

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

        combined_iterator = file_helper.get_reviews_iterator(self.classification_sources_file_path)

        sentences = []
        for tagged_doc in combined_iterator:
            sentences.append(tagged_doc)

        doc2vec_model = doc2vec_helper.init_model(sentences)
        log.info("Learnt vocab from training set")

        for i in range(self.shuffle_count):
            log.info("Shuffles remaining: " + str(self.shuffle_count - i))
            doc2vec_helper.shuffle_and_train_articles(doc2vec_model, sentences)

        # saving the doc2vec model to disk
        doc2vec_model.save(self.doc2vec_model_file_path)

        log.info("Completed execution")
