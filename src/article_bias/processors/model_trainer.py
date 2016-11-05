from src.article_bias.utils import log_helper, file_helper, doc2vec_helper, scikit_ml_helper
from src.article_bias.processors.processor import Processor


class ModelTrainer(Processor):

    def __init__(self, articles_file_path, model_file_path):

        self.log = log_helper.get_logger('ModelTrainer')
        self.articles_file_path = articles_file_path
        self.model_file_path = model_file_path
        self.shuffle_count = 0

    def process(self):

        self.log.info("Commencing execution")

        # Convert articles file into a Tagged documents for doc2vec
        articles = file_helper.get_articles_list(self.articles_file_path)
        tagged_articles, sentiment_scores_dict = doc2vec_helper.get_tagged_articles_scores(articles)

        # model initialization and vocab building
        self.log.info("Initializing the doc2vec model ...")
        doc2vec_model = doc2vec_helper.init_model(tagged_articles)

        # shuffling and training the model
        self.log.info("Training the doc2vec model ...")
        for i in range(self.shuffle_count):
            self.log.info("Shuffles remaining: " + str(self.shuffle_count - i))
            doc2vec_helper.shuffle_and_train_articles(doc2vec_model, tagged_articles)

        # Extracing parameters for and training the ML model
        x_docvecs, y_scores = scikit_ml_helper.extract_training_parameters(doc2vec_model, sentiment_scores_dict)
        self.log.info("Training the ML model ...")
        ml_model = scikit_ml_helper.train_linear_model(x_docvecs, y_scores)

        # saving the model to disk
        # doc2vec_model.save(self.model_file_path)
        scikit_ml_helper.persist_model_to_disk(ml_model, self.model_file_path)

        self.log.info("Completed execution")
