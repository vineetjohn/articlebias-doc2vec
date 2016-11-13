from src.article_bias.utils import scikit_ml_helper
from gensim.models import Doc2Vec

doc2vec_model_file_path = \
    "/home/v2john/Documents/amazon/books_doc2vec.model"

ml_model_file_path = \
    "/home/v2john/Documents/amazon/books_ml.model"

doc2vec_model = Doc2Vec.load(doc2vec_model_file_path)

ml_model = scikit_ml_helper.get_model_from_disk(ml_model_file_path)

x_docvecs = [doc2vec_model.docvecs['403202-30908']]
print(ml_model.predict(x_docvecs))
