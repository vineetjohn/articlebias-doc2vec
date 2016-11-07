from src.article_bias.utils import scikit_ml_helper
from gensim.models import Doc2Vec

doc2vec_model_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/doc2vec_headlines.model"

ml_model_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/regression.model"

doc2vec_model = Doc2Vec.load(doc2vec_model_file_path)
print doc2vec_model.docvecs['76545']

ml_model = scikit_ml_helper.get_model_from_disk(ml_model_file_path)

x_docvecs = [doc2vec_model.docvecs['76545']]
print ml_model.predict(x_docvecs)
