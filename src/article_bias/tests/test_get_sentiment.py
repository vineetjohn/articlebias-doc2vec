from src.article_bias.utils import scikit_ml_helper

ml_model_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/doc2vec_headlines.model"

model = scikit_ml_helper.get_model_from_disk(ml_model_file_path)

model.predict()
