import json

from gensim.models.doc2vec import Doc2Vec
from utils import scikit_ml_helper
from utils import log_helper

doc2vec_model_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/Veriday/2class/models/doc2vec.model"

ml_model_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/Veriday/2class/models/ml.model.d2v.logreg"

veriday_articles_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/Veriday/annotated/all_articles.json"

veriday_predicted_articles_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/Veriday/annotated/all_articles_predicted.json"

log = log_helper.get_logger("VeridayPredict2Class")


def load_models():
    doc2vec_model = Doc2Vec.load(doc2vec_model_path)
    ml_model = scikit_ml_helper.get_model_from_disk(ml_model_path)
    return doc2vec_model, ml_model

log.info("Begun execution")
doc2vec_model, ml_model = load_models()
log.info("Models loaded")

with open(veriday_articles_path) as veriday_articles_file:
    veriday_articles = json.load(veriday_articles_file)

log.info("Total veriday article count: " + str(len(veriday_articles)))

predicted_veriday_articles = list()
x_vectors = list()
predicted_veriday_article = dict()

counter = 0
for veriday_article in veriday_articles:
    if veriday_article['value']:
        log.info("Vectorizing article " + str(counter))

        predicted_veriday_article['articleid'] = veriday_article['articleid'].strip()
        predicted_veriday_article['value'] = veriday_article['value'].strip()

        x_vectors.append(doc2vec_model.infer_vector(predicted_veriday_article['value']))

        predicted_veriday_articles.append(predicted_veriday_article)
        counter += 1

log.info("Veriday article have been vectorized")

y_pred = ml_model.predict(x_vectors)
log.info("Predictions have been made")

for i in xrange(len(predicted_veriday_articles)):
    if y_pred[i] == 0:
        predicted_veriday_articles[i]['prediction'] = "objective"
    elif y_pred[i] == 1:
        predicted_veriday_articles[i]['prediction'] = "subjective"
    else:
        log.error("Invalid prediction made for article " + \
                  predicted_veriday_articles[i]['articleid'] + \
                  ": " + str(y_pred))
log.info("Articles have been annotated with predictions")

with open(veriday_predicted_articles_path, 'w') as veriday_predicted_articles_file:
    json.dump(obj=predicted_veriday_articles, fp=veriday_predicted_articles_file,
              sort_keys=True, indent=4, separators=(',', ': '))
log.info("Predictions have been dumped to file")
