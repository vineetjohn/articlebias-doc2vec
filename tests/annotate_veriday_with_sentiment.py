import json

from gensim.models import Doc2Vec

from utils import scikit_ml_helper


def get_sentiment_from_rating(rating):
    if rating > 4.0:
        return "Positive"
    elif rating < 2.0:
        return "Negative"

    return "Neutral"


doc2vec_model_file_path = "/home/v2john/Documents/amazon/books_doc2vec.model"
ml_model_file_path = "/home/v2john/Documents/amazon/books_ml.model"
veriday_articles_file = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/Veriday/annotated/402-articles-manulife-REJECT.json"
veriday_articles_annotated_file = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/" + \
    "ResearchProject/Veriday/annotated/402-articles-manulife-REJECT_with_ratings.json"

doc2vec_model = Doc2Vec.load(doc2vec_model_file_path)
ml_model = scikit_ml_helper.get_model_from_disk(ml_model_file_path)

with open(veriday_articles_file) as articles_file:
    articles = json.loads(articles_file.read())

print("There are " + str(len(articles)) + " articles to annotate")

for article in articles:
    identifier = str(article['articleid'].strip()) + "-" + str(article['version'].strip())
    article['annotations'] = None
    try:
        x_docvec = doc2vec_model.docvecs[identifier]
        y_scores = ml_model.predict([x_docvec])
        article['prediction_rating'] = y_scores[0]
        article['prediction_sentiment'] = get_sentiment_from_rating(article['prediction_rating'])
        print(article)
    except Exception as e:
        print("Skipping " + identifier)
        print e
        article['prediction_rating'] = ""
        article['prediction_sentiment'] = ""

print("Dumping to file")

with open(veriday_articles_annotated_file, 'w') as outfile:
    json.dump(obj=articles, fp=outfile,  sort_keys=True, indent=4, separators=(',', ': '))
print("Completed")
