import json

from gensim.models import Doc2Vec
from nltk import sent_tokenize, word_tokenize

from src.article_bias.utils import scikit_ml_helper, file_helper


ml_model_file_path = "/home/v2john/books_ml.model"
ml_model = scikit_ml_helper.get_model_from_disk(ml_model_file_path)


semeval_articles_file = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/" + \
    "Veriday/annotated/all_articles.json"

semeval_classified_articles_file = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/" + \
    "Veriday/annotated/all_articles_tfidf_predictions.json"


semeval_articles = file_helper.get_articles_list(veriday_articles_file)

veriday_article_text_list = list()
veriday_article_sentiment_list = list()
for semeval_article in semeval_articles:
    veriday_article_text_list.append(veriday_article['value'])


count = 0
article_vectors = list()
classified_veriday_articles = list()
for veriday_article in veriday_articles:

    print 'working on article ' + str(count)
    count += 1

    if not veriday_article['value']:
        continue

    all_words = list()
    sentences = sent_tokenize(veriday_article['value'])
    for sentence in sentences:
        all_words.extend(word_tokenize(sentence))

    article_vector = doc2vec_model.infer_vector(doc_words=all_words)
    article_vectors.append(article_vector)

    classified_veriday_article = dict()
    classified_veriday_article['articleid'] = veriday_article['articleid']
    classified_veriday_article['value'] = veriday_article['value']
    classified_veriday_articles.append(classified_veriday_article)

predictions = ml_model.predict(article_vectors)

for i in xrange(len(predictions)):
    classified_veriday_articles[i]['class'] = predictions[i]

with open(veriday_classified_articles_file, 'w') as outfile:
    json.dump(obj=classified_veriday_articles, fp=outfile,  sort_keys=True, indent=4, separators=(',', ': '))
print("Completed")
