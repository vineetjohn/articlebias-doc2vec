import sklearn
from gensim.models import Doc2Vec
from nltk import sent_tokenize, word_tokenize

from src.article_bias.utils import scikit_ml_helper, file_helper

doc2vec_model_file_path = \
    "/home/v2john/Documents/amazon/models/books_doc2vec.model"

ml_model_file_path = \
    "/home/v2john/Documents/amazon/models/books_ml_docvec.model.docvec.nb"

doc2vec_model = Doc2Vec.load(doc2vec_model_file_path)
ml_model = scikit_ml_helper.get_model_from_disk(ml_model_file_path)


semeval_classified_articles_file = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/" + \
    "semeval-2017-task-5-subtask-2/semeval_combined_fulltext_classified.json"


semeval_classified_articles = file_helper.get_articles_list(semeval_classified_articles_file)

article_vectors = list()
count = 0
y_true = list()

for semeval_classified_article in semeval_classified_articles:

    print 'working on article ' + str(count)
    count += 1

    article_text = semeval_classified_article['articleText']
    # article_text = semeval_classified_article['title']
    if not article_text:
        continue

    all_words = list()
    sentences = sent_tokenize(article_text)
    for sentence in sentences:
        all_words.extend(word_tokenize(sentence))

    article_vector = doc2vec_model.infer_vector(doc_words=all_words)
    article_vectors.append(article_vector)

    y_true.append(semeval_classified_article['label'])

predictions = ml_model.predict(article_vectors)


accuracy = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=predictions, normalize=True, sample_weight=None)

print "accuracy"
print accuracy
