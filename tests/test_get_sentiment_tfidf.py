import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import scikit_ml_helper, file_helper

ml_model_file_path = "/home/v2john/Documents/amazon/models/books_ml_tfidf.model.tfidf.log_reg"
ml_model = scikit_ml_helper.get_model_from_disk(ml_model_file_path)


semeval_classified_articles_file = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/" + \
    "semeval-2017-task-5-subtask-2/semeval_combined_fulltext_classified.json"


semeval_classified_articles = file_helper.get_articles_list(semeval_classified_articles_file)

count = 0
classified_veriday_articles = list()
x_documents = list()
y_true = list()

for semeval_classified_article in semeval_classified_articles:

    print 'working on article ' + str(count)
    count += 1

    article_text = semeval_classified_article['articleText']
    # article_text = semeval_classified_article['title']
    if not article_text:
        continue

    x_documents.append(article_text)
    y_true.append(semeval_classified_article['label'])


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
x_vectors = vectorizer.fit_transform(x_documents)
# predictions = ml_model.predict(np.asarray(x_vectors).reshape(1, -1))
predictions = ml_model.predict(x_vectors)

accuracy = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=predictions, normalize=True, sample_weight=None)

print "accuracy"
print accuracy
