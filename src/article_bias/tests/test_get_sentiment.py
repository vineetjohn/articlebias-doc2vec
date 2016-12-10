import sklearn
from gensim.models import Doc2Vec
from nltk import sent_tokenize, word_tokenize

from src.article_bias.utils import scikit_ml_helper, file_helper

doc2vec_model_file_path = \
    "/home/v2john/Documents/amazon/models/books_doc2vec.model"

logreg_ml_model_file_path = \
    "/home/v2john/Documents/amazon/models/books_ml_docvec.model.docvec.log_reg"
svm_linear_ml_model_file_path = \
    "/home/v2john/Documents/amazon/models/books_ml_docvec.model.docvec.svm_linear"
nb_ml_model_file_path = \
    "/home/v2john/Documents/amazon/models/books_ml_docvec.model.docvec.nb"

doc2vec_model = Doc2Vec.load(doc2vec_model_file_path)

logreg_model = scikit_ml_helper.get_model_from_disk(logreg_ml_model_file_path)
svm_model = scikit_ml_helper.get_model_from_disk(svm_linear_ml_model_file_path)
nb_model = scikit_ml_helper.get_model_from_disk(nb_ml_model_file_path)


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


predictions_logreg = logreg_model.predict(article_vectors)
predictions_svm = svm_model.predict(article_vectors)
predictions_nb = nb_model.predict(article_vectors)

accuracy_logreg = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=predictions_logreg, normalize=True, sample_weight=None)
accuracy_svm = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=predictions_svm, normalize=True, sample_weight=None)
accuracy_nb = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=predictions_nb, normalize=True, sample_weight=None)

print("\naccuracy_logreg: " + str(accuracy_logreg))
print("accuracy_linearsvm: " + str(accuracy_svm))
print("accuracy_nb: " + str(accuracy_nb))

print("\ncm_logreg\n" + str(scikit_ml_helper.get_confusion_matrix(y_true, predictions_logreg)))
print("\ncm_linearsvm\n" + str(scikit_ml_helper.get_confusion_matrix(y_true, predictions_svm)))
print("\ncm_nb\n" + str(scikit_ml_helper.get_confusion_matrix(y_true, predictions_nb)))
