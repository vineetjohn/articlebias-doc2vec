from random import shuffle

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def get_tagged_articles_scores(articles):

    tagged_articles = list()
    sentiment_scores_dict = dict()

    for article in articles:
        article_tag = "ART_" + str(article['id'])
        tagged_article = TaggedDocument(article['title'].split(), [article_tag])
        tagged_articles.append(tagged_article)
        sentiment_scores_dict[article_tag] = article['sentiment']

    return tagged_articles, sentiment_scores_dict


def init_model(tagged_articles):

    model = Doc2Vec(min_count=1, size=1000, iter=20, dm=0)
    model.build_vocab(tagged_articles)

    return model


def shuffle_and_train_articles(model, tagged_articles):
    shuffle(tagged_articles)
    model.train(tagged_articles)
