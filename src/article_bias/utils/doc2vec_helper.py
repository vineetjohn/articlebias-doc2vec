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

    model = Doc2Vec(min_count=1, size=1000, iter=20, dm=0, workers=8)
    model.build_vocab(tagged_articles)

    return model


def shuffle_and_train_articles(model, tagged_articles):
    shuffle(tagged_articles)
    model.train(tagged_articles)


def get_tagged_articles_veriday(articles):

    tagged_documents = list()

    for article in articles:

        if not article:
            continue

        article_text = article['value'].strip()
        article_id = str(article['articleid'].strip())
        article_version = str(article['version'].strip())
        article_sentences = article_text.split('.')

        annotations = article['annotations']
        article_orgs = set(x['entity'] for x in annotations if x['tag'] == "ORG")

        for article_sentence in article_sentences:
            if not any(org in article_sentence for org in article_orgs):
                continue

            tagged_sentence = TaggedDocument(words=article_sentence.split(), tags=[article_id, article_version])
            tagged_documents.append(tagged_sentence)

    return tagged_documents
