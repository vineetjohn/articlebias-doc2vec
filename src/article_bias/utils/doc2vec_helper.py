import json
from random import shuffle

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import sent_tokenize, word_tokenize

from src.article_bias.utils import file_helper


def get_tagged_articles_scores(articles):

    tagged_articles = list()
    sentiment_scores_dict = dict()

    for article in articles:
        article_tag = "ART_" + str(article['id'])

        all_article_text = article['title']
        if article['articleText']:
            all_article_text += " " + article['articleText']

        sentence_tokens = sent_tokenize(all_article_text)

        all_words = list()
        for sentence_token in sentence_tokens:
            all_words.extend(word_tokenize(sentence_token))

        tagged_article = TaggedDocument(all_words, [article_tag])
        tagged_articles.append(tagged_article)
        sentiment_scores_dict[article_tag] = article['sentiment']

    return tagged_articles, sentiment_scores_dict


def init_model(tagged_articles):

    model = Doc2Vec(min_count=3, size=1000, iter=20, workers=2)
    model.build_vocab(tagged_articles)

    return model


def shuffle_and_train_articles(model, tagged_articles):
    shuffle(tagged_articles)
    model.train(tagged_articles)


def get_tagged_semeval_articles(semeval_articles):

    tagged_semeval_articles = list()
    document_sentiment_classes = dict()

    for semeval_article in semeval_articles:
        if not semeval_article["articleText"]:
            continue

        # print semeval_article
        all_words = list()
        sentences = sent_tokenize(semeval_article["articleText"])
        for sentence in sentences:
            all_words.extend(word_tokenize(sentence))

        tagged_doc = TaggedDocument(words=all_words, tags=[semeval_article["id"]])
        tagged_semeval_articles.append(tagged_doc)

        document_sentiment_classes[semeval_article["id"]] = get_document_class(semeval_article["sentiment"])

    return tagged_semeval_articles, document_sentiment_classes


def get_document_class(score):
    if score > 0.33:
        return "positive"
    elif score < -0.33:
        return "negative"
    else:
        return "neutral"


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

            tagged_sentence = TaggedDocument(words=article_sentence.split(),
                                             tags=[str(article_id) + "-" + str(article_version)])
            tagged_documents.append(tagged_sentence)

    return tagged_documents


def get_tagged_amazon_reviews(input_file_path):
    """
    Parses the input review file
    :return: a list of TaggedDocs for Doc2Vec and a dict of scores
    """

    tagged_reviews = list()
    rating_dict = dict()
    for review in open(input_file_path):
        identifier, tagged_review, rating = parse_review(json.loads(review))

        tagged_reviews.append(tagged_review)
        rating_dict[identifier] = rating

    return tagged_reviews, rating_dict


def parse_review(review):
    """
    :param review: JSON object containing an Amazon review
    :return: Review Identifier, TaggedDocument for Doc2Vec usage, Review Rating
    """

    identifier = review['reviewerID'] + review['asin']
    rating = review['overall']

    review_text = review['reviewText']

    sentence_tokens = sent_tokenize(review_text)
    all_words = list()
    for sentence_token in sentence_tokens:
        all_words.extend(word_tokenize(sentence_token))

    tagged_review = TaggedDocument(words=all_words, tags=[identifier])

    return identifier, tagged_review, rating

