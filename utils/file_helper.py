import itertools
import json

from entities.classified_tagged_line_document import ClassifiedTaggedLineDocument


def get_articles_list(articles_file_path):
    with open(articles_file_path, "r") as articles_file:
        articles_data = articles_file.read()

    return json.loads(articles_data)


def get_reviews_iterator(sources_file_path):

    with open(sources_file_path) as source_cfg:
        sources_dict = json.load(source_cfg)

    positive_sentences = ClassifiedTaggedLineDocument(sources_dict['pos'], 'POS_')
    negative_sentences = ClassifiedTaggedLineDocument(sources_dict['neg'], 'NEG_')

    return izipmerge(positive_sentences, negative_sentences)


def izipmerge(a, b):
    for i, j in itertools.izip(a,b):
        yield i
        yield j
