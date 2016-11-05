import json


def get_articles_list(articles_file_path):
    with open(articles_file_path, "r") as articles_file:
        articles_data = articles_file.read()

    return json.loads(articles_data)
