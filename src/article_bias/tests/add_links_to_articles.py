import json
from time import sleep

from google import search
from src.article_bias.utils import file_helper

input_document = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/" + \
    "semeval_task/semeval-2017-task-5-subtask-2/combined.json"

output_document = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/" + \
    "semeval_task/semeval-2017-task-5-subtask-2/combined_with_links.json"


def get_url_from_headline(headline):

    sleep(5)

    for URL in search(headline, num=1, stop=1):
        return URL

    print("failed for " + headline)

articles = file_helper.get_articles_list(input_document)

for article in articles:
    print("processing article id " + str(article['id']))

    try:
        url = get_url_from_headline(article['title'])
    except Exception as e:
        print(e)
        break

    article['link'] = url

json.dump(articles, open(output_document,'w'))
print("done")
