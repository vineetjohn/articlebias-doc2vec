import json

semeval_articles_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/" + \
    "semeval-2017-task-5-subtask-2/semeval_combined_fulltext.json"

output_file_path = "/home/v2john/Documents/amazon/books_reviews_fasttext_articles_test.txt"


print "Begun"

with open(semeval_articles_file_path) as semeval_articles_file:
    semeval_articles = json.load(semeval_articles_file)


with open(output_file_path, 'w') as test_file:
    for semeval_article in semeval_articles:
        sentiment = semeval_article['sentiment']
        if sentiment > 0.0:
            test_file.write("__label__1, " + json.dumps(semeval_article['articleText']).strip('"') + "\n")
        elif sentiment < 0.0:
            test_file.write("__label__0, " + json.dumps(semeval_article['articleText']).strip('"') + "\n")
        else:
            pass

print "Completed"
