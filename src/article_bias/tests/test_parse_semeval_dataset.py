import json

semeval_articles_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/" + \
    "semeval-2017-task-5-subtask-2/semeval_combined_fulltext.json"

semeval_classified_articles_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/" + \
    "semeval-2017-task-5-subtask-2/semeval_combined_fulltext_classified.json"


print "Begun"

pos_count = 0
neg_count = 0

with open(semeval_articles_file_path) as semeval_articles_file:
    semeval_articles = json.load(semeval_articles_file)

for semeval_article in semeval_articles:

    sentiment = semeval_article['sentiment']
    if sentiment == 0.0:
        semeval_article['label'] = "neu"
    elif sentiment > 0.0:
        semeval_article['label'] = "pos"
    else:
        semeval_article['label'] = "neg"

with open(semeval_classified_articles_file_path, 'w') as semeval_classified_articles_file:
    json.dump(
        obj=semeval_articles, fp=semeval_classified_articles_file, sort_keys=True, indent=4, separators=(',', ': ')
    )

print "Completed"
