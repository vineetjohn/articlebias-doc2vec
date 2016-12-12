import csv
import json

tweets_file_path = \
    "/home/v2john/Documents/twitter/all_twitter_data.csv"

output_file_path = "/home/v2john/Documents/twitter/fasttext_tweets.txt"

print("Begun")

f = open(output_file_path, 'w')

count = 1
with open(tweets_file_path) as tweets_file:
    reader = csv.reader(tweets_file, delimiter=',')
    for row in reader:
        label = "__label__" + str(row[0])
        tweet_text = row[5]
        f.write(label + ", " + tweet_text + "\n")
        print("read line " + str(count))
        count += 1

f.close()

print("Completed")
