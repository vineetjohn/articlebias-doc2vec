import json

source_file_path = "/home/v2john/Documents/amazon/reviews_Books_5.json"
positive_articles_file_path = "/home/v2john/Documents/amazon/books_reviews_pos.txt"
negative_articles_file_path = "/home/v2john/Documents/amazon/books_reviews_neg.txt"

print "Begun"

target_review_count = 100000

pos_count = 0
neg_count = 0

pos = open(positive_articles_file_path, 'w')
neg = open(negative_articles_file_path, 'w')

with open(source_file_path) as all_amazon_reviews:

    for line in all_amazon_reviews:
        review = json.loads(line)

        if int(review['overall']) == 5 and pos_count < target_review_count:
            pos.write(review['reviewText'] + "\n")
            pos_count += 1
        elif int(review['overall']) == 1 and neg_count < target_review_count:
            neg.write(review['reviewText'] + "\n")
            neg_count += 1

        if pos_count >= target_review_count and neg_count >= target_review_count:
            break

print "Completed"
