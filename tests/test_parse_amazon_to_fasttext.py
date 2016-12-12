positive_review_file = \
    {
        "path": "/home/v2john/Documents/amazon/books_reviews_pos.txt",
        "label": 1
    }
negative_review_file = \
    {
        "path": "/home/v2john/Documents/amazon/books_reviews_neg.txt",
        "label": 0
    }

output_file_path = "/home/v2john/Documents/amazon/books_reviews_fasttext_input.txt"

print("Begun")

f = open(output_file_path, 'w')

for review_file_dict in [positive_review_file, negative_review_file]:
    path = review_file_dict.get('path')
    label = "__label__" + str(review_file_dict.get('label'))

    with open(path) as review_file:
        for line in review_file.readlines():
            f.write(label + ", " + line)

f.close()

print("Completed")
