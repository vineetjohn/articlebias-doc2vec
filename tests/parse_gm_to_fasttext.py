import os
import json

dir_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/Veriday/global-mail-5000/"

output_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/Veriday/all_gm_articles.txt"


f = open(output_file_path, 'w')
count = 0
length = 0
for filename in os.listdir(dir_path):
    file_path = os.path.join(dir_path, filename)
    # print("checking " + file_path)

    with open(file_path) as directory_file:
        gm_article = json.load(directory_file)

        # print(gm_article)
        if "content" in gm_article:
            content = gm_article["content"]
            for text_object in content:
                if "text" in text_object:
                    text_content = json.dumps(text_object["text"])
                    f.write("__label__1 " + text_content[1:-1] + "\n")
                    length += len(text_content)
                    count += 1
                    break

print("Total articles: " + str(count))
print("Avg doc length: " + str(length / count) + " characters")
