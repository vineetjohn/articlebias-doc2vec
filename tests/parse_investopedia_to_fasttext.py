import json

input_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/" + \
    "Veriday/investopedia/scraped_terms_investopedia.txt"

output_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/" + \
    "Veriday/investopedia/investopedia_fasttext.txt"

f = open(output_file_path, 'w')
f.write("__label__0 ")

count = 0
length = 0
docs_to_batch = 10

with open(input_file_path) as input_file:

    for line in input_file:
        content = json.dumps(json.loads(line)['definition'])[1:-1]
        f.write(content)
        length += len(content)
        count += 1

        if count % 10 == 0:
            f.write("\n__label__0 ")
        else:
            f.write(" ")


print(str(count) + " total documents")
print(str(length/count) + " is the avg doc size")

print("Completed")
