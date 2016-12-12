import json

fpb_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/" + \
    "FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"

output_file_path = \
    "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/" \
    "FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/fasttext_fpb.txt"


print "Begun"

f = open(output_file_path, 'w')

with open(fpb_file_path) as fpb_file:
    for line in fpb_file:
        if ".@positive" in line:
            f.write("__label__4, " + line.replace(".@positive", ""))
        elif ".@negative" in line:
            f.write("__label__0, " + line.replace(".@negative", ""))
        else:
            f.write("__label__2, " + line.replace(".@neutral", ""))

print "Completed"
