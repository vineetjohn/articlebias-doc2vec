from nltk.tokenize import sent_tokenize, word_tokenize

article = \
    "Supermarket group Morrisons has reported its second consecutive quarter of like-for-like sales growth, with " \
    "first quarter sales up 0.7 per cent. Total sales, excluding fuel, were down 1.8 per cent as a result of the " \
    "closure of loss-making stores and the group offloading 140-store M Local convenience stores last September. " \
    "The group notes sales at its Food To Go division in the 13 weeks to May 1 rose more than 17 per cent " \
    "year-on-year, and sales from its Free-From range rose 70 per cent. Chief executive David Potts said: " \
    "\"We are encouraged by progress across our six priorities. \"There is still much to do and our colleagues are " \
    "working very hard to improve the shopping trip and save customers every penny we can. \"Customers are " \
    "responding and satisfaction levels remain ahead of last year. \"We are, of course, pleased with a second " \
    "consecutive quarter of positive LFL sales, which demonstrates our aim to stabilise trade is taking effect." \
    "\" Clive Black, analyst at Shore Capital, said: \"In a patient and methodical manner, Mr Potts has worked his " \
    "way through a long 'to do' list. \"Store standards have been improved, product on offer has been enhanced, " \
    "merchandising has been developed and customer service is better. \"Those price cuts alongside the broader " \
    "instore package are helping Morrisons to compete more effectively.\" In March, Morrisons signed a landmark " \
    "deal with US online giant Amazon to supply fresh food to its customers."


sentences = sent_tokenize(article)

for sentence in sentences:
    print sentence
    words = word_tokenize(sentence)
    print words
