import requests
from google import search, get_page
from readability import Document

for url in search('EasyJet attracts more passengers in June but still lags Ryanair', num=1, stop=1):

    print(url)
    page = get_page(url)
    # print(page)

    response = requests.get(url)
    # print(response.text)

    doc = Document(response.text)
    print(doc.content())
