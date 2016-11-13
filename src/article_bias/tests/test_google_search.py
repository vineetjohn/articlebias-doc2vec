import requests

url = "http://www.google.com/search?q=__KEYWORDS_PLACEHOLDER__&btnI"

final_url = url.replace("__KEYWORDS_PLACEHOLDER__", "Morrisons book second consecutive quarter of sales growth")

response = requests.get(final_url)
print(response.text)
