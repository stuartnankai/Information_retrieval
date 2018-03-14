#!/usr/local/bin/python
import sys, wikipedia

article = wikipedia.page(" ".join(sys.argv[1:])).content
article = article.splitlines()

article_without_headlines = []
for paragraph in article:
    if(paragraph != '' and paragraph[0] != '='):
        article_without_headlines.append(paragraph)

for i, paragraph in enumerate(article_without_headlines):
    article_without_headlines[i] = paragraph.replace('...', '.').replace('!', '.').replace('?', '.').split('. ')
    article_without_headlines[i][-1] = article_without_headlines[i][-1][:-1]

print article_without_headlines
