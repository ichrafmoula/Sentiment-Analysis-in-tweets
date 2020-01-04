# Twitter Sentiment Analysis using Tweepy and TextBlob

## importer librairies

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import preprocessor as p
import tweepy
from deeppavlov import build_model, configs
from textblob import TextBlob
from wordcloud import WordCloud

os.environ["KERAS_BACKEND"] = "tensorflow"

## key twitter

consumerkey = *************************
consumerSecret = ***************************************
accessToken = ******************************************
accessTokenSecret = ************************************

## Authentication

auth = tweepy.OAuthHandler(consumerkey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

##input Search term
searchTerm = input("Enter keyword/hashtag to search about :")

##input Number Search Terms
noOfSearchTerms = 3200

tweets = api.search(q=searchTerm, lang="en", count=noOfSearchTerms, tweet_mode="extended")

for tweet in tweets:
    print(tweet.full_text)


## nettoyer twitter

def clean_tweets(txt):
    txt = " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", "", txt).split())
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.NUMBER)
    tweet_cleean = p.clean(txt)
    return tweet_cleean


## model insult

model = build_model(configs.classifiers.insults_kaggle_bert, download=True)

tw = []
cols = ['Text', 'polarity', 'subject', 'sentiment']
for tweet in tweets:
    txt = clean_tweets(tweet.full_text)
    analysis = TextBlob(txt)
    polarity = analysis.sentiment.polarity
    subject = analysis.sentiment.subjectivity
    s = model([tweet.full_text])
    if s[0] == 'Insult':
        sentiment = 'Insult'
    elif analysis.sentiment.polarity > 0.2:
        sentiment = 'positive'
    elif (analysis.sentiment.polarity >= -0.2) and (analysis.sentiment.polarity <= 0.2):
        sentiment = 'neutral'
    else:
        sentiment = 'negative'
    tw.append([txt, polarity, subject, sentiment])
df1 = pd.DataFrame(tw, columns=cols)

print(df1)


## color df1

def color_negative_red(val):
    if val == "positive":
        color = 'green'
    elif val == "negative":
        color = 'red'
    elif val == "Insult":
        color = 'Brown'
    elif val == "neutral":
        color = 'blue'
    else:
        color = 'whith'

    return 'background-color: %s' % color


s = df1.style.applymap(color_negative_red)
print(s)

positive = 0
negative = 0
neutral = 0
polarity = 0

for tweet in tweets:

    analysis = TextBlob(tweet.full_text)
    polarity += analysis.sentiment.polarity

    if analysis.sentiment.polarity > 0.2:
        positive += 1
    elif (analysis.sentiment.polarity <= 0.2) and (analysis.sentiment.polarity >= 0.2):
        neutral += 1
    else:
        negative += 1


def percentage(part, whole):
    temp = 100 * float(part) / float(whole)
    return format(temp)


led = len(df1)

positive = float(percentage(positive, led))
negative = float(percentage(negative, led))
neutral = float(percentage(neutral, led))
polarity = float(percentage(polarity, led))

print("How people are reacting on " + searchTerm + "by analyzing " + str(noOfSearchTerms) + " Tweets")

if polarity <= 0.2:
    print("Neutral")
elif polarity < -0.02:
    print("Negative")
elif polarity > 0.02:
    print("Positive")

labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]', 'Negative [' + str(negative) + '%]']

sizes = [positive, neutral, negative]

colors = ['yellow', 'gold', 'red']

patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches, labels)  # , loc="best")
plt.title('How people are reacting on' + searchTerm + 'by analyzing ' + str(noOfSearchTerms) + ' Tweets')
plt.axis('equal')
plt.show()

text = df1.Text
wordcloud = WordCloud(width=3000, height=2000, background_color='black', ).generate(str(text))
fig = plt.figure(figsize=(40, 30), facecolor='k', edgecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
