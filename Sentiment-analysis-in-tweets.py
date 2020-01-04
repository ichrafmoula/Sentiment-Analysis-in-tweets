# Twitter Sentiment Analysis using Tweepy and TextBlob

## importer librairies

from textblob import TextBlob
import sys , tweepy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import re
import preprocessor as p
from deeppavlov import build_model, configs
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

## key twitter

consumerkey = "2TZ1K5IosiWpBB1AgRzUN2OYe"
consumerSecret = "wYB8KHMlK6IRSf7Rr1Chl1jx2chaRqzPa4HVKoZsHVWIMJ7g4d"
accessToken = "1184232235488219136-bFjHLEHvbLcBt9SJ5PqXvNhfmd2wX4"
accessTokenSecret = "3GLCtypZ0ibi2DVMY9sgQwnVuooBE90raAxtUik4luNfv"

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