#!pip install vadersentiment
#!pip install tweepy

# Dependencies
import tweepy
import numpy as np
import pandas as pd

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key,
                    consumer_secret,
                    access_token,
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


# Target Search Term
target_terms = ("@SouthwestAir", "@AmericanAir", "@SpiritAirlines",
                "@Virginatlantic", "@Delta", "@AlaskaAir", "@KLM", "@FlyFrontier")

# "Real Person" Filters
min_tweets = 5
max_tweets = 10000
max_followers = 2500
max_following = 2500
lang = "en"

# List to hold sentiment
results_list = []

# Loop through all target users
for target in target_terms:

    # Variable for holding the oldest tweet
    oldest_tweet = None

    # Variables for holding sentiments
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []

    # Loop through 10 times
    for x in range(10):

        # Run search around each tweet
        public_tweets = api.search(target,
                               count=100,
                               result_type="recent",
                               max_id=oldest_tweet)

        # Loop through all tweets
        for tweet in public_tweets["statuses"]: #only gets the keys that are nested within statuses.

            # Use filters to check if user meets conditions
            if (tweet["user"]["statuses_count"] > min_tweets and
                tweet["user"]["statuses_count"] < max_tweets and
                tweet["user"]["friends_count"] < max_following and
                tweet["user"]["followers_count"] < max_followers and
                tweet["user"]["lang"] == lang):

                # Run Vader Analysis on each tweet
                    results = analyzer.polarity_scores(tweet["text"])

                # Add each value to the appropriate list
                    compound_list.append(results["compound"])
                    positive_list.append(results["pos"])
                    negative_list.append(results["neg"])
                    neutral_list.append(results["neu"])

            # Set the new oldest_tweet value
            oldest_tweet = tweet["id"] - 1
            oldest_tweeter = tweet["created_at"]

    # Create a dictionary of the Average Sentiments
    airline_results = {"Airline":target,
                      "Compound Score":np.mean(compound_list),
                      "Pos Score":np.mean(positive_list),
                      "Neg Score":np.mean(negative_list),
                      "Neut Score":np.mean(neutral_list),
                      "Oldest Tweet":oldest_tweeter}

    # Print the Sentiments
    print(airline_results)
    print("-----")

    # Append the dictionary to results_list
    results_list.append(airline_results)

Total_results = pd.DataFrame(results_list).set_index("Airline").round(3)
Total_results = Total_results.sort_values("Compound Score", ascending=False)
Total_results
