# Happy_Airplane

Main file to review: Happy_Airlines.md (for readable markdown version). 
File may be executed in python or, as originally written in Jupyter Notebook.

This script uses Tweepy (a Python library for working with the Twitter API) and Vader Sentiment Analysis to review the last 1,000 tweets about 8 popular airlines (Southwest, American, Spirit, Virgin, Delta, Alaska, and Frontier).

The script establishes several filters for the tweets it gathers...must be from a person with minimum 5 and maximum 10,000 tweets, with maximum of 2,500 followers. 

For each airline, the script reads in 1,000 tweets, and uses sentiment analysis to store lists of each tweet's compount score, positive score, negative score, and neutral score. Then the script gathers the means of each of these values, and stores them in a dictionary for each airline, and finally, combines those dictionaries into a dataframe.
