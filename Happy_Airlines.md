

```python
!pip install tweepy
```


```python
!pip install vadersentiment
```

    Collecting vadersentiment
    [?25l  Downloading https://files.pythonhosted.org/packages/86/9e/c53e1fc61aac5ee490a6ac5e21b1ac04e55a7c2aba647bb8411c9aadf24e/vaderSentiment-3.2.1-py2.py3-none-any.whl (125kB)
    [K    100% |████████████████████████████████| 133kB 3.7MB/s ta 0:00:01
    [?25hInstalling collected packages: vadersentiment
    Successfully installed vadersentiment-3.2.1
    [33mYou are using pip version 10.0.1, however version 18.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



```python
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
```


```python
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
```

    {'Airline': '@SouthwestAir', 'Compound Score': 0.1840533123028391, 'Pos Score': 0.1322018927444795, 'Neg Score': 0.046757097791798104, 'Neut Score': 0.8210362776025237, 'Oldest Tweet': 'Mon Oct 22 14:56:28 +0000 2018'}
    -----
    {'Airline': '@AmericanAir', 'Compound Score': -0.002517535545023697, 'Pos Score': 0.08508214849921011, 'Neg Score': 0.08348815165876777, 'Neut Score': 0.831436018957346, 'Oldest Tweet': 'Mon Oct 22 19:15:38 +0000 2018'}
    -----
    {'Airline': '@SpiritAirlines', 'Compound Score': 0.034161563517915314, 'Pos Score': 0.1031742671009772, 'Neg Score': 0.08340228013029315, 'Neut Score': 0.813400651465798, 'Oldest Tweet': 'Tue Oct 16 00:11:53 +0000 2018'}
    -----
    {'Airline': '@Virginatlantic', 'Compound Score': 0.16393761140819962, 'Pos Score': 0.12236363636363637, 'Neg Score': 0.04580926916221034, 'Neut Score': 0.8318163992869875, 'Oldest Tweet': 'Sat Oct 20 17:16:09 +0000 2018'}
    -----
    {'Airline': '@Delta', 'Compound Score': 0.08836128000000001, 'Pos Score': 0.1225056, 'Neg Score': 0.0690944, 'Neut Score': 0.8083935999999998, 'Oldest Tweet': 'Mon Oct 22 20:52:41 +0000 2018'}
    -----
    {'Airline': '@AlaskaAir', 'Compound Score': 0.09191547231270358, 'Pos Score': 0.11038925081433223, 'Neg Score': 0.06379967426710098, 'Neut Score': 0.8258175895765473, 'Oldest Tweet': 'Sun Oct 21 17:19:59 +0000 2018'}
    -----
    {'Airline': '@KLM', 'Compound Score': 0.09932039660056659, 'Pos Score': 0.10822946175637393, 'Neg Score': 0.055793201133144475, 'Neut Score': 0.8359830028328611, 'Oldest Tweet': 'Sat Oct 20 18:59:54 +0000 2018'}
    -----
    {'Airline': '@FlyFrontier', 'Compound Score': -0.012943305785123961, 'Pos Score': 0.07499669421487604, 'Neg Score': 0.08367272727272727, 'Neut Score': 0.8413305785123967, 'Oldest Tweet': 'Tue Oct 16 02:06:53 +0000 2018'}
    -----



```python
# Create a DataFrame using results_list and display
Total_results = pd.DataFrame(results_list).set_index("Airline").round(3)
Total_results = Total_results.sort_values("Compound Score", ascending=False)
Total_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound Score</th>
      <th>Neg Score</th>
      <th>Neut Score</th>
      <th>Oldest Tweet</th>
      <th>Pos Score</th>
    </tr>
    <tr>
      <th>Airline</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@SouthwestAir</th>
      <td>0.184</td>
      <td>0.047</td>
      <td>0.821</td>
      <td>Mon Oct 22 14:56:28 +0000 2018</td>
      <td>0.132</td>
    </tr>
    <tr>
      <th>@Virginatlantic</th>
      <td>0.164</td>
      <td>0.046</td>
      <td>0.832</td>
      <td>Sat Oct 20 17:16:09 +0000 2018</td>
      <td>0.122</td>
    </tr>
    <tr>
      <th>@KLM</th>
      <td>0.099</td>
      <td>0.056</td>
      <td>0.836</td>
      <td>Sat Oct 20 18:59:54 +0000 2018</td>
      <td>0.108</td>
    </tr>
    <tr>
      <th>@AlaskaAir</th>
      <td>0.092</td>
      <td>0.064</td>
      <td>0.826</td>
      <td>Sun Oct 21 17:19:59 +0000 2018</td>
      <td>0.110</td>
    </tr>
    <tr>
      <th>@Delta</th>
      <td>0.088</td>
      <td>0.069</td>
      <td>0.808</td>
      <td>Mon Oct 22 20:52:41 +0000 2018</td>
      <td>0.123</td>
    </tr>
    <tr>
      <th>@SpiritAirlines</th>
      <td>0.034</td>
      <td>0.083</td>
      <td>0.813</td>
      <td>Tue Oct 16 00:11:53 +0000 2018</td>
      <td>0.103</td>
    </tr>
    <tr>
      <th>@AmericanAir</th>
      <td>-0.003</td>
      <td>0.083</td>
      <td>0.831</td>
      <td>Mon Oct 22 19:15:38 +0000 2018</td>
      <td>0.085</td>
    </tr>
    <tr>
      <th>@FlyFrontier</th>
      <td>-0.013</td>
      <td>0.084</td>
      <td>0.841</td>
      <td>Tue Oct 16 02:06:53 +0000 2018</td>
      <td>0.075</td>
    </tr>
  </tbody>
</table>
</div>


