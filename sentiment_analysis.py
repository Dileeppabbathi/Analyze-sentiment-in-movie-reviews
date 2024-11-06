import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Sample reviews data
reviews = pd.DataFrame({
    'review': ["I loved this movie!", "It was terrible.", "An okay film.", "Best movie ever!", "Not good at all."]
})

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Apply the sentiment analysis to each review
reviews['sentiment_score'] = reviews['review'].apply(lambda review: sid.polarity_scores(review)['compound'])
reviews['sentiment'] = reviews['sentiment_score'].apply(lambda score: 'positive' if score > 0 else 'negative')

# Print the results
print(reviews)
