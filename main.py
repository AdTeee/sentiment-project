#Step 1: Data Collection

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset for demonstration
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Create a DataFrame
tweets = positive_tweets + negative_tweets
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

df = pd.DataFrame({'tweet': tweets, 'sentiment': labels})
#----------------------------------------------------------------------------------------#
#step 2:  Data Preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Remove URLs, mentions, hashtags, and non-alphabetic characters
    text = re.sub(r"http\S+|www\S+|https\S+|@\S+|#\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Convert to lowercase
    words = [word.lower() for word in words]
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    return " ".join(words)

df['clean_tweet'] = df['tweet'].apply(preprocess_text)
#----------------------------------------------------------------------------------------#

#Step 3: Model Building

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_tweet'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
#----------------------------------------------------------------------------------------#
#Step 4: Model Training and Evaluation
# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

#----------------------------------------------------------------------------------------#
#Step 5: Model Fine-Tuning
# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Best model
best_model = grid_search.best_estimator_

# Retrain model with best parameters
best_model.fit(X_train_tfidf, y_train)

# Evaluate the fine-tuned model
y_pred_best = best_model.predict(X_test_tfidf)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Fine-tuned Accuracy: {accuracy_best:.2f}')

# Fine-tuned classification report
print(classification_report(y_test, y_pred_best))

#----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download sample dataset for demonstration
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import twitter_samples

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Create a DataFrame
tweets = positive_tweets + negative_tweets
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

df = pd.DataFrame({'tweet': tweets, 'sentiment': labels})

# Data Preprocessing
def preprocess_text(text):
    # Remove URLs, mentions, hashtags, and non-alphabetic characters
    text = re.sub(r"http\S+|www\S+|https\S+|@\S+|#\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # Tokenization
    words = word_tokenize(text)
    
    # Convert to lowercase
    words = [word.lower() for word in words]
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    
    return " ".join(words)

df['clean_tweet'] = df['tweet'].apply(preprocess_text)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_tweet'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Best model
best_model = grid_search.best_estimator_

# Retrain model with best parameters
best_model.fit(X_train_tfidf, y_train)

# Evaluate the fine-tuned model
y_pred_best = best_model.predict(X_test_tfidf)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Fine-tuned Accuracy: {accuracy_best:.2f}')
print(classification_report(y_test, y_pred_best))

#----------------------------------------------------------------------------------------#



