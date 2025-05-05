import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download necessary resources from nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the Lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    words = text.split()

    # Remove stopwords and lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Rejoin words to form the cleaned text
    cleaned_text = ' '.join(words)

    return cleaned_text

# Load the CSV datasets
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data_solution.csv')

# Drop the unnecessary 'Unnamed' column if it exists
train_data_cleaned = train_data.drop(columns=['Unnamed: 0'], errors='ignore')
test_data_cleaned = test_data.drop(columns=['Unnamed: 0'], errors='ignore')

# Combine TITLE and DESCRIPTION into a single feature for preprocessing
train_data_cleaned['TEXT'] = train_data_cleaned['TITLE'] + " " + train_data_cleaned['DESCRIPTION']
test_data_cleaned['TEXT'] = test_data_cleaned['TITLE'] + " " + test_data_cleaned['DESCRIPTION']

# Apply preprocessing to both the training and test datasets
train_data_cleaned['PREPROCESSED_TEXT'] = train_data_cleaned['TEXT'].apply(preprocess_text)
test_data_cleaned['PREPROCESSED_TEXT'] = test_data_cleaned['TEXT'].apply(preprocess_text)

# 1. Calculate vocab_size (number of unique words in the dataset)
all_text = ' '.join(train_data_cleaned['PREPROCESSED_TEXT'])  # Combine all text into one large string
tokens = all_text.split()  # Tokenize by whitespace
vocab_size = len(set(tokens))  # Number of unique words in the vocabulary

# 2. Calculate max_length (length of the longest description)
sequence_lengths = train_data_cleaned['PREPROCESSED_TEXT'].apply(lambda x: len(x.split()))
max_length = int(np.percentile(sequence_lengths, 95))  # 95th percentile length

# 3. Calculate num_classes (number of unique genres)
num_classes = len(train_data_cleaned['GENRE'].unique())

# 4. Define training_size (number of training samples you want to use)
training_size = len(train_data_cleaned)

# Output the calculated hyperparameters
print(f"vocab_size: {vocab_size}")
print(f"max_length: {max_length}")
print(f"num_classes: {num_classes}")
print(f"training_size: {training_size}")

# 5. Tokenize the text data using TF-IDF or Keras Tokenizer
vocab_size = 10000  # Limit vocab_size to top 10,000 most frequent words
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data_cleaned['PREPROCESSED_TEXT'])

training_sequences = tokenizer.texts_to_sequences(train_data_cleaned['PREPROCESSED_TEXT'])
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding='post', truncating='post')

testing_sequences = tokenizer.texts_to_sequences(test_data_cleaned['PREPROCESSED_TEXT'])
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding='post', truncating='post')

# Save the cleaned data to new CSV files
train_data_cleaned.to_csv('train_data_cleaned.csv', index=False)
test_data_cleaned.to_csv('test_data_cleaned.csv', index=False)
