# web-scrapping

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the two dataframes
df1 = pd.read_csv('df1.csv')
df2 = pd.read_csv('df2.csv')

# preprocess the text data in each dataframe
df1['text'] = df1['text'].str.lower()
df1['tokens'] = df1['text'].str.split()
df2['text'] = df2['text'].str.lower()
df2['tokens'] = df2['text'].str.split()

# create a TfidfVectorizer to convert each set of tokens into a numerical representation
vectorizer = TfidfVectorizer()

# create a list of all tokens from both dataframes
all_tokens = df1['tokens'].tolist() + df2['tokens'].tolist()

# fit the vectorizer on all tokens to create a vocabulary
vectorizer.fit(all_tokens)

# use the vectorizer to transform the tokens in each dataframe into numerical vectors
df1_vectors = vectorizer.transform(df1['tokens'].apply(lambda x: ' '.join(x)))
df2_vectors = vectorizer.transform(df2['tokens'].apply(lambda x: ' '.join(x)))

# compute the cosine similarity between each pair of vectors
similarity_matrix = cosine_similarity(df1_vectors, df2_vectors)

# loop through each row of the similarity matrix and find the pairs with high similarity scores
for i in range(len(df1)):
    for j in range(len(df2)):
        if similarity_matrix[i,j] > 0.8:
            # store the relationship information in a new dataframe or column
            print(f"df1 row {i} is related to df2 row {j}")
