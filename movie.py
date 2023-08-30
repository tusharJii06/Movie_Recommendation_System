import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

class MOVIERECOMMEND:
    credits_df = pd.read_csv("credits.csv")
    movies_df = pd.read_csv("movies.csv")

    movies_df = movies_df.merge(credits_df, on = 'title')

    movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    movies_df.dropna(inplace = True)
    movies_df.drop_duplicates(inplace = True)

    def convert(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    movies_df['genres'] = movies_df['genres'].apply(convert)
    movies_df['keywords'] = movies_df['keywords'].apply(convert)
    movies_df.head()

    def convert3(self, obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    movies_df['cast'] = movies_df['cast'].apply(convert3)

    def fetch_director(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    movies_df['crew'] = movies_df['crew'].apply(fetch_director)

    movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())

    movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])

    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

    new_df = movies_df[['movie_id', 'title', 'tags']]

    new_df['tags'] = new_df['tags'].apply(lambda x:' '.join(x))

    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


    cv = CountVectorizer(max_features = 5000, stop_words = 'english')

    vectors = cv.fit_transform(new_df['tags']).toarray()

    ps = PorterStemmer()

    def stem(self, text):
        y = []
        for i in text.split():
            y.append(self.s.stem(i))
        return " ".join(y)

    new_df['tags'] = new_df['tags'].apply(stem)

    similarity = cosine_similarity(vectors)

    sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x: x[1])[1:6]

    def recommend(self, movie):
        movie_index = self.new_df[self.new_df['title'] == movie].index[0]
        distances = self.similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
        return movie_list

        # for i in movie_list:
        #     print(self.new_df.iloc[i[0]].title)

    recommend('Avatar')

