from nltk.corpus import movie_reviews
import nltk 
import random

for category in movie_reviews.categories():
    print(category)
    
documents = [(list(movie_reviews.words(file_id)),category) for category in movie_reviews.categories() for file_id in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = nltk.FreqDist([w.lower() for w in movie_reviews.words()])

print(len(list(set(all_words.keys()))))

word_list = list(set(all_words.keys()))[:2000]

#features function

def feature_function(movie_review_words):
    features = dict()
    for word in word_list:
        features['contains'+word] = word in movie_review_words
    return features

feature_set = [(feature_function(document),category) for document,category in documents]

test_set = feature_set[:100]
train_set = feature_set[100:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print(classifier.classify(feature_function(movie_reviews.words('pos/cv001_18431.txt'))))

print(classifier.classify(feature_function(movie_reviews.words('neg/cv999_14636.txt'))))

print(nltk.classify.accuracy(classifier,test_set)*100)

print(classifier.show_most_informative_features())