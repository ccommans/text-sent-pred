import random
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


#Data Class
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else: #Score of 4 or 5
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
        
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
        
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

#Load Data
file_name = './books_small_10000.json'

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

#Prep Data
training, test = train_test_split(reviews, test_size=0.33, random_state=42)
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

#Vectorization
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

#Classification Algorithms
#SVM
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
#DT
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)
#GNB
clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors.toarray(), train_y)
#LR
clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

#Grid Search
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)

with open('./models/sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('./models/vectorizer.pkl', 'wb') as g:
    pickle.dump(vectorizer, g)

""" 
#Evaluation
print(clf_svm.score(test_x_vectors, test_y))
print(clf_dec.score(test_x_vectors, test_y))
print(clf_gnb.score(test_x_vectors.toarray(), test_y))
print(clf_log.score(test_x_vectors, test_y))
print("F1 Scores")
print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE]))

#New Tests
test_set = ['no good', "I found it boring", 'really enjoyed this one']
new_test = vectorizer.transform(test_set)
print(clf_svm.predict(new_test)) 
"""

