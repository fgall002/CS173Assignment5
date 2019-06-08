import sys
import pickle
import string
import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# functions for cleaning up text
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split()]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 1]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = " ".join(text)
    return text

# load review input file
filename = sys.argv[1]
with open(filename, 'r') as f:
    review = f.read()

# create dataframe for the single review
df = pd.DataFrame(data={'review': [review]})

# clean up data
df["review_clean"] = df["review"].apply(lambda x: clean_text(x))

# add sentiment anaylsis columns
sid = SentimentIntensityAnalyzer()
df["sentiments"] = df["review"].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

# add counts columns
df['char_count'] = df['review'].apply(lambda x : len(x))
df['word_count'] = df['review'].apply(lambda x : len(x.split(" ")));

# load up all trained models
with open('models/rfc.pickle', 'rb') as f:
    rfc = pickle.load(f)

with open('models/doc2vec_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf.pickle', 'rb') as f:
    tfidf = pickle.load(f)

# transform each document into a vector data
doc2vec_df = df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
df = pd.concat([df, doc2vec_df], axis=1)

# add tf-idfs columns
tfidf_result = tfidf.transform(df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df, tfidf_df], axis=1)

# predict positive or negative
test = np.array(df.drop(['review', 'review_clean'], axis=1))

# print results
print('Results-')
sentiment = 'Positive'
if (rfc.predict(test)[0] == 1):
    sentiment = 'Negative'
print('sentiment:', sentiment)
print('probability of negative:', rfc.predict_proba(test)[0][1])
