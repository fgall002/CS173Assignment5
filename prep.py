import string
import pickle
import pandas as pd
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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

# load raw data
df = pd.read_csv('data/Reviews.csv')
df['is_negative'] = df['Score'].apply(lambda x: 1 if x < 3 else 0)
df['review'] = df['Text']
df = df[['review', 'is_negative']]
df = df.sample(frac=0.1, replace=False, random_state=17)
print('done loading data')

# clean up data
df["review_clean"] = df["review"].apply(lambda x: clean_text(x))

# add sentiment columns ['pos', 'neg', 'neu', 'compound']
sid = SentimentIntensityAnalyzer()
df["sentiments"] = df["review"].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

# add counts columns
df['char_count'] = df['review'].apply(lambda x : len(x))
df['word_count'] = df['review'].apply(lambda x : len(x.split(" ")));

# create doc2vec vector columns
documents = [TaggedDocument(doc,[i]) for i, doc in enumerate(df["review_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4) 

# save doc2vec model
with open('models/doc2vec_model.pickle', 'wb') as f:
    pickle.dump(model, f)

# transform each document into a vector data
doc2vec_df = df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
df = pd.concat([df, doc2vec_df], axis=1)
print('done doc2vec')

# add tf-idfs columns
tfidf = TfidfVectorizer(min_df=10)
tfidf_result = tfidf.fit_transform(df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df, tfidf_df], axis=1)
print('done tfidfVectorizer')

# save tfidfVectorizer
with open('models/tfidf.pickle','wb') as f:
    pickle.dump(tfidf, f)

# show is_negative distribution
print('negative distribution')
print(df["is_negative"].value_counts(normalize=True))

# create final features for classifier
label = 'is_negative'
features = [c for c in df.columns if c not in [label, "review", "review_clean"]]

# save feature list
with open('features.pickle', 'wb') as f:
    pickle.dump(features, f)

# add label to store it with features
features.append(label)

# save prepped_data
prepped_df = df[features]
prepped_df.to_csv('data/prepped_data.csv')
print('done')
