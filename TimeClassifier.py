import pandas as pd
import string
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from datetime import datetime


df = pd.read_csv('data/Reviews.csv')

df['is_negative'] = df['Score'].apply(lambda x: 1 if x < 3 else 0)

#df['is_cold_month'] = df['Time'].apply(lambda x: 1 if int(datetime.utcfromtimestamp(int(x)).strftime("%Y")) < 5 else 0)

#df['is_cold_month'] = int(datetime.utcfromtimestamp(df['Time'].astype(int)).strftime("%Y")).apply(lambda x: 1 if ( x < 3 and x > 8) else 0)

df['review'] = df['Text']

df['time'] = df['Time']

df = df[['review', 'time', 'is_negative']]

print(df.head())

df = df.sample(frac=0.1, replace=False, random_state=17)

# clean up data

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
    # lower text
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
    # join all
    text = " ".join(text)
    return text

# clean text data
df["review_clean"] = df["review"].apply(lambda x: clean_text(x))

# add sentiment anaylsis columns

sid = SentimentIntensityAnalyzer()
df["sentiments"] = df["review"].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)


df['char_count'] = df['review'].apply(lambda x : len(x))
df['word_count'] = df['review'].apply(lambda x : len(x.split(" ")));
df2 = df.copy()

df2['is_cold_month'] = df2['time'].apply(lambda x: 1 if (int(datetime.utcfromtimestamp(int(x)).strftime("%m")) < 2 or int(datetime.utcfromtimestamp(int(x)).strftime("%m")) > 9 ) else 0)

print(df.head(20))
print(df2.head(20))

print('done prepping data')


#create doc2vec vector columns
documents = [TaggedDocument(doc,[i]) for i, doc in enumerate(df["review_clean"].apply(lambda x: x.split(" ")))]

'''
train a Doc2Vec model with our text data
'''
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4) 

print('done doc2vec 2')

documents2= [TaggedDocument(doc,[i]) for i, doc in enumerate(df2["review_clean"].apply(lambda x: x.split(" ")))]

'''
train a Doc2Vec model with our text data
'''
model2 = Doc2Vec(documents2, vector_size=5, window=2, min_count=1, workers=4)

print('done doc2vec 2')


#transform each document into a vector data
doc2vec_df = df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
df = pd.concat([df, doc2vec_df], axis=1)

#add tf-idfs columns
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df, tfidf_df], axis=1)

#show is_bad_review distribution
df["is_negative"].value_counts(normalize = True)


# #wordcloud function
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# def show_wordcloud(data, title = None):
#     wordcloud = WordCloud(
#         background_color = 'white',
#         max_words = 200,
#         max_font_size = 40, 
#         scale = 3,
#         random_state = 42
#     ).generate(str(data))

#     fig = plt.figure(1, figsize = (20, 20))
#     plt.axis('off')
#     if title: 
#         fig.suptitle(title, fontsize = 20)
#         fig.subplots_adjust(top = 2.3)

#     plt.imshow(wordcloud)
#     plt.show()
    
# print wordcloud
# show_wordcloud(df["review"])

df[df["word_count"] >= 5].sort_values("pos", ascending = False)[["review", "pos"]].head(10)




#------------------------df2-------------------------------

#transform each document into a vector data
doc2vec_df2 = df2["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df2.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df2.columns]
df2 = pd.concat([df2, doc2vec_df2], axis=1)

#add tf-idfs columns
tfidf2 = TfidfVectorizer(min_df = 10)
tfidf_result2 = tfidf2.fit_transform(df2["review_clean"]).toarray()
tfidf_df2 = pd.DataFrame(tfidf_result2, columns = tfidf2.get_feature_names())
tfidf_df2.columns = ["word_" + str(x) for x in tfidf_df2.columns]
tfidf_df2.index = df2.index
df2 = pd.concat([df2, tfidf_df2], axis=1)

#show is_bad_review distribution
df2["is_negative"].value_counts(normalize = True)



# wordcloud function

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# def show_wordcloud(data, title = None):
#     wordcloud = WordCloud(
#         background_color = 'white',
#         max_words = 200,
#         max_font_size = 40, 
#         scale = 3,
#         random_state = 42
#     ).generate(str(data))

#     fig = plt.figure(1, figsize = (20, 20))
#     plt.axis('off')
#     if title: 
#         fig.suptitle(title, fontsize = 20)
#         fig.subplots_adjust(top = 2.3)

#     plt.imshow(wordcloud)
#     plt.show()
    
# # print wordcloud
# show_wordcloud(df["review"])

#highest positive sentiment review (with more than 5 words)
# df[df["word_count"] >= 5].sort_values("pos", ascending=False)[["review","pos"]].head

# #lowest negative sentiment review (with more than 5 words)
# df[df["word_count"] >= 5].sort_values("neg", ascending=False)[["review","neg"]].head

#train a random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

label = "is_negative"
ignore_cols = [label,"time", "review", "review_clean"]
features = [c for c in df.columns if c not in ignore_cols]

X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size = 0.20, random_state = 42)


rf = RandomForestClassifier(n_estimators = 100, random_state = 42, verbose=2)
rf.fit(X_train, y_train)

print('done with random forrest 1/2')
#-----------------------------------df2---------------------------------------------------
features2 = [c for c in df2.columns if c not in ignore_cols]

X_train2, X_test2, y_train2, y_test2 = train_test_split(df2[features2], df2[label], test_size = 0.20, random_state = 42)

rf2 = RandomForestClassifier(n_estimators = 100, random_state = 42, verbose=2)
rf2.fit(X_train2, y_train2)

print('done with random forrest 2/2')

# with open('rfc.pickle',"rb") as infile:
#     rf = pickle.load(infile)

with open("rfc.pickle","wb") as outfile:
    pickle.dump(rf,outfile)


print('done with random forrest')


#show feature importance
fi_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_[:len(features)]}).sort_values("importance", ascending = False)
print(fi_df.head(20))

fi_df2 = pd.DataFrame({"feature2": features2, "importance": rf2.feature_importances_[:len(features2)]}).sort_values("importance", ascending = False)
print(fi_df2.head(20))


#ROC curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
  
y_pred = [x[1] for x in rf.predict_proba(X_test)]
fpr,tpr, thresholds = roc_curve(y_test,y_pred,pos_label = 1)

roc_auc = auc(fpr, tpr)

y_pred2 = [x[1] for x in rf2.predict_proba(X_test2)]
fpr2,tpr2, thresholds2 = roc_curve(y_test2,y_pred2,pos_label = 1)

roc_auc2 = auc(fpr2, tpr2)

plt.figure(1, figsize = (5,5))
lw =2
line_fmt = ":";
plt.plot(fpr, tpr, color='blue',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr2, tpr2,line_fmt, color='pink',
         lw=lw, label='Winter Months (area = %0.2f)' % roc_auc2)

plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 1. Large Data')
plt.legend(loc="lower right")
plt.savefig('LargeData.png', bbox_inches= "tight")
plt.show()

