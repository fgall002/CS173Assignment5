import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# load prepped_data
df = pd.read_csv('data/prepped_data.csv')
df = df.drop(['Unnamed: 0', 'word_count.1'], axis=1)
print('done loading data')

# load features
with open('features.pickle', 'rb') as f:
    features = pickle.load(f)

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(df[features], df['is_negative'], test_size=0.15, random_state=42)

# traing Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42) #, verbose=2)
rf.fit(X_train, y_train)
print('done training')

# save random forest classifier
with open('models/rfc.pickle', 'wb') as f:
		pickle.dump(rf, f)

# show feature importance
print('number of features:', len(features))
fi_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_[:len(features)]}).sort_values("importance", ascending=False)
print(fi_df.head(20))

# ROC curve
y_pred = [x[1] for x in rf.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
roc_auc = auc(fpr, tpr)

# plot graph
plt.figure(1, figsize = (8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.savefig('roc.png')
plt.show()
