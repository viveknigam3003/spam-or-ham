import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#nltk.download_shell()

messages = [line.rstrip() for line in open('smsspamcollection\SMSSpamCollection')]
messages = pd.read_csv('smsspamcollection\SMSSpamCollection', sep = '\t', names = ['label', 'message'])
messages['length'] = messages['message'].apply(len)

from nltk.corpus import stopwords
import string

def text_process (mess):
   nopunc = [c for c in mess if c not in string.punctuation] 
   nopunc = ''.join(nopunc)
   clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
   return clean_mess

messages.head()
messages['message'].head(5).apply(text_process)

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

mess4 = messages['message'][3]
bow4 = bow_transformer.transform([mess4])

#bow_transformer.get_feature_names()[4068]

messages_bow = bow_transformer.transform(messages['message'])

sparcity = (100 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)
#tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

messages_tfidf = tfidf_transformer.transform(messages_bow)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

all_pred = spam_detect_model.predict(messages_tfidf)

from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size = 0.3)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report

print (classification_report(label_test, predictions))
