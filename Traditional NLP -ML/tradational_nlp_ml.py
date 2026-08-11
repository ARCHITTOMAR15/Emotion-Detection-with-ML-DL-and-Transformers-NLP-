import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import gensim
import warnings

warnings.filterwarnings("ignore")


train=pd.read_csv("data train test validate/train.txt",sep=';',names=["text","emotion"])
test=pd.read_csv("data train test validate/test.txt",sep=";",names=["text","emotion"])


train.head()


test.head()


# check for missing values
train.isnull().sum()


test.isnull().sum()


# checking for duplicate values 
print(train.isnull().sum())
print(test.isnull().sum())


fig,axes=plt.subplots(1,2,figsize=(10,5))
#train count plot 
sns.countplot(x=train["emotion"],ax=axes[0])

sns.countplot(x=test["emotion"],ax=axes[1])
plt.show()


train["emotion"].value_counts()


test["emotion"].value_counts()


# info 
train.info()


test.info()


# Basic preprocessing steps


import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import contractions


# stop words'Stemmer'Lemmatizer
stop_words=set(stopwords.words('english'))

ps=PorterStemmer()

lemmatizer=WordNetLemmatizer()


# Creating a Single Cleaning Function


def clean_text(text):
    #lower_case
    text=text.lower()

    # contractions 
    text=contractions.fix(text)
    #removing URL
    text=re.sub(r'http\S+','',text)
    
    # remove punctuation and number 
    text=re.sub(r'[^a-zA-Z\s]','',text)
    
    #tokenization
    words=word_tokenize(text)
    
    #stop word removal
    words=[word for word in words if word not in stop_words]
    
    #Lemmatization
    words=[lemmatizer.lemmatize(word) for word in words]

    # join 
    return " ".join(words)


# train data cleaning 
train["clean_text"]=train["text"].apply(clean_text)


train.head(20)


# Test Data cleaning 
test["clean_text"]=test["text"].apply(clean_text)


train["word_count"]=train["clean_text"].apply(lambda x:len(x.split()))


test.head()


# Train EDA 
train["word_count"].describe()


sns.histplot(train["word_count"],bins=30)
plt.show()


# Test Eda 
test["word_count"]=test["clean_text"].apply(lambda x:len(x.split()))
test["word_count"].describe()


sns.histplot(test["word_count"],bins=50)
plt.show()


# TRAIN SEPRATION
X_train=train["clean_text"]
y_train=train["emotion"]


#TEST SEPRATION
X_test=test["clean_text"]
y_test=test["emotion"]


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

y_train_en=le.fit_transform(y_train)
y_test_en=le.transform(y_test)


from sklearn.feature_extraction.text import CountVectorizer


bow=CountVectorizer()
#X_train
X_train_bow=bow.fit_transform(X_train)

# X_test 
X_test_bow=bow.transform(X_test)


print(len(bow.vocabulary_))
print(X_train_bow.shape)
print(bow.get_feature_names_out())
print(X_test_bow.toarray()[:10])


word_count=np.array(X_train_bow.sum(axis=0)).flatten()

word= bow.get_feature_names_out()

freq=pd.DataFrame({"WORD":word,"COUNT":word_count})

freq.sort_values("COUNT",ascending=False)[:20]


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


lr=LogisticRegression(max_iter=1000,random_state=42)

# fitting 
lr.fit(X_train_bow,y_train_en)

# predict on test and train 
y_train_predict_lr=lr.predict(X_train_bow)
y_test_predict_lr= lr.predict(X_test_bow)

#Accuracy 
train_accuracy_lr= accuracy_score(y_train_en,y_train_predict_lr)
test_accuracy_lr =accuracy_score(y_test_en,y_test_predict_lr)

print("Train Accuracy:",train_accuracy_lr)
print("\n")
print("Test Accuracy:",test_accuracy_lr)

print("\n")

#Confusion Matrix 
cm_lr=confusion_matrix(y_test_en,y_test_predict_lr)
print("Confusion_matrix")
print(cm_lr)

print("\n")

#classification Report
cr_lr=classification_report(y_test_en,y_test_predict_lr)
print("Classification_Report")
print(cr_lr)


from xgboost import XGBClassifier


xgb=XGBClassifier(objective='multi:softmax',num_class=len(set(y_train)),n_estimators=200,
                  max_depth=6,learning_rate=0.1,random_state=42)


xgb.fit(X_train_bow,y_train_en)


# predict on test and train 
y_train_predict_xgb=xgb.predict(X_train_bow)
y_test_predict_xgb= xgb.predict(X_test_bow)

#Accuracy 
train_accuracy_xgb= accuracy_score(y_train_en,y_train_predict_xgb)
test_accuracy_xgb =accuracy_score(y_test_en,y_test_predict_xgb)

print("Train Accuracy:",train_accuracy_xgb)
print("\n")
print("Test Accuracy:",test_accuracy_xgb)

print("\n")

#Confusion Matrix 
cm_xgb=confusion_matrix(y_test_en,y_test_predict_xgb)
print("Confusion_matrix")
print(cm_xgb)

print("\n")

#classification Report
cr_xgb=classification_report(y_test_en,y_test_predict_xgb)
print("Classification_Report")
print(cr_xgb)


from sklearn.svm import LinearSVC


svm=LinearSVC(random_state=42)


svm.fit(X_train_bow,y_train_en)


# predict on test and train 
y_train_predict_svm=svm.predict(X_train_bow)
y_test_predict_svm= svm.predict(X_test_bow)

#Accuracy 
train_accuracy_svm= accuracy_score(y_train_en,y_train_predict_svm)
test_accuracy_svm =accuracy_score(y_test_en,y_test_predict_svm)

print("Train Accuracy:",train_accuracy_svm)
print("\n")
print("Test Accuracy:",test_accuracy_svm)

print("\n")

#Confusion Matrix 
cm_svm=confusion_matrix(y_test_en,y_test_predict_svm)
print("Confusion_matrix")
print(cm_svm)

print("\n")

#classification Report
cr_svm=classification_report(y_test_en,y_test_predict_svm)
print("Classification_Report")
print(cr_svm)


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf=TfidfVectorizer()


# Train Vectorizer
X_train_tfidf=tfidf.fit_transform(X_train)

#Test Vectorizer

X_test_tfidf=tfidf.transform(X_test)


lr_tf=LogisticRegression(max_iter=1000,random_state=42)

# fitting 
lr_tf.fit(X_train_tfidf,y_train_en)

# predict on test and train 
y_train_predict_lr_tf=lr_tf.predict(X_train_tfidf)
y_test_predict_lr_tf= lr_tf.predict(X_test_tfidf)

#Accuracy 
train_accuracy_lr_tf= accuracy_score(y_train_en,y_train_predict_lr_tf)
test_accuracy_lr_tf =accuracy_score(y_test_en,y_test_predict_lr_tf)

print("Train Accuracy:",train_accuracy_lr_tf)
print("\n")
print("Test Accuracy:",test_accuracy_lr_tf)

print("\n")

#Confusion Matrix 
cm_lr_tf=confusion_matrix(y_test_en,y_test_predict_lr_tf)
print("Confusion_matrix")
print(cm_lr_tf)

print("\n")

#classification Report
cr_lr_tf=classification_report(y_test_en,y_test_predict_lr_tf)
print("Classification_Report")
print(cr_lr_tf)


xgb_tf=XGBClassifier(objective='multi:softmax',num_class=len(set(y_train)),n_estimators=200,
                  max_depth=6,learning_rate=0.1,random_state=42)


xgb_tf.fit(X_train_tfidf,y_train_en)


# predict on test and train 
y_train_predict_xgb_tf=xgb_tf.predict(X_train_tfidf)
y_test_predict_xgb_tf= xgb_tf.predict(X_test_tfidf)

#Accuracy 
train_accuracy_xgb_tf= accuracy_score(y_train_en,y_train_predict_xgb_tf)
test_accuracy_xgb_tf =accuracy_score(y_test_en,y_test_predict_xgb_tf)

print("Train Accuracy:",train_accuracy_xgb_tf)
print("\n")
print("Test Accuracy:",test_accuracy_xgb_tf)

print("\n")

#Confusion Matrix 
cm_xgb_tf=confusion_matrix(y_test_en,y_test_predict_xgb_tf)
print("Confusion_matrix")
print(cm_xgb_tf)

print("\n")

#classification Report
cr_xgb_tf=classification_report(y_test_en,y_test_predict_xgb_tf)
print("Classification_Report")
print(cr_xgb_tf)


svm_tf=LinearSVC(random_state=42)


svm_tf.fit(X_train_tfidf,y_train_en)


# predict on test and train 
y_train_predict_svm_tf=svm_tf.predict(X_train_tfidf)
y_test_predict_svm_tf= svm_tf.predict(X_test_tfidf)

#Accuracy 
train_accuracy_svm_tf= accuracy_score(y_train_en,y_train_predict_svm_tf)
test_accuracy_svm_tf =accuracy_score(y_test_en,y_test_predict_svm_tf)

print("Train Accuracy:",train_accuracy_svm_tf)
print("\n")
print("Test Accuracy:",test_accuracy_svm_tf)

print("\n")

#Confusion Matrix 
cm_svm_tf=confusion_matrix(y_test_en,y_test_predict_svm_tf)
print("Confusion_matrix")
print(cm_svm_tf)

print("\n")

#classification Report
cr_svm_tf=classification_report(y_test_en,y_test_predict_svm_tf)
print("Classification_Report")
print(cr_svm_tf)


# tokenize train and test before word2vec

tokenized_train=[sentence.split() for sentence in X_train]

tokenized_test=[sentence.split() for sentence in X_test]


from gensim.models import Word2Vec


w2v=Word2Vec(sentences=tokenized_train,vector_size=100,window=5,min_count=2,workers=4)


w2v.wv["happy"]


def document_vector(doc):
    words=doc.split()

    vectors= [w2v.wv[word] for word in words if word in w2v.wv]

    if len(vectors)==0:
        return np.zeros(100)
    return np.mean(vectors,axis=0)


X_train_w2v=np.array([document_vector(doc) for doc in X_train])

X_test_w2v=np.array([document_vector(doc) for doc in X_test])


X_train_w2v.shape


lr_w2v=LogisticRegression(max_iter=1000,random_state=42)

# fitting 
lr_w2v.fit(X_train_w2v,y_train_en)

# predict on test and train 
y_train_predict_lr_w2v=lr_w2v.predict(X_train_w2v)
y_test_predict_lr_w2v= lr_w2v.predict(X_test_w2v)

#Accuracy 
train_accuracy_lr_w2v= accuracy_score(y_train_en,y_train_predict_lr_w2v)
test_accuracy_lr_w2v =accuracy_score(y_test_en,y_test_predict_lr_w2v)

print("Train Accuracy:",train_accuracy_lr_w2v)
print("\n")
print("Test Accuracy:",test_accuracy_lr_w2v)

print("\n")

#Confusion Matrix 
cm_lr_w2v=confusion_matrix(y_test_en,y_test_predict_lr_w2v)
print("Confusion_matrix")
print(cm_lr_w2v)

print("\n")

#classification Report
cr_lr_w2v=classification_report(y_test_en,y_test_predict_lr_w2v)
print("Classification_Report")
print(cr_lr_w2v)


xgb_w2v=XGBClassifier(objective='multi:softmax',num_class=len(set(y_train)),n_estimators=200,
                  max_depth=6,learning_rate=0.1,random_state=42)


xgb_w2v.fit(X_train_w2v,y_train_en)


# predict on test and train 
y_train_predict_xgb_w2v=xgb_w2v.predict(X_train_w2v)
y_test_predict_xgb_w2v= xgb_w2v.predict(X_test_w2v)

#Accuracy 
train_accuracy_xgb_w2v= accuracy_score(y_train_en,y_train_predict_xgb_w2v)
test_accuracy_xgb_w2v =accuracy_score(y_test_en,y_test_predict_xgb_w2v)

print("Train Accuracy:",train_accuracy_xgb_w2v)
print("\n")
print("Test Accuracy:",test_accuracy_xgb_w2v)

print("\n")

#Confusion Matrix 
cm_xgb_w2v=confusion_matrix(y_test_en,y_test_predict_xgb_w2v)
print("Confusion_matrix")
print(cm_xgb_w2v)

print("\n")

#classification Report
cr_xgb_w2v=classification_report(y_test_en,y_test_predict_xgb_w2v)
print("Classification_Report")
print(cr_xgb_w2v)


svm_w2v=LinearSVC(random_state=42)


svm_w2v.fit(X_train_w2v,y_train_en)


# predict on test and train 
y_train_predict_svm_w2v=svm_w2v.predict(X_train_w2v)
y_test_predict_svm_w2v= svm_w2v.predict(X_test_w2v)

#Accuracy 
train_accuracy_svm_w2v= accuracy_score(y_train_en,y_train_predict_svm_w2v)
test_accuracy_svm_w2v =accuracy_score(y_test_en,y_test_predict_svm_w2v)

print("Train Accuracy:",train_accuracy_svm_w2v)
print("\n")
print("Test Accuracy:",test_accuracy_svm_w2v)

print("\n")

#Confusion Matrix 
cm_svm_w2v=confusion_matrix(y_test_en,y_test_predict_svm_w2v)
print("Confusion_matrix")
print(cm_svm_w2v)

print("\n")

#classification Report
cr_svm_w2v=classification_report(y_test_en,y_test_predict_svm_w2v)
print("Classification_Report")
print(cr_svm_w2v)


from sklearn.model_selection import RandomizedSearchCV


xgb_tune=XGBClassifier(objective="multi:softmax",num_class=len(set(y_train_en)),random_state=42)


param_dist={ 'n_estimators':[100,200,300,400,500],
             'max_depth':[3,4,5,6,7,8,9,10,11],
             'learning_rate':[0.01,0.05,0.1,0.2,0.3],
             'subsample':[0.7,0.8,0.9,1.0],
             'colsample_bytree':[0.7,0.8,0.9,1.0],
             'min_child_weight':[1,3,5,7],
             'gamma':[0,0.1,0.2,0.3]
           }


random_search=RandomizedSearchCV(estimator=xgb_tune,param_distributions=param_dist,n_iter=30,
                                 cv=5,scoring='accuracy',verbose=2,random_state=42,n_jobs=-1)


random_search.fit(X_train_tfidf,y_train_en)


print("Best Parameters")
print(random_search.best_params_)
print("="*50)
print("Best Score")
print(random_search.best_score_)


xgb_tf_tuned=XGBClassifier(objective='multi:softmax',num_class=len(set(y_train)),n_estimators=250,
    learning_rate=0.3,
    max_depth=6,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=3,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=5,
    random_state=42)

xgb_tf_tuned.fit(X_train_tfidf,y_train_en)

# predict on test and train 
y_train_predict_xgb_tf_tuned=xgb_tf_tuned.predict(X_train_tfidf)
y_test_predict_xgb_tf_tuned= xgb_tf_tuned.predict(X_test_tfidf)

#Accuracy 
train_accuracy_xgb_tf_tuned= accuracy_score(y_train_en,y_train_predict_xgb_tf_tuned)
test_accuracy_xgb_tf_tuned =accuracy_score(y_test_en,y_test_predict_xgb_tf_tuned)

print("Train Accuracy:",train_accuracy_xgb_tf_tuned)
print("\n")
print("Test Accuracy:",test_accuracy_xgb_tf_tuned)

print("\n")

#Confusion Matrix 
cm_xgb_tf_tuned=confusion_matrix(y_test_en,y_test_predict_xgb_tf_tuned)
print("Confusion_matrix")
print(cm_xgb_tf_tuned)

print("\n")

#classification Report
cr_xgb_tf_tuned=classification_report(y_test_en,y_test_predict_xgb_tf_tuned)
print("Classification_Report")
print(cr_xgb_tf_tuned)


print("\n")
print ("---------------------MODEL COMPARISON TABLE---------------------------")
print("="*70)
print("\n")
result_df=pd.DataFrame([["LOGISTIC REGRESSION(BOW)",98,89],["XGBOOST(BOW)",92,87],["SVM(BOW)",99,88],
                       ["LOGISTIC REGRESSION(TFIDF)",94,87],["XGBOOST(TFIDF)",93,87],["CVM(TFIDF)",98,89],
                       ["LOGISTIC REGRESSION(WORD2VEC)",36,38],["XGBOOST(WORD2VEC)",82,38],["SVM(WORD2VEC)",38,40],
                       ["XGBOOST-TFIDF(TUNED)",94,88]],columns=["MODEL","TRAIN-ACCURACY","TEST-ACCURACY"])

result_df.style.apply(
    lambda x: ['background-color: lightgreen' if x.name == 9 else '' for i in x],
    axis=1
)
