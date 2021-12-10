#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:10:50 2021

@author: jiachenfeng
"""

import pandas as pd
import json
import re
import tensorflow as tf
from datasets import Dataset,load_metric
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, default_data_collator, Trainer, set_seed
import emoji
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import ktrain
from ktrain import text


# read data
labeled_data_new = pd.read_csv('labeled_users_new.csv',lineterminator='\n')
labeled_data_old = pd.read_csv('labeled_users_old.csv')
tweets_test = json.load(open('tweets.json',))
tweets_4k = json.load(open('Twitter_User_Handles_labeled_tweets.json',))
demo = json.load(open('User demo profiles.json',))

# data preprocess
age_train=labeled_data_old.dropna(subset=['year_born'])
age_train['>=21?']=age_train.apply(lambda x:1 if x.year_born<=2000 else 0,axis=1)
age_test=labeled_data_new.dropna(subset=['human.labeled.age'])
age_test['>=21?']=labeled_data_new.apply(lambda x:1 if x['human.labeled.age']>=21 else 0,axis=1)


# concatenated string
def str_c(str_list):
    res=''
    for s in str_list:
        res+=' '+s
    return res

tweet_df=pd.DataFrame(columns=["number","tweets"])
test_tweet_df=pd.DataFrame(columns=["username","tweets"])
for n,tweet_list in tweets_4k.items():
    tweet_df=tweet_df.append({"number":int(n),"tweets":str_c(tweet_list)},ignore_index=True)
for n,tweet_list in tweets_test.items():
    test_tweet_df=test_tweet_df.append({"username":n,"tweets":str_c(tweet_list)},ignore_index=True)
    
# merge Dataframe
train_tweets=pd.merge(age_train,tweet_df,how='inner',left_on='user_id',right_on='number')
test_tweets=pd.merge(age_test,test_tweet_df,how='inner',left_on='screen_name',right_on='username')

# cleaning function
regexMap={r"@[\w]+":"",r"http[\S]+":"",r"#[\w]+":"",r" {2,}":" ",
          r"&amp;?":"and",r"&lt;":"<",r"&gt;":">",r"\n":" ",":":" "}
def cleaning(datainput):
    t=datainput
    for regx in regexMap.keys():
        t=re.sub(regx, regexMap[regx], t)
    return t

# convert emoji to text
for string,i in zip(train_tweets["tweets"],range(len(train_tweets["tweets"]))):
    train_tweets["tweets"][i]=emoji.demojize(string)
for string,i in zip(test_tweets["tweets"],range(len(test_tweets["tweets"]))):
    test_tweets["tweets"][i]=emoji.demojize(string)
    
train_tweets["tweets"]=train_tweets["tweets"].apply(cleaning)
test_tweets["tweets"]=test_tweets["tweets"].apply(cleaning)
## train_tweets.head()

# NB
## holdout validation
X_train, X_test, y_train, y_test = train_test_split(test_tweets["tweets"], test_tweets[">=21?"],
                                                    test_size = 0.2, random_state = 0)

vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(X_train.values)
vectorizer.vocabulary_

train_input=vectorizer.transform(X_train.values)
test_input=vectorizer.transform(X_test.values)

nb=MultinomialNB()
nb.fit(train_input,y_train)

pred_class=nb.predict(test_input)
metrics.accuracy_score(y_test.values,pred_class)
print(classification_report(y_true=y_test.values,y_pred=pred_class))
print(accuracy_score(y_test, pred_class))
print(confusion_matrix(y_test, pred_class))

## 5-fold CV
num_folds=5
subset_size=int(len(test_tweets)/num_folds)
accuracy_list=[]
test_tweets=shuffle(test_tweets)
for i in range(num_folds):
    testing_this_round = test_tweets[i*subset_size:][:subset_size]
    training_this_round = pd.concat([test_tweets[:i*subset_size],test_tweets[(i+1)*subset_size:]])
    X_train, X_test, y_train, y_test =training_this_round["tweets"],testing_this_round["tweets"],training_this_round[">=21?"],testing_this_round[">=21?"]
    vectorizer.fit(X_train.values)
    vectorizer.vocabulary_

    train_input=vectorizer.transform(X_train.values)
    test_input=vectorizer.transform(X_test.values)

    nb=MultinomialNB()
    nb.fit(train_input,y_train)

    pred_class=nb.predict(test_input)
    metrics.accuracy_score(y_test.values,pred_class)
    accuracy_list.append(accuracy_score(y_test, pred_class))

print(accuracy_list)
from statistics import mean
mean_accuracy=mean(accuracy_list)
print('The accuracy for 5-fold CV is : {:.4f}'.format(mean_accuracy))

# Transformers
X_train, X_test, y_train, y_test = train_test_split(test_tweets["tweets"], test_tweets[">=21?"],
                                                    test_size = 0.2, random_state = 0)
xtrain,xtest,ytrain,ytest=[],[],[],[]
for i,j in zip(X_train,y_train):
    xtrain.append(i)
    ytrain.append(j)
for i,j in zip(X_test,y_test):
    xtest.append(i)
    ytest.append(j)

# Create transformer model
MODEL_NAME = 'distilbert-base-uncased'
Categories=['age>=21','age<21']
t = text.Transformer(MODEL_NAME, maxlen=512, classes=Categories)
trn = t.preprocess_train(xtrain, ytrain)
val = t.preprocess_test(xtest, ytest)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
    
# Train the model
learner.fit_onecycle(5e-5, 4) # define learning rate and epochs

# Evaluate and Inspect the Model
learner.validate(class_names=t.get_classes())

# majority vote classification
mv_age=age_test[["screen_name",">=21?"]]
mv_df=pd.DataFrame(columns=["username","tweets"])
for n,tweet_list in tweets_test.items():
    for tweets in tweet_list:
        mv_df=mv_df.append({"username":n,"tweets":tweets},ignore_index=True)

mv_all=pd.merge(mv_age,mv_df,how='left',left_on='screen_name',right_on='username')

# cleaning
for string,i in zip(mv_all["tweets"],range(len(mv_all["tweets"]))):
    mv_all["tweets"][i]=emoji.demojize(string)
mv_all["tweets"]=mv_all["tweets"].apply(cleaning)
mv_all=mv_all.drop(mv_all[mv_all["tweets"].map(len)<=8].index)
# NB
X_train, X_test, y_train, y_test, username_train, username_test = train_test_split(
    mv_all["tweets"], mv_all[">=21?"], mv_all["username"],test_size = 0.2, random_state = 0)

vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(X_train.values)
vectorizer.vocabulary_

train_input=vectorizer.transform(X_train.values)
test_input=vectorizer.transform(X_test.values)

nb=MultinomialNB()
nb.fit(train_input,y_train)

pred_class=nb.predict(test_input)
pred_result_df=pd.concat([y_test, username_test], axis=1)
pred_class_df=pd.DataFrame(pred_class,columns=['pred_class'])
pred_result_df=pred_result_df.reset_index(drop=True)
pred_result=pd.concat([pred_result_df,pred_class_df], axis=1)
unique_username=pred_result["username"].unique()
mv_result=pd.DataFrame(columns=["username","real_class","pred_class"])
for users in unique_username:
    user_unique=pred_result[pred_result["username"]==users]
    pred_class_users=0 if len(user_unique[user_unique["pred_class"]==0])>len(user_unique[user_unique["pred_class"]==1]) else 1  
    real_class=0 if sum(user_unique[">=21?"])==0 else 1
    mv_result=mv_result.append({"username":users,"real_class":real_class,"pred_class":pred_class_users},ignore_index=True)
accuracy_score_mv=len(mv_result[mv_result["real_class"]==mv_result["pred_class"]])  / len(mv_result)  
print('The accuracy majority vote is : {:.4f}'.format(accuracy_score_mv))
mv_result['real_class'] = mv_result['real_class'].astype(int)
mv_result['pred_class'] = mv_result['pred_class'].astype(int)
real_class=mv_result["real_class"].to_numpy()
pred_class=mv_result["pred_class"].to_numpy()
print(classification_report(y_true=real_class,y_pred=pred_class))

#NB for race
train1_tweets = train_tweets.dropna(subset=['race'])
train1_tweets = train1_tweets[~train1_tweets['race'].isin([5])]
X_train, X_test, y_train, y_test = train_test_split(train1_tweets["tweets"], train1_tweets["race"],
                                                    test_size = 0.2, random_state = 0)

vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(X_train.values)
vectorizer.vocabulary_

train_input=vectorizer.transform(X_train.values)
test_input=vectorizer.transform(X_test.values)

nb=MultinomialNB()
nb.fit(train_input,y_train)

pred_class=nb.predict(test_input)
metrics.accuracy_score(y_test.values,pred_class)
print(classification_report(y_true=y_test.values,y_pred=pred_class))
print('accuracy score:', accuracy_score(y_test, pred_class))


#Logistic Regression for race with 5-fold CV
X_train, X_test, y_train, y_test = train_test_split(train1_tweets["tweets"], train1_tweets["race"],
                                                    test_size = 0.2, random_state = 0)

vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(X_train.values)
vectorizer.vocabulary_

train_input=vectorizer.transform(X_train.values)
test_input=vectorizer.transform(X_test.values)

lm=LogisticRegressionCV(cv = 5 ,class_weight = 'balanced', max_iter = 200)
lm.fit(train_input,y_train)
pred_class=lm.predict(test_input)
metrics.accuracy_score(y_test.values,pred_class)
print(classification_report(y_true=y_test.values,y_pred=pred_class))
print('accuracy score:', accuracy_score(y_test, pred_class))
print(confusion_matrix(y_test, pred_class))


#BERT
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# model = BertForSequenceClassification.from_pretrained('bert-base-cased')
        
# max_length=512
# train_tweets['input_ids'] = train_tweets['tweets'].apply(lambda x: tokenizer(x, max_length=max_length, padding="max_length",)['input_ids'])
# train_tweets['attention_mask'] = train_tweets['tweets'].apply(lambda x: tokenizer(x, max_length=max_length, padding="max_length",)['attention_mask'])
# train_tweets.rename(columns={'>=21?': 'labels'}, inplace=True)
# # train_tweets.head()
# train = train_tweets[['input_ids', 'attention_mask', 'labels']]

# train_df = train[:-512].reset_index(drop=True)
# valid_df = train[-512:].reset_index(drop=True)

# train_ds=Dataset.from_pandas(train_df)
# valid_ds=Dataset.from_pandas(valid_df)

# # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# # train_dataset = torch.FloatTensor(train_dataset).to(device)
# # valid_dataset = torch.FloatTensor(valid_dataset).to(device)

# # training
# batch_size = 16

# args = TrainingArguments(
#     output_dir='./results',  
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=3e-5,
#     gradient_accumulation_steps=8,
#     num_train_epochs=3,
#     warmup_ratio=0.1,
#     weight_decay=0.01,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
# )

 
# accuracy_metric = load_metric("accuracy")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     # metrics from the datasets library have a `compute` method
#     return accuracy_metric.compute(predictions=predictions, references=labels)

# data_collator = default_data_collator
# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_ds,
#     eval_dataset=valid_ds,
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer.train()
