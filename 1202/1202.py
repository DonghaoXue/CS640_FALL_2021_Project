#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:10:50 2021

@author: jiachenfeng
"""

import pandas as pd
import numpy as np
import json
import re
import torch
from datasets import Dataset,load_metric
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, default_data_collator, Trainer, set_seed


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
regexMap={r"@[\w]+":"",r"http[\S]+":"",r"#[\w]+":""}
def cleaning(datainput):
    t=datainput
    for regx in regexMap.keys():
        t=re.sub(regx, regexMap[regx], t)
    return t

train_tweets["tweets"]=train_tweets["tweets"].apply(cleaning)
test_tweets["tweets"]=test_tweets["tweets"].apply(cleaning)
## train_tweets.head()

#BERT
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
        
max_length=256
train_tweets['input_ids'] = train_tweets['tweets'].apply(lambda x: tokenizer(x, max_length=max_length, padding="max_length",)['input_ids'])
train_tweets['attention_mask'] = train_tweets['tweets'].apply(lambda x: tokenizer(x, max_length=max_length, padding="max_length",)['attention_mask'])
train_tweets.rename(columns={'>=21?': 'labels'}, inplace=True)
# train_tweets.head()
train = train_tweets[['input_ids', 'attention_mask', 'labels']]

train_df = train[:-256].reset_index(drop=True)
valid_df = train[-256:].reset_index(drop=True)

train_dataset=Dataset.from_pandas(train_df)
valid_dataset=Dataset.from_pandas(valid_df)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# train_dataset = torch.FloatTensor(train_dataset).to(device)
# valid_dataset = torch.FloatTensor(valid_dataset).to(device)

# training
batch_size = 16

args = TrainingArguments(
    output_dir='./results',  
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-5,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)

 
accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # metrics from the datasets library have a `compute` method
    return accuracy_metric.compute(predictions=predictions, references=labels)

data_collator = default_data_collator
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
