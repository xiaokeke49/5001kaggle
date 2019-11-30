import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import tree
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split


train_data = pd.read_csv("data/train.csv", parse_dates=['purchase_date', 'release_date'])
test_data = pd.read_csv("data/test.csv", parse_dates=['purchase_date', 'release_date'])

# ### Processing missing data
train_data['purchase_date'] = train_data.purchase_date.fillna(method='backfill')
train_data['total_positive_reviews'] = train_data.total_positive_reviews.fillna(method='backfill')
train_data['total_negative_reviews'] = train_data.total_negative_reviews.fillna(method='backfill')

test_data['purchase_date'] = test_data.purchase_date.fillna(method='backfill')
test_data['total_positive_reviews'] = test_data.total_positive_reviews.fillna(method='backfill')
test_data['total_negative_reviews'] = test_data.total_negative_reviews.fillna(method='backfill')


# date Functin
def process_date(df_copy, df, column):
    df_copy[column + '_year'] = df[column].apply(lambda x: x.year)
    df_copy[column + '_month'] = df[column].apply(lambda x: x.month)
    df_copy[column + '_day'] = df[column].apply(lambda x: x.day)
    return df_copy


# processing purchase_date
train_copy = train_data.copy()
train_extract_purchase_date = process_date(train_copy, train_data, 'purchase_date')

# processing release_date
train_extract_purchase_date_copy = train_extract_purchase_date.copy()
train_process_date = process_date(train_extract_purchase_date_copy, train_extract_purchase_date, 'release_date')

train_process_date['date_interval'] = (train_process_date['purchase_date'] - train_process_date['release_date']).apply(
    lambda x: x.days)

# processing purchase_date
train_copy = test_data.copy()
test_purchase_date = process_date(train_copy, test_data, 'purchase_date')

# processing release_date
test_purchase_date_copy = test_purchase_date.copy()
test_process_date = process_date(test_purchase_date_copy, test_purchase_date, 'release_date')
test_process_date['date_interval'] = (test_process_date['purchase_date'] - test_process_date['release_date']).apply(
    lambda x: x.days)

# processing categories data
train_categories_one_hot = train_data["categories"].str.get_dummies(",")
test_categories_one_hot = test_data["categories"].str.get_dummies(",")
categories_train_diff_test = train_categories_one_hot.columns.difference(test_categories_one_hot.columns)
categories_test_diff_train = test_categories_one_hot.columns.difference(train_categories_one_hot.columns)

test_categories_one_hot = pd.concat([test_categories_one_hot, pd.DataFrame(columns=list(categories_train_diff_test))],
                                    axis=1).fillna(0)

# processing genres
train_genres_one_hot = train_data["genres"].str.get_dummies(",")
test_genres_one_hot = test_data["genres"].str.get_dummies(",")
genres_train_diff_test = train_genres_one_hot.columns.difference(test_genres_one_hot.columns)
genres_test_diff_train = test_genres_one_hot.columns.difference(train_genres_one_hot.columns)

test_genres_one_hot = pd.concat([test_genres_one_hot, pd.DataFrame(columns=list(genres_train_diff_test))],
                                axis=1).fillna(0)

# processing tags
train_tags_one_hot = train_data["tags"].str.get_dummies(",")
test_tags_one_hot = test_data["tags"].str.get_dummies(",")
tags_train_diff_test = train_tags_one_hot.columns.difference(test_tags_one_hot.columns)
tags_test_diff_train = test_tags_one_hot.columns.difference(train_tags_one_hot.columns)

test_tags_one_hot = pd.concat([test_tags_one_hot, pd.DataFrame(columns=list(tags_train_diff_test))], axis=1).fillna(0)
train_tags_one_hot = pd.concat([train_tags_one_hot, pd.DataFrame(columns=list(tags_test_diff_train))], axis=1).fillna(0)

# ### Add features to data
train = pd.concat([train_process_date, train_categories_one_hot, train_genres_one_hot, train_tags_one_hot], axis=1)
test = pd.concat([test_process_date, test_categories_one_hot, test_genres_one_hot, test_tags_one_hot], axis=1)

# Drop no need data
train_input = train.drop(['categories', 'genres', 'tags', 'purchase_date', 'release_date'], axis=1)
test_input = test.drop(['categories', 'genres', 'tags', 'purchase_date', 'release_date'], axis=1)

# Drop label
train_x = train_input.drop(['playtime_forever'], axis=1)
train_y = train_input[['playtime_forever']]

# Do PCA to data
pca = PCA(n_components=50)
train_x_pca = pca.fit_transform(train_x)
test_x_pca = pca.fit_transform(test_input)
print('train_x_pca shape', train_x_pca.shape)
print('test_x_pca shape', test_x_pca.shape)

# Decision Tree
model_dt_1 = tree.DecisionTreeRegressor()
model_dt_1.fit(train_x_pca, train_y)
predictions_dt_1 = model_dt_1.predict(test_x_pca)
predictions_dt_df_1 = pd.DataFrame(predictions_dt_1)
predictions_dt_df_1.columns = ['playtime_forever']
result = pd.DataFrame({'id': test_data['id'], 'playtime_forever': predictions_dt_df_1['playtime_forever']})

from sklearn.model_selection import cross_val_score
scores_dt_1 = np.sqrt(-cross_val_score(model_dt_1, train_x_pca, train_y, cv=80,scoring='neg_mean_squared_error'))
mean_scores_dt_1 = np.mean(scores_dt_1)
print(scores_dt_1)
print('mean_scores_dt_1:',mean_scores_dt_1)
result.to_csv('test.csv', index=0)