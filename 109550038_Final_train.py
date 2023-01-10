import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

TRAIN_PATH = "./data/train.csv"

folds_dict = {'Fold 1': [['C', 'D', 'E'], ['A', 'B']],
              'Fold 2': [['B', 'D', 'E'], ['A', 'C']],
              'Fold 3': [['B', 'C', 'E'], ['A', 'D']],
              'Fold 4': [['B', 'C', 'D'], ['A', 'E']],
              'Fold 5': [['A', 'D', 'E'], ['B', 'C']],
              'Fold 6': [['A', 'C', 'E'], ['B', 'D']],
              'Fold 7': [['A', 'C', 'D'], ['B', 'E']],
              'Fold 8': [['A', 'B', 'E'], ['C', 'D']],
              'Fold 9': [['A', 'B', 'D'], ['C', 'E']],
              'Fold 10': [['A', 'B', 'C'], ['D', 'E']]}

features = ['loading', 'attribute_0', 'measurement_17',
            'measurement_0', 'measurement_1', 'measurement_2']

train_df = pd.read_csv(TRAIN_PATH)

feature = ['loading', 'measurement_17',
           'measurement_0', 'measurement_1', 'measurement_2']

for code in train_df.product_code.unique():
    imp = SimpleImputer(missing_values=pd.NA, strategy='mean')
    train_df.loc[train_df.product_code == code, feature] = imp.fit_transform(
        train_df.loc[train_df.product_code == code, feature])

woe = ce.WOEEncoder(cols=['attribute_0'])
woe.fit(train_df, train_df['failure'])
train_df = woe.transform(train_df)

with open('woe.pkl', 'wb') as outp:
    pickle.dump(woe, outp, pickle.HIGHEST_PROTOCOL)

auc_score = []
for i, fold in enumerate(folds_dict.keys()):
    x_train, y_train = train_df[train_df['product_code'].isin(
        folds_dict[fold][0])][features].values, train_df[train_df['product_code'].isin(folds_dict[fold][0])]['failure'].values
    x_valid, y_valid = train_df[train_df['product_code'].isin(
        folds_dict[fold][1])][features].values, train_df[train_df['product_code'].isin(folds_dict[fold][1])]['failure'].values

    model = LogisticRegression(
        max_iter=500, C=0.1, dual=False, penalty="l2", solver='newton-cg')
    model.fit(x_train, y_train)

    pred = model.predict_proba(x_valid)[:, 1].reshape(-1, 1)

    score = roc_auc_score(y_valid, pred)
    auc_score.append(score)

    joblib.dump(model, f'model_{i}.pkl')

    print(f'AUC score {fold} : {score}')

print(f'avg AUC score: {np.mean(np.array(auc_score))}')
