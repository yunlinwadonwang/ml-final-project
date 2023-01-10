import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
import joblib

TEST_PATH = "./data/test.csv"

features = ['loading', 'attribute_0', 'measurement_17',
            'measurement_0', 'measurement_1', 'measurement_2']

test_df = pd.read_csv(TEST_PATH)

feature = ['loading', 'measurement_17',
           'measurement_0', 'measurement_1', 'measurement_2']

for code in test_df.product_code.unique():
    imp = SimpleImputer(missing_values=pd.NA, strategy='mean')
    test_df.loc[test_df.product_code == code, feature] = imp.fit_transform(
        test_df.loc[test_df.product_code == code, feature])

with open('woe.pkl', 'rb') as inp:
    woe = pickle.load(inp)

test_df['tmp'] = test_df['loading']
test_df = woe.transform(test_df)

test_predictions = np.zeros((test_df.shape[0], 1))

for i in range(10):
    x_test = test_df[features].values
    model = joblib.load(f'model_{i}.pkl')
    pred = model.predict_proba(x_test)[:, 1].reshape(-1, 1)
    test_predictions += pred/10

submission = pd.read_csv(f'{TEST_PATH}/../sample_submission.csv')
submission['failure'] = test_predictions
submission['failure'] = submission['failure'].rank(pct=True).values
submission.to_csv('109550038.csv', index=False)
