import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import  svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def load_data(data_name, fp_name):
    data = pd.read_csv(data_name + '.csv', encoding='gbk')
    fingerprint = pd.read_csv(fp_name + '.csv', encoding='gbk', low_memory=False)

    # Extract features and target
    entry = list(range(len(data)))
    smiles = data['smiles'].values.tolist()
    y = data['energy_barrier'].values

    # Remove all 0/1 columns from fingerprint
    #fingerprint = fingerprint.loc[:, ~(fingerprint.isin([0, 1])).any()]

    X = fingerprint.values
    X = np.nan_to_num(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X, y, entry, smiles


def random_forest(m):
    title = r'Random Forest'
    print(title)
    # {'absolute_error', 'friedman_mse', 'squared_error', 'poisson'}
    rf = RandomForestRegressor(n_estimators=500, criterion='squared_error', oob_score=True,
                               bootstrap=True, n_jobs=-1, verbose=1)
    regressor_method(title, rf, m)


def svr(m):
    title = r'SVR'
    print(title)
    svr = svm.SVR(kernel='rbf', C=1e4, gamma=0.01)
    regressor_method(title, svr, m)


def xgboost_reg(m):
    title = r'XGBoost'
    print(title)
    xgb_reg = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    regressor_method(title, xgb_reg, m)


def regressor_method(title, regressor, m):
    global X_train, X_test, y_train, y_test, X, y, entry, smiles
    regressor.fit(X_train, y_train)
    score = round(regressor.score(X_test, y_test), 3)
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    rms_train = (np.mean((y_train - y_pred_train)**2))**0.5
    rms_test = (np.mean((y_test - y_pred_test)**2))**0.5
    print("RMS_train", rms_train)
    print("r^2 score_train", r2_score(y_train, y_pred_train))
    print("RMS_test", rms_test)
    print("r^2 score_test", r2_score(y_test, y_pred_test))
    plt.subplot(m)
    plt.scatter(y_train, y_pred_train, label='Train', c='blue', alpha=0.5)
    score = r' (Score = ' + str(score) + r')'
    plt.title(title + score)
    plt.ylabel('Predicted Y')
    x_start = min(y)-(max(y)-min(y))*0.1
    x_end = min(y)+(max(y)-min(y))*1.1
    plt.xlim((x_start, x_end))
    plt.ylim((x_start, x_end))
    plt.scatter(y_test, y_pred_test, c='lightgreen', label='Test', alpha=0.5)
    x1 = [x_start, x_end]
    y1 = x1
    plt.plot(x1, y1, c='lightcoral', alpha=0.8)
    plt.legend(loc=4)
    test5 = 'RMS_train = '+str(round(rms_train, 3))
    test6 = 'r^2 score_train = '+str(round(r2_score(y_train, y_pred_train), 3))
    point_x1 = min(y)-(max(y)-min(y))*0.05
    point_y1 = (max(y)-min(y))*1+min(y)
    point_y2 = (max(y)-min(y))*0.89+min(y)
    plt.text(point_x1, point_y1, test5, weight="light", bbox=dict(facecolor="blue", alpha=0.2))
    plt.text(point_x1, point_y2, test6, weight="light", bbox=dict(facecolor="blue", alpha=0.2))
    test1 = 'RMS_test = '+str(round(rms_test, 3))
    test2 = 'r^2 score_test = '+str(round(r2_score(y_test, y_pred_test), 3))
    point_y3 = (max(y)-min(y))*0.76+min(y)
    point_y4 = (max(y)-min(y))*0.65+min(y)
    plt.text(point_x1, point_y3, test1, weight="light", bbox=dict(facecolor="lightgreen", alpha=0.2))
    plt.text(point_x1, point_y4, test2, weight="light", bbox=dict(facecolor="lightgreen", alpha=0.2))
    plt.tight_layout()


if __name__ == '__main__':
    # file name
    data_name = '5__cleaned_data'
    fp_name = ['fp_spoc__5__cleaned_data',]
    for fp in fp_name:
        X_train, X_test, y_train, y_test, X, y, entry, smiles = load_data(
            data_name, fp)
        plt.figure(figsize=(20, 6))
        random_forest(131)
        xgboost_reg(132)
        svr(133)
        plt.savefig('scikit_'+fp+'.png', dpi=300)

