from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb


class RegressionModels:
    def __init__(self, features, target, feature_list, random_state = 42, test_size = 0.2):
        self.features = features
        self.target = target
        self.feature_list = feature_list
        self.random_state = random_state
        self.test_size = test_size

    def standard_scaler(self, features):
        scaled_features = StandardScaler().fit_transform(self.features)
        return pd.DataFrame(scaled_features, columns = self.features.columns)

    def spliting(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size = self.test_size, random_state = self.random_state)
        print(f'X_train shape : {X_train.shape}')
        print(f'X_test shape : {X_test.shape}')
        print(f'y_train shape : {y_train.shape}')
        print(f'y_test shape : {y_test.shape}')
        return X_train, X_test, y_train, y_test

    def linear_regressor(self, X_train, X_test, y_train, y_test):
        model = LinearRegression().fit(X_train[self.feature_list], y_train)
        pred_linear = model.predict(X_test[self.feature_list])
        mae = mean_absolute_error(pred_linear, y_test)
        mse = mean_squared_error(pred_linear, y_test)
        r2 = r2_score(pred_linear, y_test)
        print(f'LinearRegression {self.feature_list} MAE : {mae:.4f}')
        print(f'LinearRegression {self.feature_list} MSE : {mse:.4f}')
        print(f'LinearRegression {self.feature_list} R2 : {r2:.4f}')
        return pred_linear, [np.round(mae, 4), np.round(mse, 4), np.round(r2, 4)]

    def random_forest_regressor(self, X_train, X_test, y_train, y_test, n_estimators = 100):
        model = RandomForestRegressor(n_estimators = n_estimators).fit(X_train[self.feature_list], y_train)
        pred_rf = model.predict(X_test[self.feature_list])
        mae = mean_absolute_error(pred_rf, y_test)
        mse = mean_squared_error(pred_rf, y_test)
        r2 = r2_score(pred_rf, y_test)
        print(f'LinearRegression {self.feature_list} MAE : {mae:.4f}')
        print(f'LinearRegression {self.feature_list} MSE : {mse:.4f}')
        print(f'LinearRegression {self.feature_list} R2 : {r2:.4f}')
        return pred_rf, [np.round(mae, 4), np.round(mse, 4), np.round(r2, 4)]

    # def lightGBM_regressor(self, X_train, X_test, y_train, y_test, n_estimators):
    #     params = {
    #     'task': 'train', 
    #     'boosting': 'gbdt',
    #     'objective': 'regression',
    #     'num_leaves': 10,
    #     'learnnig_rage': 0.05,
    #     'verbose': -1
    #     }
        
    #     lgb_train = lgb.Dataset(X_train, y_train)
    #     lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        
    #     model = lgb.train(params,
    #              train_set=lgb_train,
    #              valid_sets=lgb_eval)
        
    #     pred_lgbm = model.predict(X_test)
    #     mae = mean_absolute_error(pred_lgbm, y_test)
    #     mse = mean_squared_error(pred_lgbm, y_test)
    #     r2 = r2_score(pred_lgbm, y_test)
    #     print(f'lightGBMregressor {self.feature_list} MAE : {mae:.4f}')
    #     print(f'lightGBMregressor {self.feature_list} MSE : {mse:.4f}')
    #     print(f'lightGBMregressor {self.feature_list} R2 : {r2:.4f}')
    #     return pred_lgbm, [np.round(mae, 4), np.round(mse, 4), np.round(r2, 4)]

    def lightGBM_regressor(self, X_train, X_test, y_train, y_test, num_boost_round = 1000, **kwargs):
        print(f'your kwargs are : {kwargs}')
        lgb_train = lgb.Dataset(X_train, label = y_train)
        lgb_test = lgb.Dataset(X_test, label = y_test)
        model_lgbm = lgb.train(kwargs, lgb_train, num_boost_round, lgb_test, verbose_eval = 100, early_stopping_rounds = 100)
        pred_lgbm = model_lgbm.predict(X_test)
        mae = mean_absolute_error(pred_lgbm, y_test)
        mse = mean_squared_error(pred_lgbm, y_test)
        r2 = r2_score(pred_lgbm, y_test)
        print(f'lightGBMregressor {self.feature_list} MAE : {mae:.4f}')
        print(f'lightGBMregressor {self.feature_list} MSE : {mse:.4f}')
        print(f'lightGBMregressor {self.feature_list} R2 : {r2:.4f}')
        return pred_lgbm, [np.round(mae, 4), np.round(mse, 4), np.round(r2, 4)]

    def ploting(self, data, x_axis, ): # ???????????? ???????????? ???????????? ?????? ?????????
        pass


# e.g.)
# feature, target = origin_csv.iloc[:, :-1], origin_csv.iloc[:, -1]
# rm = RegressionModels(feature, target, list(feature.columns))                                     # class??? rm?????? ??????????????? ?????? feature, target, ????????? fit??? feature??? ??????
# scaled_feature_df = rm.standard_scaler(feature)                                                   # StandardScaler??? scaling
# X_train, X_test, y_train, y_test = rm.spliting(scaled_feature_df, target)                         # train_test_split??? ??????, ??????????????? X_train, X_test, y_train, y_test??? shape??? ??????
# rf_pred_value = rm.random_forest_regressor(X_train, X_test, y_train, y_test, n_estimators = 10)   # ????????? ?????? X_train, X_test, y_train, y_test??? RandomForestRegressor??? ??????
                                                                                                    # ??????????????? RF??? ????????? ????????? MAE, MSE, R2??? ????????? ???????????? ??????

# e.g. 2) RegressionModels.lightGBM_regressor ??????
# feature, target = origin_csv.iloc[:, :-1], origin_csv.iloc[:, -1]
# rm = RegressionModels(feature, target, list(feature.columns))
# scaled_feature_df = rm.standard_scaler(feature)
# X_train, X_test, y_train, y_test = rm.spliting(scaled_feature_df, target)
# pred_lgbm, error_lgbm = rm.lightGBM_regressor(X_train, X_test, y_train, y_test,                                                  # train??? test data??? ??????????????? ???????????? ??????
#                                               task = 'train', objective = 'regressor', boosting = 'gbdt', learning_rate = 0.01,  # ??????????????? **kwargs??? ???????????? options
#                                               num_leaves = 10, num_threads = 4, metric = ['mse', 'mae'], seed = 42,
#                                               max_depth = 16, is_training_metric = True, feature_fraction = 0.9,
#                                               bagging_fraction = 0.7, bagging_freg = 5)