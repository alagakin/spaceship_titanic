import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

SPENDING_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
CAT_FEATURES = ['HomePlanet', 'cabin_deck', 'cabin_side', 'Destination', 'age_group']
PIPELINE_STEPS = [
    'group_features', 'cryo_impute', 'group_mode_impute', 'missing_flags',
    'total_spend', 'has_any_spend', 'age_group', 'split_cabin',
    'group_target_enc', 'cast_types',
]


class GroupFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['p_group'] = X['PassengerId'].str[:4].astype(int)
        X['p_group_id'] = X['PassengerId'].str[5:].astype(int)
        X['g_size'] = X.groupby('p_group')['p_group'].transform('count')
        return X.drop(columns=['PassengerId', 'Name'])


class CryoSpendingImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        all_zero = (X[SPENDING_COLS].sum(axis=1) == 0) & X[SPENDING_COLS].notna().all(axis=1)
        X.loc[X['CryoSleep'].isna() & all_zero, 'CryoSleep'] = True
        X.loc[X['CryoSleep'].isna() & (X[SPENDING_COLS].sum(axis=1) > 0), 'CryoSleep'] = False
        for col in SPENDING_COLS:
            X.loc[X['CryoSleep'].eq(True) & X[col].isna(), col] = 0
        return X


class GroupModeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for col in ['HomePlanet', 'Destination']:
            X[col] = X.groupby('p_group')[col].transform(
                lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
            )
        return X


class MissingFlagAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.missing_cols_ = X.columns[X.isna().any()].tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.missing_cols_:
            X[f'{col}_was_missing'] = X[col].isna().astype(int)
        return X


class TotalSpendAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['total_spend'] = X[SPENDING_COLS].sum(axis=1)
        return X


class HasAnySpendAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['has_any_spend'] = (X['total_spend'] > 0).astype(int)
        return X


class AgeGroupAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['age_group'] = pd.cut(
            X['Age'],
            bins=[0, 17, 64, float('inf')],
            labels=['child', 'adult', 'senior'],
        ).astype(object)
        return X


class CabinSplitter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        parts = X['Cabin'].str.split('/', expand=True)
        X['cabin_deck'] = parts[0]
        X['cabin_num'] = pd.to_numeric(parts[1], errors='coerce')
        X['cabin_side'] = parts[2]
        return X.drop(columns=['Cabin'])


class GroupTargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        agg = X.assign(_target=y.values).groupby('p_group')['_target'].agg(['sum', 'count'])
        self.group_sum_ = agg['sum']
        self.group_count_ = agg['count']
        self.global_mean_ = float(y.mean())
        return self

    def transform(self, X):
        # val / test: group mean from training data
        X = X.copy()
        group_mean = self.group_sum_ / self.group_count_
        X['group_transport_rate'] = X['p_group'].map(group_mean).fillna(self.global_mean_)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        # training: leave-one-out so solo passengers don't see their own label
        self.fit(X, y)
        X = X.copy()
        group_sum = X['p_group'].map(self.group_sum_)
        group_count = X['p_group'].map(self.group_count_)
        loo = (group_sum - y.values) / (group_count - 1)
        X['group_transport_rate'] = loo.where(group_count > 1, self.global_mean_).values
        return X


class TypeCaster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X['CryoSleep'] = X['CryoSleep'].astype(float)
        X['VIP'] = X['VIP'].astype(float)
        for col in CAT_FEATURES:
            X[col] = X[col].fillna('missing')
        return X


def build_pipeline(model_params: dict, *, random_seed: int = 42, verbose: int = 0) -> Pipeline:
    return Pipeline([
        ('group_features',   GroupFeaturesExtractor()),
        ('cryo_impute',      CryoSpendingImputer()),
        ('group_mode_impute', GroupModeImputer()),
        ('missing_flags',    MissingFlagAdder()),
        ('total_spend',      TotalSpendAdder()),
        ('has_any_spend',    HasAnySpendAdder()),
        ('age_group',        AgeGroupAdder()),
        ('split_cabin',      CabinSplitter()),
        ('group_target_enc', GroupTargetEncoder()),
        ('cast_types',       TypeCaster()),
        ('model', CatBoostClassifier(
            **model_params,
            cat_features=CAT_FEATURES,
            random_seed=random_seed,
            verbose=verbose,
        )),
    ])
