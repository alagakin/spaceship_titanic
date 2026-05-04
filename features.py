import pandas as pd
from typing import Callable

SPENDING_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
CAT_FEATURES = ['HomePlanet', 'cabin_deck', 'cabin_side', 'Destination', 'age_group']

TransformFn = Callable[[pd.DataFrame], pd.DataFrame]


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    df['p_group'] = df['PassengerId'].str[:4].astype(int)
    df['p_group_id'] = df['PassengerId'].str[5:].astype(int)
    df['g_size'] = df.groupby('p_group')['p_group'].transform('count')
    return df.drop(columns=['PassengerId', 'Name'])


def impute_cryo_spending(df: pd.DataFrame) -> pd.DataFrame:
    all_zero = (df[SPENDING_COLS].sum(axis=1) == 0) & df[SPENDING_COLS].notna().all(axis=1)
    df.loc[df['CryoSleep'].isna() & all_zero, 'CryoSleep'] = True
    df.loc[df['CryoSleep'].isna() & (df[SPENDING_COLS].sum(axis=1) > 0), 'CryoSleep'] = False
    for col in SPENDING_COLS:
        df.loc[df['CryoSleep'].eq(True) & df[col].isna(), col] = 0
    return df


def impute_group_mode(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['HomePlanet', 'Destination']:
        df[col] = df.groupby('p_group')[col].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )
    return df


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = df.columns[df.isna().any()].tolist()
    for col in missing_cols:
        df[f'{col}_was_missing'] = df[col].isna().astype(int)
    return df


def add_total_spend(df: pd.DataFrame) -> pd.DataFrame:
    df['total_spend'] = df[SPENDING_COLS].sum(axis=1)
    return df


def add_has_any_spend(df: pd.DataFrame) -> pd.DataFrame:
    df['has_any_spend'] = (df['total_spend'] > 0).astype(int)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df['age_group'] = pd.cut(
        df['Age'],
        bins=[0, 17, 64, float('inf')],
        labels=['child', 'adult', 'senior'],
    ).astype(object)
    return df


def split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    parts = df['Cabin'].str.split('/', expand=True)
    df['cabin_deck'] = parts[0]
    df['cabin_num'] = pd.to_numeric(parts[1], errors='coerce')
    df['cabin_side'] = parts[2]
    return df.drop(columns=['Cabin'])


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df['CryoSleep'] = df['CryoSleep'].astype(float)
    df['VIP'] = df['VIP'].astype(float)
    for col in CAT_FEATURES:
        df[col] = df[col].fillna('missing')
    return df


def build_pipeline() -> list[TransformFn]:
    return [
        add_group_features,
        impute_cryo_spending,
        impute_group_mode,
        add_missing_flags,
        add_total_spend,
        add_has_any_spend,
        add_age_group,
        split_cabin,
        cast_types,
    ]


def preprocess(df: pd.DataFrame, is_train: bool = True):
    df = df.copy()

    passenger_ids = df['PassengerId'].copy()

    if is_train:
        y = df['Transported'].astype(int)
        df = df.drop(columns=['Transported'])
    else:
        y = passenger_ids

    for step in build_pipeline():
        df = step(df)

    return df, y
