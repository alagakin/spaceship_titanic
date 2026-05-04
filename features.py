import pandas as pd

SPENDING_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
CAT_FEATURES = ['HomePlanet', 'Cabin', 'Destination']


def preprocess(df: pd.DataFrame, is_train: bool = True):
    """
    Returns (X, y) for train, (X, passenger_ids) for test.
    """
    df = df.copy()

    df['p_group'] = df['PassengerId'].str[:4].astype(int)
    df['p_group_id'] = df['PassengerId'].str[5:].astype(int)
    df['g_size'] = df.groupby('p_group')['p_group'].transform('count')

    passenger_ids = df['PassengerId'].copy()
    df = df.drop(columns=['PassengerId', 'Name'])

    if is_train:
        y = df['Transported'].astype(int)
        df = df.drop(columns=['Transported'])
    else:
        y = passenger_ids

    # CryoSleep <-> spending mutual imputation
    all_zero = (df[SPENDING_COLS].sum(axis=1) == 0) & df[SPENDING_COLS].notna().all(axis=1)
    df.loc[df['CryoSleep'].isna() & all_zero, 'CryoSleep'] = True
    df.loc[df['CryoSleep'].isna() & (df[SPENDING_COLS].sum(axis=1) > 0), 'CryoSleep'] = False
    for col in SPENDING_COLS:
        df.loc[(df['CryoSleep'] == True) & df[col].isna(), col] = 0

    # HomePlanet / Destination — mode within travel group
    for col in ['HomePlanet', 'Destination']:
        df[col] = df.groupby('p_group')[col].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )

    # Flag columns that still have missing values
    for col in df.columns[df.isna().any()]:
        df[f'{col}_was_missing'] = df[col].isna().astype(int)

    df['CryoSleep'] = df['CryoSleep'].astype(float)
    df['VIP'] = df['VIP'].astype(float)
    for col in CAT_FEATURES:
        df[col] = df[col].fillna('missing')

    return df, y
