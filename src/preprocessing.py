import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def create_preprocessing_pipeline(meta):
    list_of_columntransformers = []

    # if there is numeric features, use standard scaler
    if meta['numeric_features'] !=None:
        #ensure there are no duplicates
        numerical_columns = list(set(meta['numeric_features']))
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        list_of_columntransformers.append( ('num', numerical_transformer, numerical_columns) )

    # if there is categorical features, use ohe
    if meta['categorical_features'] !=None:
        #ensure there are no duplicates
        categorical_columns = list(set(meta['categorical_features']))
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        list_of_columntransformers.append( ('cat', categorical_transformer, categorical_columns) )

    # if there is ordinal features, use ordinal encoder
    if meta['ordinal_features'] !=None:
        for col in meta['ordinal_features'].keys():
            ordinal_transformer = Pipeline([
                (f'ordinal_{col}', OrdinalEncoder(categories=[meta['ordinal_features'][col]], handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            list_of_columntransformers.append( (f'ord_{col}', ordinal_transformer, [col] ) )

    preprocessor = ColumnTransformer(list_of_columntransformers)
    return preprocessor

def compute_zscore(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df

def one_hot_encode(df, columns):
    concat_df = pd.concat([pd.get_dummies(df[col], drop_first=True, prefix=col) for col in columns], axis=1, copy=True)
    one_hot_cols = concat_df.columns
    return concat_df, one_hot_cols
