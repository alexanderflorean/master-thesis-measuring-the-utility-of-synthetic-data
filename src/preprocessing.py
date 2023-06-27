from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler, Binarizer

def create_preprocessing_pipeline(meta):
    list_of_columntransformers = []

    # if there is numeric features, use standard scaler
    if meta['numeric_features'] !=None:
        numerical_columns = meta['numeric_features'] 
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        list_of_columntransformers.append( ('num', numerical_transformer, numerical_columns) )

    # if there is categorical features, use ohe
    if meta['categorical_features'] !=None:
        categorical_columns = meta['categorical_features']
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
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