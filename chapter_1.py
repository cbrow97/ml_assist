import seaborn as sns
import sklearn

#iris = sns.load_dataset('iris')
#iris_features = iris.drop('species', axis=1)
#iris_target = iris['species']

def rescale_normalization(df):
    for column in df.columns:
        max_val = df[column].max()
        min_val = df[column].min()
        
        df[column] = df[column].apply(lambda x: ((x - min_val) / (max_val - min_val)))

    return df

def rescale_standarization(df):
    for column in df.columns:
        mean = df[column].mean()
        std = df[column].std()

        df[column] = df[column].apply(lambda x: ((x - mean) / (std)))
    
    return df

