import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import chapter_1 as c1

titanic = sns.load_dataset('titanic')
X = titanic[['sex', 'age', 'fare', 'class', 'embark_town', 'alone']]
y = titanic['survived']



def remove_nulls(df, percent_thres):
    '''
    Evaluates how to handle nulls for each feature in a DataFrame 
    
    Parameters:
        df - the DataFrame to manipulate and return
            NOTE: The input DataFrame should be fit_transformed before processing this function. If not, you will error on features that are non-numerical values

        percent_thres - the percent threshold to which you want the mean to replace nulls if null value percent is higher than the input percent_thres
    '''
    for column in df.columns:
        na_counter = df[column].isnull().sum()
        percent_of_total = (na_counter / len(df)) * 100
        
        #if the percent of nulls exceeds the precent_thres value, nulls will be replaced by the mean of the feature 
        if percent_of_total > percent_thres:
            mean = df[column].mean()
            mean = round(mean)
            df[column].fillna(mean, inplace=True)

    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def remove_numerical_outliers(df, standard_deviations=3):   
    for column in df.columns:
        try:
            df.loc[0, column] + ''
        except TypeError:
            if df[column].dtype != 'bool':
                min_thres = df[column].mean() - (standard_deviations * df[column].std())

                max_thres = df[column].mean() + (standard_deviations *df[column].std())

                df = df[df[column] >= min_thres]
                df = df[df[column] <= max_thres]

    df = df.reset_index(drop=True)
    return df

def remove_string_outliers(df, percent_thres=5):
    for column in df.columns:
        try:
            df.loc[0, column] + 0
        except Exception:
            total_values = df[column].value_counts()
            print(total_values)
            print(type(total_values))
            ###NEED TO WORK ON
            ###TODO
            ### explore pd.series.iteritems() to iterate over each category and determine if it's threshold (default 5%) is not exceeded (will remove values less than 5% of total data)
            #for item in df.itertuples():
            #    print(item)
            #print(df[column].value_counts())
                
def fit_transform_features(df):
    try:
        enc = LabelEncoder()
    except Exception:
        exit('The fit_transform_features function utilizes the LabelEncoder() class from sklearn.preprocessing\n'\
                'Process the following import string to resolve: from sklearn.preprocessing import LabelEncoder')
    for column in df.columns:
    #Use try/except block to ignore any columns that aren't a string datatypes
        try:
            df.loc[0, column] + 0
        #Utilize LabelEncoder to change string values to numerical values which represent their string values
        #Think SQL table normalization
        except TypeError:
            df[column] = pd.Series(enc.fit_transform(df[column].astype('str')))
        
        if df[column].dtype == 'bool':
            df[column] = pd.Series(enc.fit_transform(df[column].astype('str'))) 
    
    return df



X = remove_nulls(X, 15)
X = remove_numerical_outliers(X)
remove_string_outliers(X)
#X = fit_transform_features(X)
#print(X)
#
#X = c1.rescale_normalization(X)
#print(X)
#detect_outliers(X)

#X = fit_transform_features(X)
#X = c1.rescale_normalization(X)
#print(X)
