import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import chapter_1 as c1

#titanic = sns.load_dataset('titanic')
titanic = pd.read_csv(r'C:\Users\cb049c\Desktop\titanic_data.csv')
df = pd.DataFrame(titanic)

X = titanic[['sex', 'age', 'fare', 'class', 'embark_town', 'alone']]
y = titanic['survived']



def remove_nulls(df, percent_thres):
    """
    Evaluates how to handle nulls for each feature in a DataFrame 
    
    Parameters:
        df - the DataFrame to manipulate and return
            NOTE: The input DataFrame should be fit_transformed before processing this function. If not, you will error on features that are non-numerical values

        percent_thres - the percent threshold to which you want the mean to replace nulls if null value percent is higher than the input percent_thres
    """
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

def remove_string_outliers(df, percent_thres=5, output=False):
    """
    Removes any non-numeric values from a DataFrame if a data point is makes up less than the provided percent_thres parameter (default is 5%)

    The function takes a count of each value per column in a DataFrame. It will divide this count into the total values in the column to determine
    if the value makes up less than the provided percent_thres input. If the condition returns true, the value will be removed from the specificed
    column within the DataFrame which inturn removes the entire row.
    
    There is an optional output of the column, its value, and the percent total that can printed in the terminal (use output=True in the function call)

    Parameters:
        df - the DataFrame to manipulate and return

        percent_thres - the percent threshold to remove values if the total percent of outputs are less than the provided percent_threshold
        NOTE: Default is set to 5

        output - set equal to True to return an output of the column, its value, and the percent total in the terminal

    """
    remove_values_disc = []
    total_values = len(df)
    for column in df.columns:
    #    try:
    #        df.loc[0, column] + 0
    #        
    #    except Exception:
        if df[column].dtype not in ['int', 'long', 'float', 'complex']:
            #create list of all values present in the column    
            column_values = [value for value in df[column].value_counts().index.values]
            
            #create list of all counts within the column
            value_counts = [count for count in df[column].value_counts().values]
            
            #zip the column_values and value_counts list to create a dictionary
            value_counts_dict = dict(zip(column_values, value_counts))

            #Create another dictionary of values to remove from a column. Each value count is compared to the percent_thres parm to check if it is lower than the 
            #provided threshold
            values_to_remove = {value:count for (value, count) in value_counts_dict.items() if ((count / total_values) * 100) < percent_thres}                        
            
            #A blank dictionary to add what values are to be removed and the percent of the dataset they make up. The dictionary is remade through each column loop
            #as the remove_values_disc list is appended for each item removed and the details of that item (the colum it's in, the value, and the percent of the dataset)
            removed_values = {}
        
            for value, counts in values_to_remove.items():
                df = df[df[column] != value]
                removed_values.update({value:round(((counts / total_values) * 100), 2)})
            
            #Appends to remove_values_disc to later iterate over and provide the user with an output of what was removed
            #User will need to use optional argument of output=True to utilize this functionality
            for value, percent in removed_values.items():
                message = f'The value of "{value}" within the column [{column}] was removed. This value only made up {percent}% of the dataset and is considered an outlier.'
                remove_values_disc.append(message)
    
    #Returns output of values that were removed
    #User will need to use optional argument of output=True to utilize this functionality
    if output:
        if remove_values_disc:
            for message in remove_values_disc:
                print(message)
        else:
            print('No values in the passed DataFrame exceeded the percent_thres parm')
    
    return df

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
X = remove_string_outliers(X, percent_thres=35, output=True)
print(X)
#X = fit_transform_features(X)
#print(X)
#
#X = c1.rescale_normalization(X)
#print(X)
#detect_outliers(X)

#X = fit_transform_features(X)
#X = c1.rescale_normalization(X)
#print(X)
