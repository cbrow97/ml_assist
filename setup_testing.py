import seaborn as sns
import pandas as pd

#    column_values = [value for value in df['sex'].value_counts().index.values]
#    value_counts = [count for count in df['sex'].value_counts().values]
#    test_dict = dict(zip(column_values, value_counts))
#    values_to_remove = [value for (value, count) in test_dict.items() if ((count / sum(value_counts)) * 100) < 50]
#    for value in values_to_remove:
#        df = df[df.sex != value]
#    print(values_to_remove)
#
#    print(df)

    remove_values_disc = []
    total_values = len(df)
    removed_values = {}
    for column in df.columns:
        if df[column].dtype not in ['int', 'long', 'float', 'complex']:
 
            column_values = [value for value in df[column].value_counts().index.values]

            value_counts = [count for count in df[column].value_counts().values]
            
            value_counts_dict = dict(zip(column_values, value_counts))

            values_to_remove = {value:count for (value, count) in value_counts_dict.items() if ((count / total_values) * 100) < percent_thres}                        

            for value, counts in values_to_remove.items():
                removed_values.update({column + '!@#' + str(value):round(((counts / total_values) * 100), 2)})

    for value, percent in removed_values.items():
        column_clean = value[:value.find('!@#')]
        value_clean =  value[value.find('!@#'):].replace('!@#', '')
        
        df = df[df[column_clean] != value_clean]

        message = f'The value of "{value_clean}" within the column [{column_clean}] was removed. This value only made up {percent}% of the dataset and is considered an outlier.'
        remove_values_disc.append(message)

    if output:
        if remove_values_disc:
            for message in remove_values_disc:
                print(message)
        else:
            print('No values in the passed DataFrame exceeded the percent_thres parm')

    return df