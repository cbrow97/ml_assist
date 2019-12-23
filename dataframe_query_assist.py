from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import re

data_source = pd.read_excel(r'\\CAFRFD1CDFILE08.itservices.sbc.com\ABAnalytics\JiraExtracts\CROTickets_All_Cleaned.xlsx')

class DataSource:
    """Creates DataFrame from data_source variable. Manipulates DataFrame for initial cleanup"""
    def __init__(self, data_source):
        self.data_source = data_source

    def create_df(self):
        """
        Creates DataFrame from data_source variable. data_source is an excel file saved on the network:
        \\CAFRFD1CDFILE08.itservices.sbc.com\ABAnalytics\JiraExtracts\CROTickets_All_Cleaned.xlsx
        """
        df = pd.DataFrame(self.data_source)        
        df.columns = df.columns.str.replace(' ', '')
        return df
        

class ConfigureDataFrame:
    """
    Includes functions to filter DataFrame columns to only include 'Hypothesis' and 'TestResult', 
    filter DataFrame 'TestResult' to losses only or wins only.
    """
    def __init__(self):
        pass
    
    def query_df(self, df, *args, **kwargs):
        """
        Quickly query a DataFrame by passing paramters

        Paramters
        ---------
        df : Pandas DataFrame
            The DataFrame to evaluate a query against

        args : conditional statements (and, or, between)
            Takes input of conditional statements for the query. The total conditionals should equal one minus the total number
            of columns being queried (column arguments are stated through kwargs)

            Error catching has been implemented to run if a conditional is passed that isn't in the approved_conditionals list (and, or, between)

        kwargs : 
            Key:
                The key is the column header name that you want to query
                NOTE: Spaces should be stripped from column headers prior to running this function
            Value:
                The value portion of the arugment is prefixed by an operator statements (!=, ==, <, >)
                
                Error catching has been implemented to run if an operator is passed that isn't in the approved_operators list (!=, ==, <, >)
            
            Example input:
                TestResult='!=Win'

        Example function call:
        
        df = cdf.query_df(df, 'and', 'and', TestResult='!=Win', AnalysisLead = '==Colton Brown', OptimizationLead='==Sarah V Postal')
        """    
        try:
            if len(args) != len(kwargs) - 1:
                error_message_1 = 'ERROR_1 occured when processing custom function: query_df\n' \
                                + 'The count of conditional statements must be exactly one less than the number of column arguments\n' \
                                + 'Please check the input values and try again'
                raise ValueError         
        except:
            exit(error_message_1)
        
        

        #List of approved operators and conditions. Any values outside of these lists will produce errors
        approved_operators = ['!=', '==', '<', '>']
        approved_conditions = ['and', 'or', 'between']

        #Searches value input string for operators and appends to operators list
        used_operators = []
        for column, value in kwargs.items():
            for operator in approved_operators:
                if re.search(operator, value):
                    used_operators.append(operator)
                    kwargs[column] = value.replace(operator, '')

        #Checks to see if an invalid operator was used
        #If the length of used_operators list is less than the length of kwargs, this error will throw
        try: 
            if len(used_operators) < len(kwargs):
                error_message_2 = 'ERROR_2 occured when processing custom function: query_df\n' \
                                + 'Operators must be one of the following: !=, ==, <, or >\n' \
                                + 'Please check the input values and try again'
                raise ValueError
        except ValueError:
            exit(error_message_2)
        
        
        #Add all used conditions to a list called used_conditions
        used_conditions = [condition for condition in args]

        #Checks to see if an invalid condition was used
        #If there are any conditions in the used_conditions list that do not exist in the approved_conditions list, this error will throw
        invalid_condinitions = [condition for condition in used_conditions if condition not in approved_conditions]
        try:
            if invalid_condinitions:
                error_message_3 = 'ERROR_3 occured when processing custom function: query_df\n' \
                                + 'Conditions must be one of the following: and, or, between\n' \
                                + 'Please check the input values and try again'
                raise ValueError
        except ValueError:
            exit(error_message_3)


        #Finds operator in value string and removes the operator from value
        #Value refers to the query parameter value
        #Exmaple:
        #   The key TestResult with a value of '!=Win' would turn into:
        #   TestResult='Win'       
        
        #for column, value in kwargs.items():
        #    for operator in operators:
        #        if re.search(operator, value):
        #            kwargs[column] = value.replace(operator, '')
        message = 'This is the proposed query syntax: \n'
        i = 0 

        for column, value in kwargs.items():
            if i < len(used_conditions):
                message += column + ' ' + used_operators[i] + ' ' + value + ' ' + used_conditions[i] + ' '
                i+=1
            else:
                message += column + ' ' + used_operators[i] + ' ' + value + ' '     
        
        #Creates query string for df.query function. 
        #used_operators and used_condition list are referenced to concat with the column, value pair from the kwargs items        
        i = 0
        query = ''
        if len(used_conditions) == 0:
            query +=  f'{column}' + ' ' + used_operators[i] + ' "' + f'{value}' + '"'
        else:
            for column, value in kwargs.items():
                if i < len(used_conditions):
                    query +=  f'{column}' + ' ' + used_operators[i] + ' "' + f'{value}' + '" ' + used_conditions[i] + ' '
                    i += 1
                #There will always be one less condition than operators
                #This else statement is to account for the list of used_condition being one smaller than the used_operators list
                else:
                    query +=  f'{column}' + ' ' + used_operators[i] + ' "' + f'{value}' + '" '                               
        df = df.query(query)

        print(message)
        user_input = input('Do you want to proceed with the query execution (y/n): ')

        if user_input.lower() == 'y':
            pass
        else:
            return df  

    def select_columns(self, df, *args):
        columns = list(args)
        invalid_columns = [column for column in columns if column not in df.columns]
        
        try:
            if invalid_columns:
                error_message_1 = 'Column(s) passed as arguments were not found in the DataFrame: \n'
                for column in invalid_columns:
                    error_message_1 += str(column)
                error_message_1 += '\nCorrect the column names and try again'
                raise ValueError
        except ValueError:
            exit(error_message_1)
        
        df = df.loc[:, columns]

        return df



ds = DataSource(data_source)
df = ds.create_df() #this is the excel output


cdf = ConfigureDataFrame()
df = cdf.select_columns(df, 'Hypothesis', 'TestResult', 'AnalysisLead')
df = cdf.query_df(df, 'and', TestResult='==Win', AnalysisLead='==Michael Benton')
#print(df)
#df = cdf.query_df(df, 'and', TestResult='==Win', AnalysisLead='==Colton Brown')


