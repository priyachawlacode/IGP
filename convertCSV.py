import pandas as pd
import os

SuccessThreshold = 360

"""

Success threshold is time in seconds. I've set it as 240 as a default as the
dataset used shows very long calls on average (with 40 seconds every call was classified as successful)

"""

#load the data
path = os.path.abspath(os.path.dirname(__file__)) + "\dummy_data_large_v1.csv"
data = pd.read_csv(path)
start = data['start']
end = data['end']
data = data.drop(['cin'],axis=1)
data = data.drop(['success'],axis=1)
data = data.drop(['start'],axis=1)
data = data.drop(['end'],axis=1)
""" data['date'] = pd.to_datetime(start, dayfirst=True).dt.date
data['WeekDay'] = pd.to_datetime(data['date']).dt.day_of_week 

Specific date in DD/MM/YYYY is not needed as we are trying to categorise them into a specific day of the week for the decision tree
"""
data['type'] = data['type'].replace({'oc':0,'ic':1,'ib':2})
data['WeekDay'] = pd.to_datetime(start, dayfirst=True).dt.day_of_week
data['StartTime'] = pd.to_datetime(start, dayfirst=True).dt.hour

""" data['StartTime'] = pd.to_datetime(start, dayfirst=True).dt.time
data['EndTime'] = pd.to_datetime(end, dayfirst=True).dt.time

StartTime and Endtime in HH/MM/SS format is not needed as we are trying to categorise them into each hour for the decision tree
 """
data['CallDuration'] = pd.to_datetime(end, dayfirst=True) - pd.to_datetime(start, dayfirst=True)
data['CallDuration'] = data['CallDuration'].dt.total_seconds()
data['SuccessfulCall'] = (data['CallDuration'] >= SuccessThreshold)

data = data.drop(['CallDuration'],axis=1)

"""
Call duration not needed once we've classified successful calls or not. Can keep if you want.
"""


print(data)
data.to_csv(os.path.dirname(__file__) + "\dummy_data_large_v2.csv",sep=';', index=False)
