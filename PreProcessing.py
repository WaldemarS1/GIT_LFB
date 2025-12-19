#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 12:16:38 2025

@author: Schock
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
pd.set_option('display.max_columns',None)

# -----------------------------------------
# Initialize variables t
# -----------------------------------------

# Paths and files
path = '/Users/Schock/AnacondaProjects/LondonFireBrigade/Data/'
file_i = 'LFB Incident data total.csv' #all 3 data files merged to 1 file for simplicity
file_m = 'LFB Mobilisation data total.csv' #all 3 data files merged to 1 file for simplicity
file_f = 'LondonFireStation.csv'
file_pc_district = 'postcode-outcodes.csv'
file_pc_full = 'ukpostcodes.csv'
file_final = 'LFB_Final.csv'
# -----------------------------------------


# -----------------------------------------
# switches -  control the preprocessing
# -----------------------------------------

# 1 = read and process raw data
# 0 = read a previously saved file from file_final
# in case of 0 no processing will be done apart from reading the data from the file
process_data = 0

# Distance Calculation
# 1 = calculate distance
# 0 = do not calculate distance
dist_calc = 1

# Save final file
# 1 = save processed data in file_final
save_final_file = 1
# -----------------------------------------

start_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
print('Start Time:', datetime.now().strftime('%d-%m-%Y %H:%M:%S'))

def read_raw_data():    
    
    #Read Data
    df_i = pd.read_csv(path + file_i, error_bad_lines=False)
    df_m = pd.read_csv(path + file_m, error_bad_lines=False)
    df_f = pd.read_csv(path + file_f, error_bad_lines=False,index_col = 'Station')
    df_pc_district = pd.read_csv(path + file_pc_district, error_bad_lines=False)
    df_pc_district.drop(columns = 'id', inplace = True)
    df_pc_full = pd.read_csv(path + file_pc_full, error_bad_lines=False)
    df_pc_full.drop(columns = 'id', inplace = True)
    
    return df_i, df_m, df_f, df_pc_district, df_pc_full

def enrich_data(df_i, df_m, df_f, df_pc_district, df_pc_full):
    #add missing coordinates in order to avoid missing distance calculation at a later stage
    #63k records have missing coordinates
    #by adding the following 4 coordinates the number will be reduced to 100 records
    df_pc_full_add = {'postcode': ['SE1 7SD','RM3 8UN','SW3 1AL','TW14 0LH','E22'],
                      'latitude': [51.499217,51.605638,51.500688,51.461035,51.501196],
                      'longitude': [-0.116508,0.22164,-0.161027,-0.413161,-0.02122]}
    
    #merge with ukpostcodes
    df_pc_full = pd.concat([df_pc_full, pd.DataFrame.from_dict(df_pc_full_add)], axis=0)
    
    #combine coordinates of districts and full postcodes to get 1 single list
    df_pc = pd.concat([df_pc_district, df_pc_full], axis=0)
    df_pc.set_index('postcode', inplace=True)
    
    #merge Incident with Mobilisation
    df = df_i.merge(right = df_m, on = 'IncidentNumber', how = 'inner')
    
    #create derived incident postcode from Postcode_district and Postcode_full
    df['Postcode_incident'] = np.where(df['Postcode_full'].isna(),df['Postcode_district'],df['Postcode_full'])
    
    #create derived columns Incident Latitude and Incident Longitude from map to df_pc
    #when information is missing or 0 in df then use the values from df_pc
    map_lat = df_pc['latitude'].to_dict()
    map_lon = df_pc['longitude'].to_dict()
    
    df['Latitude_incident'] = np.where((df['Latitude']==0) | (df['Latitude'].isna()),df['Postcode_incident'].map(map_lat),df['Latitude'])
    df['Longitude_incident'] = np.where((df['Longitude']==0) | (df['Longitude'].isna()),df['Postcode_incident'].map(map_lon),df['Longitude'])
    
    #Create new variable LocationAccuracy and set to 0 when Postcode_full is empty
    #This helps the model learn because we flag imprecise locations
    df['LocationAccuracy'] = np.where(df['Postcode_full'].isna(),0,1)
    
    #Map Postcode_station
    map_station = df_f['PostalCode'].to_dict()
    df['Postcode_station'] = df['DeployedFromStation_Name'].map(map_station)
    
    #Map Latitude_station and Longitude_station
    df['Latitude_station'] = df['Postcode_station'].map(map_lat)
    df['Longitude_station'] = df['Postcode_station'].map(map_lon)  
    
    return df

def feature_engineering(df):
    
    #create new column UPRN_D
    #whenever PropertyCategory is Dwelling then UPRN = 0
    #Use USRN when UPRN is 0, else UPRN
    df['UPRN_D'] = np.where(df['UPRN']==0,df['USRN'],df['UPRN'])
    
    #create new column IncidentGroup_D
    #when SpecialServiceType = NAN then IncidentGroup, else SpecialServiceType
    df['IncidentGroup_D'] = np.where(df['SpecialServiceType'].isna(),df['IncidentGroup'],df['SpecialServiceType'])

    #create new column Easting_D
    #when Easting_m = NAN then Easting_rounded, else Easting_m
    df['Easting_D'] = np.where(df['Easting_m'].isna(),df['Easting_rounded'],df['Easting_m'])    

    #create new column Northing_D
    #when Northing_m = NAN then Northing_rounded, else Northing_m
    df['Northing_D'] = np.where(df['Northing_m'].isna(),df['Northing_rounded'],df['Northing_m'])

    # create new column PumpMinutes_D
    df['PumpMinutes_D'] = (pd.to_datetime(df['DateAndTimeLeft']) - \
                              pd.to_datetime(df['DateAndTimeArrived'])) \
                                  / pd.Timedelta(minutes=1)
                      
    #create date columns
    df['DateAndTimeMobilised'] = pd.to_datetime(df['DateAndTimeMobilised'])
    df['call_year'] = df['DateAndTimeMobilised'].dt.year
    df['call_month'] = df['DateAndTimeMobilised'].dt.month
    df['call_day'] = df['DateAndTimeMobilised'].dt.day
    df['call_weekday'] = df['DateAndTimeMobilised'].dt.weekday
    df['call_hour'] = df['DateAndTimeMobilised'].dt.hour
    
    #remove outliers
    df.loc[df['PumpCount'] > 200, 'PumpCount'] = df.PumpCount.mean()
    df.loc[df['NumCalls'] > 200, 'NumCalls'] = df.NumCalls.mean()
    df.loc[df['Notional Cost (£)'] > 500000, 'Notional Cost (£)'] = df['Notional Cost (£)'].mean()

    
    return df

def drop_col(df):
    cols_drop = [
        #Columns from Incident data
        'IncidentNumber','DateOfCall', 'CalYear_x', 'TimeOfCall','HourOfCall_x', \
        'IncGeo_BoroughName', 'ProperCase','IncGeo_WardName', 'IncGeo_WardNameNew', \
        'Easting_m', 'Northing_m', 'FirstPumpArriving_AttendanceTime', \
        'FirstPumpArriving_DeployedFromStation', 'SecondPumpArriving_AttendanceTime', \
        'SecondPumpArriving_DeployedFromStation', 'Latitude','Longitude', \
        'Postcode_full','FRS','SpecialServiceType','UPRN','USRN', \
        'IncidentGroup','Easting_rounded','Northing_rounded','Postcode_district', \
        'NumPumpsAttending','PumpMinutesRounded',
        
        #Columns from Mobilisation data
        'ResourceMobilisationId','TurnoutTimeSeconds', 'TravelTimeSeconds', \
        'DateAndTimeLeft', 'DateAndTimeReturned', 'DeployedFromStation_Name', \
        'CalYear_y', 'HourOfCall_y', 'BoroughName', 'WardName', 'DateAndTimeMobile', \
        'DateAndTimeArrived', 'DelayCode_Description','PlusCode_Description','PerformanceReporting', \
        'DateAndTimeMobilised'
        ]
        
    df.drop(columns = cols_drop,inplace=True)
    
    return df

def process_nan(df):
    cols_dropna = ['PropertyCategory', 'PropertyType', 'IncGeo_WardCode','IncidentStationGround', \
                   'DeployedFromStation_Code','Latitude_incident','Longitude_incident','Postcode_station', \
                   'Latitude_station','Longitude_station','UPRN_D']
        
    df.dropna(subset = cols_dropna,inplace=True)
    
    #replace NAN with mode
    df['NumCalls'].fillna(df_i['NumCalls'].mode()[0],inplace=True)
    df['DeployedFromLocation'].fillna(df_m['DeployedFromLocation'].mode()[0],inplace=True)
    
    #replace NAN with 0
    df['DelayCodeId'] = df['DelayCodeId'].fillna(0)
    
    #replace NAN with median
    df['PumpMinutes_D'].fillna(df['PumpMinutes_D'].mean(),inplace=True)
    
    return df

def distance_calc(df):
    #calculate the distance between station and incident
    #it takes 5 mins to calculate for entire dataframe
    import geopy.distance
    
    #separate records where coordinates are incomplete
    df_incomp = df[(df['Latitude_incident'].isna()) | \
                 (df['Longitude_incident'].isna()) | \
                     (df['Latitude_station'].isna()) | \
                         (df['Longitude_station'].isna())]
    
    #dropna where coordinates are not complete
    df.dropna(subset = ['Latitude_incident','Longitude_incident','Latitude_station','Longitude_station'],inplace=True)
    
    #define lambda function for distance calculation
    x = lambda a : geopy.distance.geodesic((a['Latitude_station'], a['Longitude_station']), (a['Latitude_incident'], a['Longitude_incident'])).km
    
    #calculate distance
    df['Distance'] = df.apply(x,axis=1)
    
    #Explore Distance
    #round(df['Distance'].describe(),3)
    
    #we can see 3900 records where distance is between 120 and 140km
    #all records dispatch the pump from Postcode_station = PE38 9BD
    #this postcode is in the city of Downham market which is 100km away from London
    #attendance time is reasonable and is an indication that the postcode is probably wrong
    # -> replace by distance mean
    df.loc[df['Distance'] > 100, 'Distance'] = df.Distance.mean()
    
    #plt.boxplot(df['Distance'])
    #plt.hist(df['Distance'], bins=200)
    
    return df

# MAIN CONTROL
if process_data:
    #Read Data
    print('\nRead Data')
    df_i, df_m, df_f, df_pc_district, df_pc_full = read_raw_data()

    #Enrich Data
    print('\nEnrich Data')
    df_enrich = enrich_data(df_i, df_m, df_f, df_pc_district, df_pc_full)

    #Distance calculation
    if dist_calc:
        print('\nCalculate Distance')
        df_enrich = distance_calc(df_enrich)
    else:
        print('\n.....Distance calculation deactivated')

    print('\nFeature Engineering')
    #Feature Engineering
    df_eng = feature_engineering(df_enrich)

    #Drop Columns
    print('\nDrop Columns')
    df = drop_col(df_eng)

    #Process NAN
    print('\nProcess NAN')
    df = process_nan(df)    
    
    if save_final_file:
        print('\nSave prepared data in physical file')
        df.to_csv(path + file_final, index = False)
    else:
        print('\n.....File Saving deactivated')

if process_data == 0:
    print('\nRead previously saved file')
    df = pd.read_csv(path + file_final, error_bad_lines=False)    
    

print('Start Time:', start_time)
print('Finish Time', datetime.now().strftime('%d-%m-%Y %H:%M:%S'),'\n')
