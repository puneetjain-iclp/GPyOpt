from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

def is_weekend(x):
    """
    Returns whether it is a weekend or not
    """
    
    if (x==5 or x==6) :
        weekend =  1
    else:
        weekend =  0
    
    return weekend
    
def make_date_features(df, date_col):
    """ 
    Returns a dataframe with date features
    """
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    if df[date_col].dt.hour.any():
        df[date_col + '_hour_of_day'] = df[date_col].dt.hour
        
    df[date_col + '_week'] = df[date_col].dt.week
    df[date_col + '_month'] = df[date_col].dt.month
    df[date_col + "_year"] = df[date_col].dt.year
    df[date_col + '_day_of_year'] = df[date_col].dt.dayofyear
    df[date_col + '_week_of_year'] = df[date_col].dt.weekofyear
    df[date_col + "_weekday"] = df[date_col].dt.weekday
    df[date_col + "_quarter"] = df[date_col].dt.quarter
    df[date_col + "_day_of_month"] = df[date_col].dt.day
    df[date_col + "_is_weekend"] = df[date_col + "_weekday"].apply(is_weekend)
    
    df.drop(date_col,axis=1,inplace=True)
    
    return df


def LightGBM_transform(df):
    
    df = make_date_features(df, 'BornDateTime')
    df = make_date_features(df, 'startDate')
    df = make_date_features(df, 'endDate')
        
    
    df.rename(columns={
                    "Gold_GTB_rating"   : "gold_gtb_rating_final",
                    "Bronze_GTB_rating" : "bronze_gtb_rating_final",
                    "Silver_GTB_rating" : "silver_gtb_rating_final"
                        },inplace = True)

    cat_cols = ['SourceType','policyType','areaCode','pricing_area_code','groupType','ageBand',
               'AMD','AYG','OAP','ITN','VOP','INF','AMR','quote_GRPAGE'] + [x for x in df.columns if 'Date' in x]
    
    df = df.fillna(-999)
    
    label_encode = LabelEncoder()

    for x in cat_cols:
        df[x] = label_encode.fit_transform(df[x].astype(str))
    
    return df