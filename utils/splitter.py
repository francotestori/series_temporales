from pandas import DataFrame
import numpy as np


def get_validation_year_week(year: int):
    """"""
    return [f'{year}-{week}' for week in range(31, 39)]


def train_test_split_folds(df: DataFrame):
    """"""
    folds = []
    for year in np.sort(df.year.unique()):
        train_upper_limit = f'{year}-31'
        validation_weeks = get_validation_year_week(year)

        train_condition = (~df.year_week.isin(validation_weeks)) & (df.year_week < train_upper_limit)
        val_condition = (df.year_week.isin(validation_weeks))

        train_split = df[train_condition]
        val_split = df[val_condition]

        cols = ['DateAsFloat', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
        folds.append({
            'train': train_split[cols],
            'validation': val_split[cols]
        })

    return folds