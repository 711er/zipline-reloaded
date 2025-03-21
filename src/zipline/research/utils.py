from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline

from zipline.data import bundles
import dateutil.parser
import pandas_datareader.data as yahoo_reader

from zipline.pipeline import Pipeline
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.data.data_portal import DataPortal
from zipline.utils.calendar_utils import get_calendar
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.assets import Equity
from zipline.pipeline.data import Column
from zipline.pipeline.domain import US_EQUITIES

BUNDLE_DATA = None
PRICING_LOADER = None
END_DT = None
DOMAIN = None


def set_bundle_data(bundle_name='quandl'):
    global BUNDLE_DATA, PRICING_LOADER, DOMAIN
    BUNDLE_DATA = bundles.load(bundle_name)
    print(f"Global BUNDLE_DATA@{id(BUNDLE_DATA)}"
          f" is set for bundle '{bundle_name}'.")

    PRICING_LOADER = USEquityPricingLoader.without_fx(
        BUNDLE_DATA.equity_daily_bar_reader,
        BUNDLE_DATA.adjustment_reader
    )


def make_loader(custom_loader=None):
    """
    Return choose_loader which takes a custom loader as input and 
    returns appropriate loader for use in pipline engine creation
    """
    if custom_loader is None:
        def choose_loader(column):
            """ 
            Define the function for the get_loader parameter
            Set the dataloader
            """
            if column not in USEquityPricing.columns:
                raise Exception('Column %s not in USEquityPricing' % column)
            return PRICING_LOADER
    else:
        def choose_loader(column):
            if column in USEquityPricing.columns:
                return PRICING_LOADER
            try:
                return custom_loader.get(column)
            except KeyError:
                raise ValueError(
                    "No PipelineLoader registered for column %s." % column
                )
    
    return choose_loader


def default_loader():
    return make_loader()


def make_pipeline(pipeline_dict: Dict[str, Pipeline]=None,
                  universe=None, domain=None):
    """
    Make pipeline based on input dictionary of pipelines.
    """
    
    if pipeline_dict is None:
        pipeline = Pipeline()   # empty pipeline
    else:
        pipeline = Pipeline(columns=pipeline_dict, 
                            screen=universe,
                            domain=domain)
    
    return pipeline


def run_pipeline(pipeline, start_date, end_date, loader=None):
    global BUNDLE_DATA
    if BUNDLE_DATA is None:
        set_bundle_data(bundle_name='quandl')
    print(f'BUNDLE_DATA@{id(BUNDLE_DATA)} is used.')
    
    custom_loader = make_loader(loader)
    engine = SimplePipelineEngine(get_loader=custom_loader,
                                  asset_finder=BUNDLE_DATA.asset_finder)
    
    return engine.run_pipeline(pipeline, start_date, end_date)


# def dataframe_loader(df: pd.DataFrame):
#     """
#     Generic custom dataframe loader
#     """
#     column = Column(dtype=float)
#     loader = {column: df}
#     return loader
    
def create_data_portal(_bundle_name, _trading_calendar, start_date):
    global BUNDLE_DATA
    if not BUNDLE_DATA:
        set_bundle_data(_bundle_name)
    
    print(f'BUNDLE_DATA@{id(BUNDLE_DATA)} is used.')
    # Create a data portal
    data_portal = DataPortal(
        BUNDLE_DATA.asset_finder,
        trading_calendar=_trading_calendar,
        first_trading_day=start_date,
        equity_daily_reader=BUNDLE_DATA.equity_daily_bar_reader,
        adjustment_reader=BUNDLE_DATA.adjustment_reader
    )
    return data_portal


def get_pricing(data_portal: DataPortal, trading_calendar, 
                assets: List[Equity], 
                start_date, end_date, field='close'):
    # Set the given start and end dates to Timestamps. The frequency string C is used to
    # indicate that a CustomBusinessDay DateOffset is used

    global END_DT
    END_DT = end_date
    start_dt = start_date

    # Get the locations of the start and end dates
    end_loc = trading_calendar.closes.index.get_loc(END_DT)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    # return the historical data for the given window
    return data_portal.get_history_window(
        assets=assets, end_dt=END_DT, 
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily'
    )


def create_pipeline_engine(bundle_name='quandl', get_loader=default_loader):
    global BUNDLE_DATA
    if not BUNDLE_DATA:
        set_bundle_data(bundle_name)
    
    print(f'BUNDLE_DATA@{id(BUNDLE_DATA)} is used.')
    # Create a Pipeline engine
    engine = SimplePipelineEngine(get_loader=get_loader,
                                  asset_finder=BUNDLE_DATA.asset_finder)
    return engine


def get_equity(symbol):
    return BUNDLE_DATA.asset_finder.lookup_symbol(symbol, END_DT)


def get_pipeline_output_for_equity(df, symbol, drop_level=False):
    """
    pipeline output contains many equities, if you want to view the pipeline for jsut one equity
    you could use this method which slices a multiindex df (dates and equities)
    :param df:
    :param symbol:
    :param drop_level: if True it will drop the equity (level 1) index and return df with 1 level index.
    :return:
    """
    equity = get_equity(symbol)
    df = df[df.index.get_level_values(1) == equity]
    if drop_level:
        df.index = df.index.droplevel(1)
    return df


def pipeline_train_test_split(X,
                              y,
                              test_size=0.3,
                              validate_size=0.3,
                              should_validate=False):
    """
    sklearn train_test_split
    :param df:
    :return:
    """
    a, b, c, d = train_test_split(X.index.levels[0],
                                  y.index.levels[0],
                                  test_size=test_size,
                                  random_state=101)
    X_train = X.loc[list(a)]
    X_test = X.loc[list(b)]
    y_train = y.loc[list(c)]
    y_test = y.loc[list(d)]
    if should_validate:
        a, b, c, d = train_test_split(X_train.index.levels[0],
                                      y_train.index.levels[0],
                                      test_size=validate_size,
                                      random_state=101)
        X_train = X.loc[list(a)]
        X_validate = X.loc[list(b)]
        y_train = y.loc[list(c)]
        y_validate = y.loc[list(d)]
        return X_train, X_validate, X_test, y_train, y_validate, y_test
    return X_train, X_test, y_train, y_test


class DATE(str):
    """
    date string in the format YYYY-MM-DD
    """
    def __new__(cls, value):
        if not value:
            raise ValueError('Unexpected empty string')
        if not isinstance(value, str):
            raise TypeError(f'Unexpected type for DATE: "{type(value)}"')
        if value.count("-") != 2:
            raise ValueError(f'Unexpected date structure. expected '
                             f'"YYYY-MM-DD" got {value}')
        try:
            dateutil.parser.parse(value)
        except Exception as e:
            msg = f"{value} is not a valid date string: {e}"
            raise Exception(msg)
        return str.__new__(cls, value)


def get_benchmark(symbol=None, 
                  start: DATE = None, 
                  end: DATE = None, 
                  other_file_path=None):
    bm = yahoo_reader.DataReader(symbol,
                                 'yahoo',
                                 pd.Timestamp(DATE(start)),
                                 pd.Timestamp(DATE(end)))['Close']
    bm.index = bm.index.tz_localize('UTC')
    return bm.pct_change(periods=1).fillna(0)
