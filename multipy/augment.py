from summarize.insights import get_vector_type_combos
from preppy.clean_list import clean_dates
import numpy as np
import pandas as pd
import itertools


def get_column_type_combos(column_combo,column_types):
    """Get possible semantic data type combinations for a given combination 
    of column headers.

    Args:
        column_combo: A tuple of column headers
        column_types: Dict of column header -> semantic types

    Returns:
        An iterable of all possible data type combinations
    """

    combo_types = []
    for header in column_combo:
        c_types = [(header, data_type) for data_type in column_types[header]]
        combo_types.append(c_types)

    combos = itertools.product(*combo_types)
    return combos


def process_coords(column_values,column_types):
    """Take a latitude and longitude and return a formatted coordinate pair, 
    for example: [<lat>87.123,<lng>179.234] becomes '87.123,179.234'

    Args:
        column_values: A list of (column_header,column_values) tuples
        column_types: Dict of column header -> semantic types

    Returns:
        A list of (new_column_header,new_column_values) tuples
    """

    lat_column = [c for c in column_values if 'latitude' in column_types[c[0]]]
    lng_column = [c for c in column_values if 'longitude' in column_types[c[0]]]
    new_column_name = '_'.join([lat_column[0],lng_column[0]])

    if len(lat_column) != 1 or len(lng_column) != 1:
        return []

    lat_values = np.array(lat_column[1]).astype(str)
    lng_values = np.array(lng_column[1]).astype(str)
    commas = [',' for _ in range(len(lat_values))]

    new_values = np.core.defchararray.add(
                                    np.core.defchararray.add(lat_values,commas),
                                    lng_values)

    return [(new_column_name,new_values)]


def process_address_components(column_values,column_types):
    """Take components of an address and return a single value, ideal for 
    geocoding address strings. 

    For example: [<street>'123 Main St', <city>'Austin', <state>'TX'] becomes
    '123 Main St Austin, TX'

    Args:
        column_values: A list of (column_header,column_values) tuples
        column_types: Dict of column header -> semantic types

    Returns:
        A list of (new_column_header,new_column_values) tuples
    """

    street_column = [('street',c[1]) for c in column_values if 'street' in column_types[c[0]]]
    city_column = [('city',c[1]) for c in column_values if 'city' in column_types[c[0]]]
    region_column = [('state',c[1]) for c in column_values if 'state' in column_types[c[0]]]

    columns = []
    for c in [street_column,city_column,region_column]:
        if len(c) > 0:
            columns.append(c[0])

    df = pd.DataFrame(np.array([c[1] for c in columns]).T,columns=[c[0] for c in columns])
    
    if len(street_column) > 0:
        address = df['street'].astype(str).str.cat(df['city'], sep=' ')
    else:
        address = df['city']

    if len(region_column) > 0:
        address = address.str.cat(df['state'], sep = ', ')

    return [('full_address', address.values)]


def process_datetime(column_values,column_types):
    """Take a datetime and return the day of month, day of week and hour of 
    day for each observation.

    For example: '2016-01-01T04:15:23' becomes
    [<hour_of_day>04,<day_of_week>Friday,<day_of_month>1]

    Args:
        column_values: A list of (column_header,column_values) tuples
        column_types: Dict of column header -> semantic types

    Returns:
        A list of (new_column_header,new_column_values) tuples
    """

    values = column_values[0][1]
    header = column_values[0][0]
    week_header = header+'_day_of_week'
    month_header = header+'_day_of_month'
    hour_header = header+'_hour_of_day'

    tmp_col = 'Date'
    values = clean_dates(values)
    df = pd.DataFrame(values,columns=[tmp_col])

    new_columns = []

    day_of_week = {
        0:'Monday', 
        1:'Tuesday', 
        2:'Wednesday', 
        3:'Thursday', 
        4:'Friday', 
        5:'Saturday', 
        6:'Sunday'
    }
    
    week_column = (week_header, df[tmp_col].dt.dayofweek.map(day_of_week).values)
    month_column = (month_header,df[tmp_col].dt.day)

    if len(list(set(week_column[1]))) == 1:
        week_column = None
    else:
        new_columns.append(week_column)
    
    if len(list(set(month_column[1]))) == 1:
        month_column = None
    else:
        new_columns.append(month_column)

    times = df[tmp_col].dt.time.astype(str)
    if any([t != '00:00:00' for t in times]):
        hour_column = (hour_header,df[tmp_col].dt.hour.values)
        new_columns.append(hour_column)

    return new_columns


def augment_columns(columns=None, df=None, column_types={}):
    """Create new columns of data from columns or combinations of columns that 
    match pre-defined semantic data type patterns. Accepts either a list of 
    (column_header, column_values) tuples or a pandas dataframe.

    Mapping between semantic type patterns and the function for processing 
    columns/column combinations that fit that pattern are defined in 
    the `processors` dict. 

    Function iterates over every possible combination of columns of lengths 
    1 to 3 (so [Column A,Column B,Column C], [Column A,Column B], 
    [Column B,Column C], etc). Each column has a list of associated semantic 
    types, so every column combination may have many semantic type combinations. 
    Any of these combinations that matches a pattern defined in `processors` 
    will handled by the mapped processor function, and generate new columns.

    Any given processor will be used only once with the most available 
    information. So if a dataset contains a street address, city name, and 
    region name, creating a full address from concatenating all three of these 
    properties is preferred over creating an address from just two of them. 

    Args:
        columns: A list of (column_header,column_values) tuples
        df: Pandas dataframe
        column_types: Dict of column header -> semantic types

    Returns:
        A list of (new_column_header,new_column_values) tuples
    """

    if not columns and not df:
        raise('columns or df is required')

    if not columns:
        headers = df.columns
        column_values = zip(df.columns,df.as_matrix())
    else:
        headers = [c[0] for c in columns]
        column_values = columns

    processors = {
        'latitude,longitude': {
            'process': process_coords
        },
        'datetime': {
            'process': process_datetime
        },
        'city,state,street': {
            'process': process_address_components
        },
        'city,street': {
            'process': process_address_components
        }
    }

    new_columns = []
    performed = []

    for L in reversed(range(1,4)):
        for column_combo in itertools.combinations(headers, L):
            type_combos = get_column_type_combos(column_combo,column_types)
            combo_values = [cv for cv in column_values if cv[0] in list(column_combo)]

            new_columns = []
            for combo in type_combos:
                """ Check if there are any processors associated with this pattern """
                pattern_str = ','.join(sorted([c[1] for c in combo]))
                if pattern_str not in processors:
                    continue

                f = processors[pattern_str]['process']
                if f not in performed:
                    new_columns += (combo_values,column_types)
                    performed.append(f)

    return new_columns
