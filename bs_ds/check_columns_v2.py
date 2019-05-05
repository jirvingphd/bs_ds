def check_unique(df, columns=None):
    """
    Prints unique values for all columns in dataframe. If passed list of columns,
    it will only print results for those columns
    8************  >
    Params:
    df: pandas DataFrame
    columns: list containing names of columns (strings)

    Returns: None
        prints values only
    """
    if columns is None:
        columns = df.columns

    for col in columns:
        nunique = df[col].nunique()
        unique_df = pd.DataFrame(df[col].value_counts())
        print(f"\n{col} Type: {df[col].dtype}\nNumber unique values: {nunique}")
        display(unique_df)
    pass


# Check columns returns the datatype, null values and unique values of input series
def check_column(panda_obj, columns=None,nlargest='all'):
    """
    Prints column name, dataype, # and % of null values, and unique values for the nlargest # of rows (by valuecount_.
    it will only print results for those columns
    ************
    Params:
    panda_object: pandas DataFrame or Series
    columns: list containing names of columns (strings)

    Returns: None
        prints values only
    """
    # Check for DF vs Series
    if type(panda_obj)==pd.core.series.Series:

        print(f"Column: df['{series.name}']':")
        print(f"dtype: {series.dtype}")
        print(f"isna: {series.isna().sum()} out of {len(series)} - {round(series.isna().sum()/len(series)*100,3)}%")

        print(f'\nUnique non-na values:')
        if nlargest =='all':
            print(series.value_counts())
        else:
            print(series.value_counts().nlargest(nlargest))


    elif type(panda_obj)==pd.core.frame.DataFrame:
        df = panda_obj
        for col_name in df.columns:
            col = df[col_name]
            print(f"Column: df['{col_name}']':")
            print(f"dtype: {col.dtypes}")
            print(f"isna: {col.isna().sum()} out of {len(col)} - {round(col.isna().sum()/len(col)*100,3)}%")

            print(f'\nUnique non-na values:')
            if nlargest =='all':
                print(col.value_counts())
            else:
                print(col.value_counts().nlargest(nlargest))
