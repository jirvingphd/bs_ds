# -*- coding: utf-8 -*-
"""Collection of DataFrame inspection, styling, and EDA plotting."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as sts
from IPython.display import display

def big_pandas(user_options=None):
    """Changes the default pandas display setttings to show all columns and all rows.
    User may replace settings with a kwd dictionary matching available options.
    """
    import pandas as pd
    if user_options==None:
        options = {
            'display' : {
                'max_columns' : None,
                'expand_frame_repr':False,
                'max_rows':None,
                'precision' : 4,
            }
        }
    else:
        options = user_options

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+
            print(f'{category}.{op}={value}')


def reset_pandas():
    """Resets all pandas options back to default state."""
    pd.reset_option('all')


def ignore_warnings():
    """Ignores all deprecation warnings (future,and pending categories too)."""
    import warnings
    warnings.simplefilter(action='ignore', category=(FutureWarning,DeprecationWarning,PendingDeprecationWarning))

def reset_warnings():
    """Restore the default warnings settings"""
    import warnings
    warnings.simplefilter(action='default', category=(FutureWarning,DeprecationWarning,PendingDeprecationWarning))

def list2df(list):#, sort_values='index'):
    """ Quick turn an appened list with a header (row[0]) into a pretty dataframe.
    Ex: list_results = [["Test","N","p-val"]] #... (some sort of analysis performed to produce results)
        list_results.append([test_Name,length(data),p])
        list2df(list_results)
    """
    with pd.option_context("display.max_rows", None, "display.max_columns", None ,
    'display.precision',3,'display.notebook_repr_htm',True):

        df_list = pd.DataFrame(list[1:],columns=list[0])
        return df_list


def check_df_for_columns(df, columns=None):

    """
    Checks df for presence of columns.

    args:
    **********
    df: pd.DataFrame to find columns in
    columns: str or list of str. column names
    """
    if not columns:
        print('check_df_for_columns expected to be passed a list of column names.')
    else:
        for column in columns:
            if not column in df.columns:
                continue
            else:
                print(f'{column} is a valid column name')
    pass


def check_unique(df, columns=None):
    """
    Prints unique values for all columns in dataframe. If passed list of columns,
    it will only print results for those columns
    8************  >
    Params:
    df: pandas DataFrame, or pd.Series
    columns: list containing names of columns (strings)

    Returns: None
        prints values only
    """
    # check for columns
#     if columns is None:
        # Check if series, even though this is unnecesary because you could simply
        # Call pd.series.sort_values()
    if isinstance(df, pd.Series):
        # display all the value counts
        nunique = df.nunique()
        print(f'\n---------------------------\n')
        print(f"{df.name} Type: {df.dtype}\nNumber unique values: {nunique}")
        return pd.DataFrame(df.value_counts())

    else:
        if columns is None:
            columns = df.columns

        for col in columns:
            nunique = df[col].nunique()
            unique_df = pd.DataFrame(df[col].value_counts())
            print(f'\n---------------------------')
            print(f"\n{col} Type: {df[col].dtype}\nNumber unique values: {nunique}.")
            display(unique_df)
        pass


def check_numeric(df, columns=None, unique_check=True, return_list=False):

    """
    Iterates through columns and checks for possible numeric features labeled as objects.
    Params:
    ******************
    df: pandas DataFrame

    unique_check: bool. (default=True)
        If true, distplays interactive interface for checking unique values in columns.

    return_list: bool, (default=False)
        If True, returns a list of column names with possible numeric types.
    **********>
    Returns: list of column names if return_list=True
    """
    from bs_ds.bamboo import list2df

    display_list = [['Column', 'Numeric values','Total Values', 'Percent']]
    outlist = []
    print(f'\n---------------------------------------------------\n')
    print(f'# of Identified Numeric Values in "Object" columns:')

    # Check for user column list
    columns_to_check = []
    if columns == None:
        columns_to_check = df.columns
    else:
        columns_to_check = columns
    # Iterate through columns

    for col in columns_to_check:

        # Check for object dtype,
        if df[col].dtype == 'object':

            # If object, check for numeric
            if df[col].str.isnumeric().any():

                # If numeric, get counts
                vals = df[col].str.isnumeric().sum()
                percent = round((df[col].str.isnumeric().sum()/len(df[col]))*100, 2)
                display_list.append([col, vals,len(df[col]), percent])
                outlist.append(col)

    list2show = list2df(display_list)
    list2show.set_index('Column',inplace=True)
    display(list2show)

    if unique_check:
        unique = input("display unique values? (Enter 'y' for all columns, a column name, or 'n' to quit):")

        while unique != 'n':

            if unique == 'y':
                check_unique(df, outlist)
                break

            elif unique in outlist:
                name = [unique]
                check_unique(df, name)

            unique = input('Enter column name or n to quit:')

    if return_list:
        return outlist
    pass

def check_null(df, columns=None):
    """
    Iterates through columns and checks for null values and displays # and % of column.
    Params:
    ******************
    df: pandas DataFrame

    columns: list of columns to check
    **********>
    Returns: list of column names if return_list=True
    """
    from bs_ds.bamboo import list2df
    display_list = [['Column', 'Null values', 'Total Values','Percent']]
    outlist = []
    print(f'\n----------------------------\n')
    print(f'# of Identified Null Values:')

    # Check for user column list
    columns_to_check = []
    if columns==None:
        columns_to_check = df.columns
    else:
        columns_to_check = columns
    # Iterate through columns

    for col in columns_to_check:

        # Check for object dtype,
        # if df[col].dtype == 'object':

        # If object, check for numeric


        # If numeric, get counts
        vals = df[col].isna().sum()
        percent = round((vals/len(df[col]))*100, 3)
        display_list.append([col, vals, len(df[col]), percent])
        outlist.append(col)

    list2show=list2df(display_list)
    list2show.set_index('Column',inplace=True)
    display(list2show)


class LabelLibrary():
    """A Multi-column version of sklearn LabelEncoder, which fits a LabelEncoder
   to each column of a df and stores it in the index dictionary where
   .index[keyword=colname] returns the fit encoder object for that column.

   Example:
   lib =LabelLibrary()

   # Be default, lib will fit all columns.
   lib.fit(df)
   # Can also specify columns
   lib.fit(df,columns=['A','B'])

   # Can then transform
   df_coded = lib.transform(df,['A','B'])
   # Can also use fit_transform
   df_coded = lib.fit_transform(df,columns=['A','B'])

   # lib.index contains each col's encoder by col name:
   col_a_classes = lib.index('A').classes_

   """

    def __init__(self):#,df,features):
        """creates self.index and self.encoder"""
        self.index = {}
        from sklearn.preprocessing import LabelEncoder as encoder
        self.encoder=encoder
        # self. = df
        # self.features = features



    def fit(self,df,columns=None):
        """ Creates an encoder object and fits to each columns.
        Fit encoder is saved in the index dictionary by key=column_name"""
        if columns==None:
            columns = df.columns
#             if any(df.isna()) == True:
#                 num_null = sum(df.isna().sum())
#                 print(f'Replacing {num_null}# of null values with "NaN".')
#                 df.fillna('NaN',inplace=True)


        for col in columns:

            if any(df[col].isna()):
                num_null = df[col].isna().sum()
                Warning(f'For {col}: Replacing {num_null} null values with "NaN".')
                df[col].fillna('NaN',inplace=True)

            # make the encoder
            col_encoder = self.encoder()

            #fit with label encoder
            self.index[col] = col_encoder.fit(df[col])


    def transform(self,df, columns=None):
        df_coded = pd.DataFrame()

        if columns==None:
            df_columns=df.columns
            columns = df_columns
        else:
            df_columns = df.columns


        for dfcol in df_columns:
            if dfcol in columns:
                fit_enc = self.index[dfcol]
                df_coded[dfcol] = fit_enc.transform(df[dfcol])
            else:
                df_coded[dfcol] = df[dfcol]
        return df_coded

    def fit_transform(self,df,columns=None):
        self.fit(df,columns)
        df_coded = self.transform(df,columns)
        return df_coded

    def inverse_transform(self,df,columns = None):

        df_reverted = pd.DataFrame()

        if columns==None:
            columns=df.columns

        for col in columns:
            fit_enc = self.index[col]
            df_reverted[col] = fit_enc.inverse_transform(df[col])
        return df_reverted





def compare_duplicates(df1, df2, to_drop=True, verbose=True, return_names_list=False):
    """
    Compare two dfs for duplicate columns, drop if to_drop=True, useful
    to us before concatenating when dtypes are different between matching column names
    and df.drop_duplicates is not an option.
    Params:
    --------------------
    df1, df2 : pandas dataframe suspected of having matching columns
    to_drop : bool, (default=True)
        If True will give the option of dropping columns one at a time from either column.
    verbose: bool (default=True)
        If True prints column names and types, set to false and return_names list=True
        if only desire a list of column names and no interactive interface.
    return_names_list: bool (default=False),
        If True, will return a list of all duplicate column names.
    --------------------
    Returns: List of column names if return_names_list=True, else nothing.
    """
    catch = []
    dropped1 = []
    dropped2 = []
    if verbose:
        print("Column |   df1   |   df2   ")
        print("*----------------------*")

    # Loop through columns, inspect for duplicates
    for col in df1.columns:
        if col in df2.columns:
            catch.append(col)

            if verbose:
                print(f"{col}   {df1[col].dtype}   {df2[col].dtype}")

            # Accept user input and drop columns one by one
            if to_drop:
                choice = input("\nDrop this column? Enter 1. df1, 2. df2 or n for neither")

                if choice ==  "1":
                    df1.drop(columns=col, axis=1, inplace=True)
                    dropped1.append(col)

                elif choice == "2":
                    df2.drop(columns=col, axis=1, inplace=True)
                    dropped2.append(col)
                else:

                    continue
    # Display dropped columns and orignating df
    if to_drop:
        if len(dropped1) >= 1:
            print(f"\nDropped from df1:\n{dropped1}")
        if len(dropped2) >= 1:
            print(f"\nDropped from df1:\n{dropped2}")

    if return_names_list:
        return catch
    else:
        pass


## Dataframes styling
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
        series=panda_obj
        print(f'\n----------------------------\n')
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
            print("\n-----------------------------------------------")
            print(f"Column: df['{col_name}']':")
            print(f"dtype: {col.dtypes}")
            print(f"isna: {col.isna().sum()} out of {len(col)} - {round(col.isna().sum()/len(col)*100,3)}%")

            print(f'\nUnique non-na values:\nnlargest={nlargest}\n-----------------')
            if nlargest =='all':
                print(col.value_counts())
            else:
                print(col.value_counts().nlargest(nlargest))


def hover(hover_color="gold"):
    """DataFrame Styler: Called by highlight to highlight row below cursor.
        Changes html background color.

        Parameters:

        hover_Color
    """
    from IPython.display import HTML
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])


def highlight(df,hover_color="gold"):
    """DataFrame Styler:
        Highlight row when hovering.
        Accept and valid CSS colorname as hover_color.
    """
    styles = [
        hover(hover_color),
        dict(selector="th", props=[("font-size", "115%"),
                                   ("text-align", "center")]),
        dict(selector="caption", props=[("caption-side", "bottom")])
    ]
    html = (df.style.set_table_styles(styles)
              .set_caption("Hover to highlight."))
    return html


def color_true_green(val):
    """DataFrame Styler:
    Changes text color to green if value is True
    Ex: style_df = df.style.applymap(color_true_green)
        style_df #to display"""
    color='green' if val==True else 'black'
    return f'color: {color}'

# Style dataframe for easy visualization


def color_scale_columns(df,matplotlib_cmap = "Greens",subset=None,):
    """DataFrame Styler:
    Takes a df, any valid matplotlib colormap column names
    (matplotlib.org/tutorials/colors/colormaps.html) and
    returns a dataframe with a gradient colormap applied to column values.

    Example:
    df_styled = color_scale_columns(df,cmap = "YlGn",subset=['Columns','to','color'])

    Parameters:
    -----------
        df:
            DataFrame containing columns to style.
    subset:
         Names of columns to color-code.
    cmap:
        Any matplotlib colormap.
        https://matplotlib.org/tutorials/colors/colormaps.html

    Returns:
    ----------
        df_style:
            styled dataframe.

    """
    from IPython.display import display
    import seaborn as sns
    cm = matplotlib_cmap
    #     cm = sns.light_palette("green", as_cmap=True)
    df_style = df.style.background_gradient(cmap=cm,subset=subset)#,low=results.min(),high=results.max())
    # Display styled dataframe
#     display(df_style)
    return df_style


    ## DataFrame Creation, Inspection, and Exporting
def inspect_df(df, n_rows=3, verbose=True):
    """ EDA:
    Show all pandas inspection tables.
    Displays df.head(), df.info(), df.describe().
    By default also runs check_null and check_numeric to inspect
    columns for null values and to check string columns to detect
    numeric values. (If verbose==True)
    Parameters:
        df(dataframe):
            dataframe to inspect
        n_rows:
            number of header rows to show (Default=3).
        verbose:
            If verbose==True (default), check_null and check_numeric.
    Ex: inspect_df(df,n_rows=4)
    """
    from bs_ds.bamboo import check_column, check_null, check_numeric, check_unique
    with pd.option_context("display.max_columns", None ,'display.precision',4):
        display(df.info()), display(df.describe())

        if verbose == True:
            check_numeric(df,unique_check=False)
            check_null(df)

        display(df.head(n_rows))





def drop_cols(df, list_of_strings_or_regexp):#,axis=1):
    """EDA: Take a df, a list of strings or regular expression and recursively
    removes all matching column names containing those strings or expressions.
    # Example: if the df_in columns are ['price','sqft','sqft_living','sqft15','sqft_living15','floors','bedrooms']
    df_out = drop_cols(df_in, ['sqft','bedroom'])
    df_out.columns # will output: ['price','floors']

    Parameters:
        DF --
            Input dataframe to remove columns from.
        regex_list --
            list of string patterns or regexp to remove.

    Returns:
        df_dropped -- input df without the dropped columns.
    """
    regex_list=list_of_strings_or_regexp
    df_cut = df.copy()
    for r in regex_list:
        df_cut = df_cut[df_cut.columns.drop(list(df_cut.filter(regex=r)))]
        print(f'Removed {r}.')
    df_dropped = df_cut
    return df_dropped

CSS = """
table.dataframe td, table.dataframe th { /* This is for the borders for columns)*/
    border: 2px solid black
    border-collapse:collapse;
    text-align:center;
}
table.dataframe th {
    /*padding:1em 1em;*/
    background-color: #000000;
    color: #ffffff;
    text-align: center;
    font-weight: bold;
    font-size: 12pt
    font-weight: bold;
    padding: 0.5em 0.5em;
}
table.dataframe td:not(:th){
    /*border: 1px solid ##e8e8ea;*/
    /*background-color: ##e8e8ea;*/
    background-color: gainsboro;
    text-align: center;
    vertical-align: middle;
    font-size:10pt;
    padding: 0.7em 1em;
    /*padding: 0.1em 0.1em;*/
}
table.dataframe tr:not(:last-child) {
    border-bottom: 1px solid gainsboro;
}
table.dataframe {
    /*border-collapse: collapse;*/
    background-color: gainsboro; /* This is alternate rows*/
    text-align: center;
    border: 2px solid black;
}
table.dataframe th:not(:empty), table.dataframe td{
    border-right: 1px solid white;
    text-align: center;
}
"""
# HTML('<style>.output {flex-direction: row;}</style>')
# HTML(f"<style>{CSS}</style>")
# def html_off():
#     HTML('<style></style>')
# def html_on(CSS):
#     HTML('<style>%s</style>'.format(CSS))

##
# EDA / Plotting Functions
def multiplot(df,annot=True,fig_size=None):
    """EDA: Plots results from df.corr() in a correlation heat map for multicollinearity.
    Returns fig, ax objects"""
    import seaborn as sns
    sns.set(style="white")
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    if fig_size==None:
        figsize=(16,16)
    else:
        figsize = fig_size

    f, ax = plt.subplots(figsize=(figsize))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=annot, cmap=cmap, center=0,

    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return f, ax

# Plots histogram and scatter (vs price) side by side
def plot_hist_scat(df, target=None, figsize=(12,9),fig_style='dark_background',font_dict=None,plot_kwds=None):
    """EDA: Great summary plots of all columns of a df vs target columne.
    Shows distplots and regplots for columns im datamframe vs target.
    Parameters:
        df (DataFrame):
            DataFrame.describe() columns will be plotted.
        target (string):
            Name of column containing target variable.assume first column.
        figsize (tuple):
            Tuple for figsize. Default=(12,9).
        fig_style:
            Figure style to use (in this context, will not change others in notebook).
            Default is 'dark_background'.
        font_dict:
            A keywork dictionry containing values for font properties under the following keys:
            - "fontTitle": font dictioanry for titles
            , fontAxis, fontTicks

    **plot_kwds:
        A kew_word dictionary containing any of the following keys for dictionaries containing
        any valid matplotlib key:value pairs for plotting:
            "hist_kws, kde_kws, line_kws,scatter_kws"
        Accepts any valid matplotlib key:value pairs passed by searborn to matplotlib.
        Subplot 1: hist_kws, kde_kws
        Subplot 2: line_kws,scatter_kws

    Returns:
        fig:
            Figure object.
        ax:
            Subplot axes with format ax[row,col].
            Subplot 1 = ax[0,0]; Subplot 2 = ax[0,1]
   """
    import matplotlib.ticker as mtick
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set target as first column if not specified
    if target==None:
        target= df.iloc[:,0]

    ###  DEFINE AESTHETIC CUSTOMIZATIONS  -------------------------------##
    # Checking for user font_dict, if not setting defaults:
    if font_dict == None:
        # Axis Label fonts
        fontTitle = {'fontsize': 16,
                   'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontAxis = {'fontsize': 14,
                   'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontTicks = {'fontsize': 12,
                   'fontweight':'bold',
                    'fontfamily':'serif'}

    else:

        if 'fontTitle' in font_dict.keys():
            fontTitle = font_dict['fontTitle']
        else:
            fontTitle = {'fontsize': 16, 'fontweight': 'bold','fontfamily':'serif'}

        if 'fontAxis' in font_dict.keys():
            fontAxis = font_dict['fontAxis']
        else:
            fontAxis = {'fontsize': 14,'fontweight': 'bold', 'fontfamily':'serif'}

        if 'fontTicks' in font_dict.keys():
            fontTicks = font_dict['fontTicks']
        else:
            fontTicks = {'fontsize': 12,'fontweight':'bold','fontfamily':'serif'}

    # Checking for user plot_kwds
    if plot_kwds == None:
        hist_kws = {"linewidth": 1, "alpha": 1, "color": 'steelblue','edgecolor':'w','hatch':'\\'}
        kde_kws = {"color": "white", "linewidth": 3, "label": "KDE",'alpha':0.7}
        line_kws={"color":"white","alpha":0.5,"lw":3,"ls":":"}
        scatter_kws={'s': 2, 'alpha': 0.8,'marker':'.','color':'steelblue'}

    else:
        kwds = plot_kwds
        # Define graphing keyword dictionaries for distplot (Subplot 1)
        if 'hist_kws' in kwds.keys():
            hist_kws = kwds['hist_kws']
        else:
            hist_kws = {"linewidth": 1, "alpha": 1, "color": 'steelblue','edgecolor':'w','hatch':'\\'}

        if 'kde_kws' in kwds.keys():
            kde_kws = kwds['kde_kws']
        else:
            kde_kws = {"color": "white", "linewidth": 3, "label": "KDE",'alpha':0.7}

        # Define the kwd dictionaries for scatter and regression line (subplot 2)
        if 'line_kws' in kwds.keys():
            line_kws = kwds['line_kws']
        else:
            line_kws={"color":"white","alpha":0.5,"lw":3,"ls":":"}

        if 'scatter_kws' in kwds.keys():
            scatter_kws = kwds['scatter_kws']
        else:
            scatter_kws={'s': 2, 'alpha': 0.8,'marker':'.','color':'steelblue'}


    with plt.style.context(fig_style):
        # Formatting dollar sign labels
        # fmtPrice = '${x:,.0f}'
        # tickPrice = mtick.StrMethodFormatter(fmtPrice)

        ###  PLOTTING ----------------------------- ------------------------ ##

        # Loop through dataframe to plot
        for column in df.describe():

            # Create figure with subplots for current column
            fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=2)

            ##  SUBPLOT 1 --------------------------------------------------##
            i,j = 0,0
            ax[i,j].set_title(column.capitalize(),fontdict=fontTitle)


            # Plot distplot on ax[i,j] using hist_kws and kde_kws
            sns.distplot(df[column], norm_hist=True, kde=True,
                         hist_kws = hist_kws, kde_kws = kde_kws,
                         label=column+' histogram', ax=ax[i,j])


            # Set x axis label
            ax[i,j].set_xlabel(column.title(),fontdict=fontAxis)

            # Get x-ticks, rotate labels, and return
            xticklab1 = ax[i,j].get_xticklabels(which = 'both')
            ax[i,j].set_xticklabels(labels=xticklab1, fontdict=fontTicks, rotation=0)
            ax[i,j].xaxis.set_major_formatter(mtick.ScalarFormatter())


            # Set y-label
            ax[i,j].set_ylabel('Density',fontdict=fontAxis)
            yticklab1=ax[i,j].get_yticklabels(which='both')
            ax[i,j].set_yticklabels(labels=yticklab1,fontdict=fontTicks)
            ax[i,j].yaxis.set_major_formatter(mtick.ScalarFormatter())


            # Set y-grid
            ax[i, j].set_axisbelow(True)
            ax[i, j].grid(axis='y',ls='--')




            ##  SUBPLOT 2-------------------------------------------------- ##
            i,j = 0,1
            ax[i,j].set_title(column.capitalize(),fontdict=fontTitle)



            # Plot regplot on ax[i,j] using line_kws and scatter_kws
            sns.regplot(df[column], df[target],
                        line_kws = line_kws,
                        scatter_kws = scatter_kws,
                        ax=ax[i,j])

            # Set x-axis label
            ax[i,j].set_xlabel(column.title(),fontdict=fontAxis)

             # Get x ticks, rotate labels, and return
            xticklab2=ax[i,j].get_xticklabels(which='both')
            ax[i,j].set_xticklabels(labels=xticklab2,fontdict=fontTicks, rotation=0)
            ax[i,j].xaxis.set_major_formatter(mtick.ScalarFormatter())

            # Set  y-axis label
            ax[i,j].set_ylabel(target.title(),fontdict=fontAxis)

            # Get, set, and format y-axis Price labels
            yticklab = ax[i,j].get_yticklabels()
            ax[i,j].set_yticklabels(yticklab,fontdict=fontTicks)
            ax[i,j].yaxis.set_major_formatter(mtick.ScalarFormatter())

            # Set y-grid
            ax[i, j].set_axisbelow(True)
            ax[i, j].grid(axis='y',ls='--')

            ## ---------- Final layout adjustments ----------- ##
            # Deleted unused subplots
            fig.delaxes(ax[1,1])
            fig.delaxes(ax[1,0])

            # Optimizing spatial layout
            fig.tight_layout()
            # figtitle=column+'_dist_regr_plots.png'
            # plt.savefig(figtitle)
    return fig, ax


## Mike's Plotting Functions
def draw_violinplot(x , y, hue=None, data=None, title=None,
                    ticklabels=None, leg_label=None):

    '''Plots a violin plot with horizontal mean line, inner stick lines
    y must be arraylike in order to plot mean line. x can be label in data'''


    fig,ax = plt.subplots(figsize=(12,10))

    sns.violinplot(x, y, hue=hue,
                   data = data,
                   cut=2,
                   split=True,
                   scale='count',
                   scale_hue=True,
                   saturation=.7,
                   alpha=.9,
                   bw=.25,
                   palette='Dark2',
                   inner='stick'
                  ).set_title(title)

    ax.set(xlabel= x.name.title(),
          ylabel= y.name.title(),
           xticklabels=ticklabels)

    ax.axhline( y.mean(),
               label='Total Mean',
               ls=':',
               alpha=.2,
               color='xkcd:yellow')

    ax.legend().set_title(leg_label)

    plt.show()
    return fig, ax
## Finding outliers and statistics
# Tukey's method using IQR to eliminate
def detect_outliers(df, n, features):
    """Uses Tukey's method to return outer of interquartile ranges to return indices if outliers in a dataframe.
    Parameters:
    df (DataFrame): DataFrame containing columns of features
    n: default is 0, multiple outlier cutoff

    Returns:
    Index of outliers for .loc

    Examples:
    Outliers_to_drop = detect_outliers(data,2,["col1","col2"]) Returning value
    df.loc[Outliers_to_drop] # Show the outliers rows
    data= data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
"""

# Drop outliers

    outlier_indices = []
    # iterate over features(columns)
    for col in features:

        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        from collections import Counter
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers


def find_outliers(column):
    quartile_1, quartile_3 = np.percentile(column, [25, 75])
    IQR = quartile_3 - quartile_1
    low_outlier = quartile_1 - (IQR * 1.5)
    high_outlier = quartile_3 + (IQR * 1.5)
    outlier_index = column[(column < low_outlier) | (column > high_outlier)].index
    return outlier_index

# describe_outliers -- calls find_outliers
def describe_outliers(df):
    """ Returns a new_df of outliers, and % outliers each col using detect_outliers.
    """
    out_count = 0
    new_df = pd.DataFrame(columns=['total_outliers', 'percent_total'])
    for col in df.columns:
        outies = find_outliers(df[col])
        out_count += len(outies)
        new_df.loc[col] = [len(outies), round((len(outies)/len(df.index))*100, 2)]
    new_df.loc['grand_total'] = [sum(new_df['total_outliers']), sum(new_df['percent_total'])]
    return new_df

