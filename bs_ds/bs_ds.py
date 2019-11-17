# -*- coding: utf-8 -*-

"""Main Moule of Data Pipelines and Data Transformation functions & classes."""
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns
# import scipy.stats as sts
from IPython.display import display

def list2df(list, index_col=None, set_caption=None, return_df=True,df_kwds=None): #, sort_values='index'):
    
    """ Quick turn an appened list with a header (row[0]) into a pretty dataframe.

        
        Args
            list (list of lists):
            index_col (string): name of column to set as index; None (Default) has integer index.
            set_caption (string):
            show_and_return (bool):
    
    EXAMPLE USE:
    >> list_results = [["Test","N","p-val"]] 
    
    # ... run test and append list of result values ...
    
    >> list_results.append([test_Name,length(data),p])
    
    ## Displays styled dataframe if caption:
    >> df = list2df(list_results, index_col="Test",
                     set_caption="Stat Test for Significance")
    

    """
    from IPython.display import display
    import pandas as pd
    df_list = pd.DataFrame(list[1:],columns=list[0],**df_kwds)
    
        
    if index_col is not None:
        df_list.reset_index(inplace=True)
        df_list.set_index(index_col, inplace=True)
        
    if set_caption is not None:
        dfs = df_list.style.set_caption()
        display(dfs)
    return df_list





######
def make_gdrive_file_url(share_url_from_gdrive):
    raise Exception('You want to use "convert_gdrive_url()"')


def convert_gdrive_url(share_url_from_gdrive):
    """accepts gdrive share url with format 'https://drive.google.com/open?id=`
    and returns a pandas-usable link with format ''https://drive.google.com/uc?export=download&id='"""
    import re
    file_id = re.compile(r'id=(.*)')
    fid = file_id.findall(share_url_from_gdrive)
    prepend_url = 'https://drive.google.com/uc?export=download&id='
    output_url = prepend_url + fid[0]
    return output_url


#####
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
        import pandas as pd
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
        import pandas as pd

        df_reverted = pd.DataFrame()

        if columns==None:
            columns=df.columns

        for col in columns:
            fit_enc = self.index[col]
            df_reverted[col] = fit_enc.inverse_transform(df[col])
        return df_reverted


## James' Tree Classifier/Regressor

# def tune_params_trees (helpers: performance_r2_mse, performance_roc_auc)
def performance_r2_mse(y_true, y_pred):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error as mse

    r2 = r2_score(y_true,y_pred)
    MSE = mse(y_true,y_pred)
    return r2, MSE

# def performance_roc_auc(X_test,y_test,dtc,verbose=False):
def performance_roc_auc(y_true,y_pred):
    """Tests the results of an already-fit classifer.
    Takes y_true (test split), and y_pred (model.predict()), returns the AUC for the roc_curve as a %"""
    from sklearn.metrics import roc_curve, auc
    FP_rate, TP_rate, _ = roc_curve(y_true,y_pred)
    roc_auc = auc(FP_rate,TP_rate)
    roc_auc_perc = round(roc_auc*100,3)
    return roc_auc_perc

def tune_params_trees(param_name, param_values, DecisionTreeObject, X,Y,test_size=0.25,perform_metric='r2_mse'):
    '''Test Decision Tree Regressor or Classifier parameter with the values in param_values
     Displays color-coed dataframe of perfomance results and subplot line graphs.
    Parameters:
        parame_name (str)
            name of parameter to test with values param_values
        param_values (list/array),
            list of parameter values to try
        DecisionTreeObject,
            Existing DecisionTreeObject instance.
        perform_metric
            Can either 'r2_mse' or 'roc_auc'.

    Returns:
    - df of results
    - Displays styled-df
    - Displays subplots of performance metrics.
    '''
    from bs_ds import list2df
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(test_size=test_size)

    # Create results depending on performance metric
    if perform_metric=='r2_mse':
        results = [['param_name','param_value','r2_test','MSE_test']]

    elif perform_metric=='roc_auc':
        results =  [['param_name','param_value','roc_auc_test']]
    print(f'Using performance metrics: {perform_metric}')

    # Rename Deicision Tree for looping
    dtr_tune =  DecisionTreeObject

    # Loop through each param_value
    for value in param_values:

        # Set the parameters and fit the model
        dtr_tune.set_params(**{param_name:value})
        dtr_tune.fit(X_train,y_train)

        # Get predicitons and test_performance
        y_preds = dtr_tune.predict(X_test)

        # Perform correct performance metric and append results
        if perform_metric=='r2_mse':

            r2_test, mse_test = performance_r2_mse(y_test,y_preds)
            results.append([param_name,value,r2_test,mse_test])

        elif perform_metric=='roc_auc':

            roc_auc_test = performance_roc_auc(y_test,y_preds)
            results.append([param_name,value,roc_auc_test])


    # Convert results to dataframe, set index
    df_results = list2df(results)
    df_results.set_index('param_value',inplace=True)


    # Plot the values in results
    df_results.plot(subplots=True,sharex=True)

    # Style dataframe for easy visualization
    import seaborn as sns
    cm = sns.light_palette("green", as_cmap=True)
    df_style = df_results.style.background_gradient(cmap=cm,subset=['r2_test','MSE_test'])#,low=results.min(),high=results.max())
    # Display styled dataframe
    from IPython.display import display
    display(df_style)

    return df_results


# Display graphviz tree
def viz_tree(tree_object):
    '''Takes a Sklearn Decision Tree and returns a png image using graph_viz and pydotplus.'''
    # Visualize the decision tree using graph viz library
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus
    dot_data = StringIO()
    export_graphviz(tree_object, out_file=dot_data, filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    tree_viz = Image(graph.create_png())
    return tree_viz



# #### Cohen's d
# def Cohen_d(group1, group2):
#     '''Compute Cohen's d.
#     # taken directly from learn.co lesson.
#     # group1: Series or NumPy array
#     # group2: Series or NumPy array
#     # returns a floating point number
#     '''
#     diff = group1.mean() - group2.mean()

#     n1, n2 = len(group1), len(group2)
#     var1 = group1.var()
#     var2 = group2.var()

#     # Calculate the pooled threshold as shown earlier
#     pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)

#     # Calculate Cohen's d statistic
#     d = diff / np.sqrt(pooled_var)

#     return d



#####
def subplot_imshow(images, num_images,num_rows, num_cols, figsize=(20,15)):
    '''
    Takes image data and plots a figure with subplots for as many images as given.

    Parameters:
    -----------
    images: str, data in form data.images ie. olivetti images
    num_images: int, number of images
    num_rows: int, number of rows to plot.
    num_cols: int, number of columns to plot
    figize: tuple, size of figure default=(20,15)

    returns:  figure with as many subplots as images given
    '''

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    for i in range(num_images):
        ax = fig.add_subplot(num_rows,num_cols, i+1, xticks=[], yticks=[])
        ax.imshow(images[i],cmap=plt.gray)

    plt.show()

    return fig, ax
#####


###########
def plot_wide_kde_thin_mean_sem_bars(series1,sname1, series2, sname2):
    '''EDA / Hypothesis Testing:
    Two subplot EDA figure that plots series1 vs. series 2 against with sns.displot
    Large  wide kde plot with small thing mean +- standard error of the mean (sem)
    Overlapping sem error bars is an excellent visual indicator of significant difference.
    .'''

    ## ADDING add_gridspec usage
    import pandas as pd
    import numpy as np
    from scipy.stats import sem

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as ticker

    import seaborn as sns

    from matplotlib import rcParams
    from matplotlib import rc
    rcParams['font.family'] = 'serif'

    # Plot distributions of discounted vs full price groups
    plt.style.use('default')
    # with plt.style.context(('tableau-colorblind10')):
    with plt.style.context(('seaborn-notebook')):

        ## ----------- DEFINE AESTHETIC CUSTOMIZATIONS ----------- ##
       # Axis Label fonts
        fontSuptitle ={'fontsize': 22,
                   'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontTitle = {'fontsize': 10,
                   'fontweight': 'medium',
                    'fontfamily':'serif'}

        fontAxis = {'fontsize': 10,
                   'fontweight': 'medium',
                    'fontfamily':'serif'}

        fontTicks = {'fontsize': 8,
                   'fontweight':'medium',
                    'fontfamily':'serif'}


        ## --------- CREATE FIG BASED ON GRIDSPEC --------- ##

        plt.suptitle('Quantity of Units Sold', fontdict = fontSuptitle)

        # Create fig object and declare figsize
        fig = plt.figure(constrained_layout=True, figsize=(8,3))


        # Define gridspec to create grid coordinates
        gs = fig.add_gridspec(nrows=1,ncols=10)

        # Assign grid space to ax with add_subplot
        ax0 = fig.add_subplot(gs[0,0:7])
        ax1 = fig.add_subplot(gs[0,7:10])

        #Combine into 1 list
        ax = [ax0,ax1]

        ### ------------------  SUBPLOT 1  ------------------ ###

        ## --------- Defining series1 and 2 for subplot 1------- ##
        ax[0].set_title('Histogram + KDE',fontdict=fontTitle)

        # Group 1: data, label, hist_kws and kde_kws
        plotS1 = {'data': series1, 'label': sname1.title(),

                   'hist_kws' :
                    {'edgecolor': 'black', 'color':'darkgray','alpha': 0.8, 'lw':0.5},

                   'kde_kws':
                    {'color':'gray', 'linestyle': '--', 'linewidth':2,
                     'label':'kde'}}

        # Group 2: data, label, hist_kws and kde_kws
        plotS2 = {'data': series2,
                    'label': sname2.title(),

                    'hist_kws' :
                    {'edgecolor': 'black','color':'green','alpha':0.8 ,'lw':0.5},


                    'kde_kws':
                    {'color':'darkgreen','linestyle':':','linewidth':3,'label':'kde'}}

        # plot group 1
        sns.distplot(plotS1['data'], label=plotS1['label'],

                     hist_kws = plotS1['hist_kws'], kde_kws = plotS1['kde_kws'],

                     ax=ax[0])


        # plot group 2
        sns.distplot(plotS2['data'], label=plotS2['label'],

                     hist_kws=plotS2['hist_kws'], kde_kws = plotS2['kde_kws'],

                     ax=ax[0])


        ax[0].set_xlabel(series1.name, fontdict=fontAxis)
        ax[0].set_ylabel('Kernel Density Estimation',fontdict=fontAxis)

        ax[0].tick_params(axis='both',labelsize=fontTicks['fontsize'])
        ax[0].legend()


        ### ------------------  SUBPLOT 2  ------------------ ###

        # Import scipy for error bars
        from scipy.stats import sem

        # Declare x y group labels(x) and bar heights(y)
        x = [plotS1['label'], plotS2['label']]
        y = [np.mean(plotS1['data']), np.mean(plotS2['data'])]

        yerr = [sem(plotS1['data']), sem(plotS2['data'])]
        err_kws = {'ecolor':'black','capsize':5,'capthick':1,'elinewidth':1}

        # Create the bar plot
        ax[1].bar(x,y,align='center', edgecolor='black', yerr=yerr,error_kw=err_kws,width=0.6)


        # Customize subplot 2
        ax[1].set_title('Average Quantities Sold',fontdict=fontTitle)
        ax[1].set_ylabel('Mean +/- SEM ',fontdict=fontAxis)
        ax[1].set_xlabel('')

        ax[1].tick_params(axis=y,labelsize=fontTicks['fontsize'])
        ax[1].tick_params(axis=x,labelsize=fontTicks['fontsize'])

        ax1=ax[1]
        # test = ax1.get_xticklabels()
        # labels = [x.get_text() for x in test]
        ax1.set_xticklabels([plotS1['label'],plotS2['label']], rotation=45,ha='center')


        plt.show()

        return fig,ax




# HTML(f"<style>{CSS}</style>")
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None,
                          print_matrix=True):
    """Check if Normalization Option is Set to True. If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function."""
    import warnings
    warnings.warn('Future versions will be moving plot_confusion_matrix to bs_ds.glassboxes module.')
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt
    if cmap==None:
        cmap = plt.get_cmap("Blues")

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def column_report(df,index_col='iloc', sort_column='iloc', ascending=True,name_for_notes_col = 'Notes',notes_by_dtype=False,
 decision_map=None, format_dict=None,   as_qgrid=True, qgrid_options=None, qgrid_column_options=None,qgrid_col_defs=None, qgrid_callback=None,
 as_df = False, as_interactive_df=False, show_and_return=True):
    """
    Returns a datafarme summary of the columns, their dtype,  a summary dataframe with the column name, column dtypes, and a `decision_map` dictionary of
    datatype.
    [!] Please note if qgrid does not display properly, enter this into your terminal and restart your temrinal.
        'jupyter nbextension enable --py --sys-prefix qgrid'# required for qgrid
        'jupyter nbextension enable --py --sys-prefix widgetsnbextension' # only required if you have not enabled the ipywidgets nbextension yet
    
    Default qgrid options:
       default_grid_options={
        # SlickGrid options
        'fullWidthRows': True,
        'syncColumnCellResize': True,
        'forceFitColumns': True,
        'defaultColumnWidth': 50,
        'rowHeight': 25,
        'enableColumnReorder': True,
        'enableTextSelectionOnCells': True,
        'editable': True,
        'autoEdit': False,
        'explicitInitialization': True,

        # Qgrid options
        'maxVisibleRows': 30,
        'minVisibleRows': 8,
        'sortable': True,
        'filterable': True,
        'highlightSelectedCell': True,
        'highlightSelectedRow': True
    }
    """
    from ipywidgets import interact
    import pandas as pd
    from IPython.display import display
    import qgrid
    small_col_width = 20

    # default_col_options={'width':20}

    default_column_definitions={'column name':{'width':60}, '.iloc[:,i]':{'width':small_col_width}, 'dtypes':{'width':30}, '# zeros':{'width':small_col_width},
                    '# null':{'width':small_col_width},'% null':{'width':small_col_width}, name_for_notes_col:{'width':100}}

    default_grid_options={
        # SlickGrid options
        'fullWidthRows': True,
        'syncColumnCellResize': True,
        'forceFitColumns': True,
        'defaultColumnWidth': 50,
        'rowHeight': 25,
        'enableColumnReorder': True,
        'enableTextSelectionOnCells': True,
        'editable': True,
        'autoEdit': False,
        'explicitInitialization': True,

        # Qgrid options
        'maxVisibleRows': 30,
        'minVisibleRows': 8,
        'sortable': True,
        'filterable': True,
        'highlightSelectedCell': True,
        'highlightSelectedRow': True
    }

    ## Set the params to defaults, to then be overriden
    column_definitions = default_column_definitions
    grid_options=default_grid_options
    # column_options = default_col_options

    if qgrid_options is not None:
        for k,v in qgrid_options.items():
            grid_options[k]=v

    if qgrid_col_defs is not None:
        for k,v in qgrid_col_defs.items():
            column_definitions[k]=v
    else:
        column_definitions = default_column_definitions


    # format_dict = {'sum':'${0:,.0f}', 'date': '{:%m-%Y}', 'pct_of_total': '{:.2%}'}
    # monthly_sales.style.format(format_dict).hide_index()
    def count_col_zeros(df, columns=None):
        import pandas as pd
        import numpy as np
        # Make a list of keys for every column  (for series index)
        zeros = pd.Series(index=df.columns)
        # use all cols by default
        if columns is None:
            columns=df.columns

        # get sum of zero values for each column
        for col in columns:
            zeros[col] = np.sum( df[col].values == 0)
        return zeros


    ##
    df_report = pd.DataFrame({'.iloc[:,i]': range(len(df.columns)),
                            'column name':df.columns,
                            'dtypes':df.dtypes.astype('str'),
                            '# zeros': count_col_zeros(df),
                            '# null': df.isna().sum(),
                            '% null':df.isna().sum().divide(df.shape[0]).mul(100).round(2)})
    ## Sort by index_col
    if 'iloc' in index_col:
        index_col = '.iloc[:,i]'

    df_report.set_index(index_col ,inplace=True)

    ## Add additonal column with notes
    # decision_map_keys = ['by_name', 'by_dtype','by_iloc']
    if decision_map is None:
        decision_map ={}
        decision_map['by_dtype'] = {'object':'Check if should be one hot coded',
                        'int64':'May be  class object, or count of a ',
                        'bool':'one hot',
                        'float64':'drop and recalculate'}

    if notes_by_dtype:
        df_report[name_for_notes_col] = df_report['dtypes'].map(decision_map['by_dtype'])#column_list
    else:
        df_report[name_for_notes_col] = ''
#     df_report.style.set_caption('DF Columns, Dtypes, and Course of Action')

    ##  Sort column
    if sort_column is None:
        sort_column = '.iloc[:,i]'


    if 'iloc' in sort_column:
        sort_column = '.iloc[:,i]'

    df_report.sort_values(by =sort_column,ascending=ascending, axis=0, inplace=True)

    if as_df:
        if show_and_return:
            display(df_report)
        return df_report

    elif as_qgrid:
        print('[i] qgrid returned. Use gqrid.get_changed_df() to get edited df back.')
        qdf = qgrid.show_grid(df_report,grid_options=grid_options, column_options=qgrid_column_options, column_definitions=column_definitions,row_edit_callback=qgrid_callback  )
        if show_and_return:
            display(qdf)
        return qdf

    elif as_interactive_df:

        @interact(column= df_report.columns,direction={'ascending':True,'descending':False})
        def sort_df(column, direction):
            return df_report.sort_values(by=column,axis=0,ascending=direction)
    else:
        raise Exception('One of the output options must be true: `as_qgrid`,`as_df`,`as_interactive_df`')


#################### GENERAL HELPER FUNCTIONS #####################
def is_var(name):
    x=[]
    try: eval(name)
    except NameError: x = None

    if x is None:
        return False
    else:
        return True



def print_docstring_template(style='google',object_type='function',show_url=False, to_clipboard=False):
    """ Prints out docstring template for that is copy/paste ready.
    May choose 'google' or 'numpy' style docstrings and templates
    are available different types ('class','function','module_function').
    
    Args:
        style (str, optional): Which docstring style to return. Options are 'google' and 'numpy'. Defaults to 'google'.
        object_type (str, optional): Which type of template to return. Options are 'class','function','module_function'. Defaults to 'function'.
        show_url (bool, optional): Whether to display link to reference page for style-type. Defaults to False.
    
    Returns:
        [type]: [description]
    """
    template_dict ={}
    template_dict['numpy']={}
    template_dict['numpy']['url']='https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy'
    template_dict['numpy']['function'] = '''
    def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    '''
    template_dict['numpy']['module_function'] = '''
    def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Parameters`` section.
    The name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    If \*args or \*\*kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name : type
            description

            The description may span multiple lines. Following lines
            should be indented to match the first line of the description.
            The ": type" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : :obj:`str`, optional
        The second parameter.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    bool
        True if successful, False otherwise.

        The return type is not optional. The ``Returns`` section may span
        multiple lines and paragraphs. Following lines should be indented to
        match the first line of the description.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    """'''
    
    template_dict['numpy']['class'] = '''
    class ExampleClass(object):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes
    ----------
    attr1 : str
        Description of `attr1`.
    attr2 : :obj:`int`, optional
        Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1 : str
            Description of `param1`.
        param2 : :obj:`list` of :obj:`str`
            Description of `param2`. Multiple
            lines are supported.
        param3 : :obj:`int`, optional
            Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list of str: Doc comment *before* attribute, with type specified
        self.attr4 = ["attr4"]

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

        @property
        def readonly_property(self):
            """str: Properties should be documented in their getter method."""
            return "readonly_property"

        @property
        def readwrite_property(self):
            """:obj:`list` of :obj:`str`: Properties with both a getter and setter
            should only be documented in their getter method.

            If the setter method contains notable behavior, it should be
            mentioned here.
            """
            return ["readwrite_property"]

        @readwrite_property.setter
        def readwrite_property(self, value):
            value

        def example_method(self, param1, param2):
            """Class methods are similar to regular functions.

            Note
            ----
            Do not include the `self` parameter in the ``Parameters`` section.

            Parameters
            ----------
            param1
                The first parameter.
            param2
                The second parameter.

            Returns
            -------
            bool
                True if successful, False otherwise.

            """
            return True

        def __special__(self):
            """By default special members with docstrings are not included.

            Special members are any methods or attributes that start with and
            end with a double underscore. Any special member with a docstring
            will be included in the output, if
            ``napoleon_include_special_with_doc`` is set to True.

            This behavior can be enabled by changing the following setting in
            Sphinx's conf.py::

                napoleon_include_special_with_doc = True

            """
            pass

        def __special_without_docstring__(self):
            pass

        def _private(self):
            """By default private members are not included.

            Private members are any methods or attributes that start with an
            underscore and are *not* special. By default they are not included
            in the output.

            This behavior can be changed such that private members *are* included
            by changing the following setting in Sphinx's conf.py::

                napoleon_include_private_with_doc = True

            """
            pass

        def _private_without_docstring(self):
            pass
        '''
            
       
    template_dict ={}
    template_dict['google']={}
    template_dict['google']['url']="https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google"
    template_dict['google']['function'] = '''
    Example function with types documented in the docstring.

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    '''

    template_dict['google']['module_function'] = '''
    def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    If \*args or \*\*kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name (type): description
            The description may span multiple lines. Following
            lines should be indented. The "(type)" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    if param1 == param2:
        raise ValueError('param1 may not be equal to param2')
    return True
    '''
    
    
    template_dict['google']['class'] = '''
    class ExampleClass(object):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list of str: Doc comment *before* attribute, with type specified
        self.attr4 = ['attr4']

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def readonly_property(self):
        """str: Properties should be documented in their getter method."""
        return 'readonly_property'

    @property
    def readwrite_property(self):
        """:obj:`list` of :obj:`str`: Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return ['readwrite_property']

    @readwrite_property.setter
    def readwrite_property(self, value):
        value

    def example_method(self, param1, param2):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        return True

    def __special__(self):
        """By default special members with docstrings are not included.

        Special members are any methods or attributes that start with and
        end with a double underscore. Any special member with a docstring
        will be included in the output, if
        ``napoleon_include_special_with_doc`` is set to True.

        This behavior can be enabled by changing the following setting in
        Sphinx's conf.py::

            napoleon_include_special_with_doc = True

        """
        pass

    def __special_without_docstring__(self):
        pass

    def _private(self):
        """By default private members are not included.

        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.

        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::

            napoleon_include_private_with_doc = True

        """
        pass

    def _private_without_docstring(self):
        pass
    '''
    
    
    ### Select output
    style_dict = template_dict[style]
    print_template = style_dict[object_type]
    url = style_dict['url']
    
    if show_url:
        print(f'Template source for {style} style docstrings: {url} ')

    if to_clipboard==False:        
        print(print_template)
    else:
        import pyperclip
        print('Template copied to clipboard.')
        return pyperclip.copy(print_template)
        
    
