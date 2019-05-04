# -*- coding: utf-8 -*-

"""Main module."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as sts
from IPython.display import display
import xgboost
import sklearn
import scipy
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.stats import randint, expon
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xbg
from xgboost import XGBClassifier
import time
import re

## Styled Dataframe
from IPython.display import HTML
pd.set_option('display.precision',3)
pd.set_option('display.html.border',2)
pd.set_option('display.notebook_repr_htm',True)
pd.set_option('display.max_columns',None)

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
import_dict = {'pandas':'pd',
                 'numpy':'np',
                 'matplotlib':'mpl',
                 'matplotlib.pyplot':'plt',
                 'seaborn':'sns',
                 'scip.stats.':'sts'}
            
# index_range = list(range(1,len(import_dict)))
df_imported= pd.DataFrame.from_dict(import_dict,orient='index')
df_imported.columns=['Module/Package Handle']
display(df_imported)
## DataFrame Creation, Inspection, and Exporting
def inspect_df(df,n_rows=3):
    """Displays df.head(),df.info(),df.describe() for dataframe. 
    Ex: inspect_df(df)"""
    pd.set_option('display.precision',3)
    pd.set_option('display.html.border',2)
    pd.set_option('display.notebook_repr_htm',True)
    display(df.head(n_rows))
    display(df.info()), display(df.describe())

def list2df(list):#, sort_values='index'):
    """ Take a list where each row becomes a dataframe row and the row[0] contains the columns names.
    Ex: list_results = [["Test","N","p-val"]] #... (some sort of analysis performed to produce results)
        list_results.append([test_Name,length(data),p])
        list2df(list_results)
    """    
    df_list = pd.DataFrame(list[1:],columns=list[0])        
    return df_list

def drop_cols(df, list_of_strings_or_regexp):#,axis=1):
    """Take a df, a list of strings or regular expression and recursively 
    removes all matching column names containing those strings or expressions.
    # Example: if the df_in columns are ['price','sqft','sqft_living','sqft15','sqft_living15','floors','bedrooms']
    df_out = drop_cols(df_in, ['sqft','bedroom'])
    df_out.columns # will output: ['price','floors']
    
    Parameters:
        DF -- input dataframe to remove columns from.
        regex_list -- list of string patterns or regexp to remove.
    
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

## Dataframes styling

# def highlight (helper: hover)
def hover(hover_color="gold"):
    from IPython.display import HTML
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])
def highlight(df,hover_color="gold"):
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
    """Changes text color to green if value is True
    Ex: style_df = df.style.applymap(color_true_green)
        style_df #to display"""
    color='green' if val==True else 'black'
    return f'color: {color}' 

# Style dataframe for easy visualization
def color_scale_columns(df,matplotlib_cmap = "Greens",subset=None,):
    """
    Takes a df, any valid matplotlib colormap column names(matplotlib.org/tutorials/colors/colormaps.html) 
    and returns a dataframe with a gradient colormap applied to column values.
    Ex. color_scale_columns(df,"YlGn")
    
    Parameters:
    -----------
    df: DataFrame containing columns to style.
    subset: Names of columns to color-code.
    cmap: Any matplotlib colormap. https://matplotlib.org/tutorials/colors/colormaps.html
    Colormap to use instead of default seaborn green.  
    
    Returns:
    ----------
    df_style : df.style

    """ 
    from IPython.display import display  
    import seaborn as sns
    cm = matplotlib_cmap
    #     cm = sns.light_palette("green", as_cmap=True)
    df_style = df.style.background_gradient(cmap=cm,subset=subset)#,low=results.min(),high=results.max())
    # Display styled dataframe
#     display(df_style)
    return df_style

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
    '''Takes a parame_name (str), param_values (list/array), a DecisionTreeObject, and a perform_metric.
    Loops through the param_values and re-fits the model and saves performance metrics. Displays color-mapped dataframe of results and line graph.
    
    Perform_metric can be 'r2_mse' or 'roc_auc'.
    Returns:
    - df of results
    - styled-df'''

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

# EDA / Plotting Functions
def multiplot(df):
    """Plots results from df.corr() in a correlation heat map for multicollinearity.
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
    f, ax = plt.subplots(figsize=(16, 16))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, center=0,
                
    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return f, ax

# Plots histogram and scatter (vs price) side by side
def plot_hist_scat(df, target='index'):
    """Plots seaborne distplots and regplots for columns im datamframe vs target.

    Parameters:
    df (DataFrame): DataFrame.describe() columns will be used. 
    target = name of column containing target variable.assume first coluumn. 
    
    Returns:
    Figures for each column vs target with 2 subplots.
   """
    import matplotlib.ticker as mtick
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    with plt.style.context(('dark_background')):
        ###  DEFINE AESTHETIC CUSTOMIZATIONS  -------------------------------##
        figsize=(14,10)

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

            # Define graphing keyword dictionaries for distplot (Subplot 1)
            hist_kws = {"linewidth": 1, "alpha": 1, "color": 'steelblue','edgecolor':'w','hatch':'\\'}
            kde_kws = {"color": "white", "linewidth": 3, "label": "KDE",'alpha':0.7}

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

            # Define the kwd dictionaries for scatter and regression line (subplot 2)
            line_kws={"color":"white","alpha":0.5,"lw":3,"ls":":"}
            scatter_kws={'s': 2, 'alpha': 0.8,'marker':'.','color':'steelblue'}

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


#### Cohen's d
def Cohen_d(group1, group2):
    '''Compute Cohen's d.
    # group1: Series or NumPy array
    # group2: Series or NumPy array
    # returns a floating point number 
    '''
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d



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
def plot_wide_kde_thin_bars(series1,sname1, series2, sname2):
    '''Plot series1 and series 2 on wide kde plot with small mean+sem bar plot.'''
    
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
        test = ax1.get_xticklabels()
        labels = [x.get_text() for x in test]
        ax1.set_xticklabels([plotS1['label'],plotS2['label']], rotation=45,ha='center')
        

        plt.show()

        return fig,ax


def scale_data(data, method='minmax', log=False):
    
    """Takes df or Series, scales it using desired method and returns scaled df.
    
    Parameters
    -----------
    data : pd.Series or pd.DataFrame
        entire dataframe of series to be scaled
    method : str
        The method for scaling to be implemented(default is 'minmax').
        Other options are 'standard' or 'robust'.
    log : bool, optional
        Takes log of data if set to True(deafault is False).
        
    Returns
    --------
    pd.DataFrame of scaled data.
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    
    scale = np.array(data)
    
    # reshape if needed
    if len(scale.shape) == 1:
        scale = scale.reshape(-1,1)
        
    # takes log if log=True  
    if log == True:
        scale = np.log(scale)
        
        
    # creates chosen scaler instance
    if method == 'robust':
        scaler = RobustScaler()
        
    elif method == 'standard':
        scaler = StandardScaler()
        
    else:
        scaler = MinMaxScaler()   
    scaled = scaler.fit_transform(scale)
    
    
    # reshape and create output DataFrame
    if  scaled.shape[1] > 1:
        df_scaled = pd.DataFrame(scaled, index=data.index, columns=data.columns)
        
    else:
        scaled = np.squeeze(scaled)
        scaled = pd.Series(scaled) 
        df_scaled = pd.DataFrame(scaled, index=data.index)
        
    return df_scaled


## Mike's modeling:
# import xgboost
# import sklearn
# import scipy
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
# from scipy.stats import randint, expon
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.metrics import roc_auc_score
# import xgboost as xbg
# from xgboost import XGBClassifier
# import time
# import re

def select_pca(features, n_components_list):
    
    '''
    Takes features and list of n_components to run PCA on
    
    Params:
    ----------
    features: pd.Dataframe
    n_components: list of ints to pass to PCA n_component parameter
    
    returns:
    ----------
    pd.DataFrame, displays number of components and their respective 
    explained variance ratio
    '''
    
    # from bs_ds import list2df
    from sklearn.decomposition import PCA
    
    # Create list to store results in
    results = [['Model','n_components', 'Explained_Variance_ratio_']]
    
    # Loop through list of components to do PCA on
    for n in n_components_list:
        
        # Creat instance of PCA class
        pca = PCA(n_components=n)
        pca.fit_transform(features)
        
        # Create list of n_component and Explained Variance
        component_variance = ['PCA',n, np.sum(pca.explained_variance_ratio_)]
        
        # Append list results list
        results.append(component_variance)
        
        # Use list2df to display results in DataFrame
    return list2df(results)



def train_test_dict(X, y, test_size=.25, random_state=42):
    
    """
    Splits data into train/test sets and returns diction with each variable its own key and value.
    """

    train_test = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
    train_test['X_train'] = X_train
    train_test['y_train'] = y_train
    train_test['X_test'] = X_test
    train_test['y_test'] = y_test

    return train_test

def make_estimators_dict():
    
    """
    Instantiates models as first step for creating pipelines.
    
    """
    # instantiate classifier objects
    xgb = XGBClassifier()
    svc = SVC()
    lr = LogisticRegression()
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    ab = AdaBoostClassifier()

    estimators = {
        'xgb': xgb,
        'SVC': svc,
        'Logisic Regression': lr,
        'GradientBoosting': gb,
        'Random Forest': rf,
        'Decision Tree': dt,
        'AdaBoost': ab
    }
    return estimators


def make_pipes(estimators_dict, scaler=StandardScaler, n_components='mle', random_state=42):

    """
    Makes pipelines for given models, outputs dictionaries with keys as names and pipeline objects as values.
    
    Parameters:
    ---------------
    estimators: dict,
            dictionary with name (str) as key and estimator objects as values.
    scaler: sklearn.preprocessing instave
    """

    # Create dictionary to store pipelines
    pipe_dict = {}

    # Instantiate piplines for each model
    for k, v in estimators_dict.items():
        pipe = Pipeline([('scaler', scaler()),
                        ('pca', PCA(n_components=n_components,random_state=random_state)),
                        ('clf', v(random_state=random_state))])
        # append to dictionary
        pipe_dict[k] = pipe
    
    return pipe_dict





def fit_pipes(pipes_dict, train_test, predict=True, verbose=True, score='accuracy'):

    """
    Fits piplines to training data, if predict=True, it displays a dataframe of scores.
    score can be either 'accuracy' or 'roc_auc'. rco_auc_score should be used with binary classification.
    
     """

    fit_pipes = {}
    score_display = [['Estimator', f'Test {score}']]
    
    # Assert test/train sets are approriate types
    if type(train_test) == dict:
        X = train_test['X_train']
        y = train_test['y_train']
        X_test = train_test['X_test']
        y_test = train_test['y_test']

    elif type(train_test) == list:
        X = train_test[0]
        y = train_test[1]
        X_test = train_test[2]
        y_test = train_test[3]

    else:
        raise ValueError('train_test must be either list or dictionary')
        
    # Implement timer    
    start = time.time()
    
    if verbose:
        print(f'fitting {len(pipes_dict)} models')

    # Fit pipes, predict if True
    for name, pipe in pipes_dict.items():

        fit_pipe = pipe.fit(X, y)
        fit_pipes['name'] = fit_pipe
        
        # Get accuracy or roc_auc score ,append to display list
        if predict:
            print(f'\nscoring {name} model')

            if score == 'accuracy':
                score_display.append(name, fit_pipe.score(X_test, y_test))

            elif score == 'roc_auc':
                score_display.append(name, roc_auc_score(y_test,fit_pipe.decision_function(X_test)))

            else:
                raise ValueError(f"score expected 'accuracy' of 'roc_auc', was given {score}")
    # End timer
    stop = time.time()
    
    if verbose:
        print(f'\nTime to fit all pipeline:{(stop-start)/60} minutes')

    # display results dataframe if prediction and verbosity
    if predict:
        display(list2df(score_display))
    
    return fit_pipes
config_dict = {
    sklearn.linear_model.LogisticRegressionCV:[{

        }],
        sklearn.linear_model.LogisticRegression:[{
            'clf__penalty':['l1'],
            'clf__C':[0.1, 1, 10, 15 ],
            'clf__tol':[1e-5, 1e-4, 1e-3],
            'clf__solver':['liblinear', 'newton-cg'],
            'clf__n_jobs':[-1]
            }, {
            'clf__penalty':['l2'],
            'clf__C':[0.1, 1, 10, 15 ],
            'clf__tol':[1e-5, 1e-4, 1e-3],
            'clf__solver':['lbfgs', 'sag'],
            'clf__n_jobs':[-1]
            }], 
            sklearn.ensemble.RandomForestClassifier:[{
                'clf__n_estimators':[10, 50, 100], 
                'clf__criterion':['gini', 'entropy'],
                'clf__max_depth':[4, 6, 10], 
                'clf__min_samples_leaf':[0.1, 1, 5, 15],
                'clf__min_samples_split':[0.05 ,0.1, 0.2],
                'clf__n_jobs':[-1]
                }],
                sklearn.svm.SVC:[{
                    'clf__C': [0.1, 1, 10], 
                    'clf__kernel': ['linear']
                    },{
                    'clf__C': [1, 10], 
                    'clf__gamma': [0.001, 0.01], 
                    'clf__kernel': ['rbf']
                    }],
                    sklearn.ensemble.GradientBoostingClassifier:[{
                        'clf__loss':['deviance'], 
                        'clf__learning_rate': [0.1, 0.5, 1.0],
                        'clf__n_estimators': [50, 100, 150]
                        }], 
                        xgboost.sklearn.XGBClassifier:[{
                            'clf__learning_rate':[.001, .01],
                            'clf__n_estimators': [1000,  100],
                            'clf__max_depth': [3, 5]
                            }]
    }
    
random_config_dict = {
    sklearn.ensemble.RandomForestClassifier:{
        'clf__n_estimators': [100 ,500, 1000],
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': randint(1,100),
        'clf__max_features': randint(1,100),
        'clf__min_samples_leaf': randint(1, 100),
        'clf__min_samples_split': randint(2, 10),
        'clf__n_jobs':[-1]
        }, 
        xgboost.sklearn.XGBClassifier:{
            'clf__silent': [False],
            'clf__max_depth': [6, 10, 15, 20],
            'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
            'clf__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'clf__colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'clf__min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
            'clf__gamma': [0, 0.25, 0.5, 1.0],
            'clf__reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
            'clf__n_estimators': [100]
            },
            sklearn.svm.SVC:{
                'clf__C': scipy.stats.expon(scale=100), 
                'clf__gamma': scipy.stats.expon(scale=.1),
                'clf__kernel': ['linear','rbf'], 
                'clf__class_weight':['balanced', None]
                }
    }



def pipe_search(estimator, params, X_train, y_train, X_test, y_test, n_components='mle',
                scaler=StandardScaler(), random_state=42, cv=3, verbose=2, n_jobs=-1):

    """
    Fits pipeline and performs a grid search with cross validation using with given estimator
    and parameters.
    
    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface. 
            Ex. sklearn.svm.SVC
    params: dict, list of dicts,
            Dictionary with parameters names (string) as keys and lists of parameter 
            settings to try as values, or a list of such dictionaries, in which case
            the grids spanned by each dictionary in the list are explored.This enables
            searching over any sequence of parameter settings.
            MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    X_train, y_train, X_test, y_test: 
            training and testing data to fit, test to model
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of 
            possible values instead of scipy.stats distributions. If int, random_state is the 
            seed used by the random number generator; If RandomState instance, random_state 
            is the random number generator; If None, the random number generator is the 
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend 
            context. -1 means using all processors. See Glossary for more details.

    Returns:
    ------------
        dictionary:
                keys are: 'test_score' , 'best_accuracy' (training validation score),
                'best_params', 'best_estimator', 'results'
    """
    # create dictionary to store results.
    results = {}
    # Instantiate pipeline object.
    pipe = Pipeline([('scaler', scaler),
                        ('pca', PCA(n_components=n_components,random_state=random_state)),
                        ('clf', estimator(random_state=random_state))])

    # start timer and fit pipeline.                        
    start = time.time()
    pipe.fit(X_train, y_train)

    # Instantiate and fit gridsearch object.
    grid = GridSearchCV(estimator = pipe,
        param_grid = params,
        scoring = 'accuracy',
        cv = cv, verbose = verbose, n_jobs=n_jobs, return_train_score = True)

    grid.fit(X_train, y_train)

    # Store results in dictionary.
    results['test_score'] = grid.score(X_test, y_test)
    results['best_accuracy'] = grid.best_score_
    results['best_params'] = grid.best_params_
    results['best_estimator'] = grid.best_estimator_
    results['results'] = grid.cv_results_

    # End timer and print results if verbosity higher than 0.
    end = time.time()
    if verbose > 0:
        name = str(estimator).split(".")[-1].split("'")[0]
        print(f'{name} \nBest Score: {grid.best_score_} \nBest Params: {grid.best_params_} ')
        print(f'\nest Estimator: {grid.best_estimator_}')
        print(f'\nTime Elapsed: {((end - start))/60} minutes')
    
    return results


def random_pipe(estimator, params, X_train, y_train, X_test, y_test, n_components='mle',
                scaler=StandardScaler(), n_iter=10, random_state=42, cv=3, verbose=2, n_jobs=-1):

    """
    Fits pipeline and performs a randomized grid search with cross validation.
    
    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface. 
            Ex. sklearn.svm.SVC 
    params: dict, 
            Dictionary with parameters names (string) as keys and distributions or
             lists of parameters to try. Distributions must provide a rvs method for 
             sampling (such as those from scipy.stats.distributions). 
             If a list is given, it is sampled uniformly.
            MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    X_train, y_train, X_test, y_test: 
            training and testing data to fit, test to model
    scaler: sklearn.preprocessing class instance,
            MUST BE IN FORM: StandardScaler(), (default=StandardScaler())
    n_iter: int,
            Number of parameter settings that are sampled. n_iter trades off 
            runtime vs quality of the solution.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of 
            possible values instead of scipy.stats distributions. If int, random_state is the 
            seed used by the random number generator; If RandomState instance, random_state 
            is the random number generator; If None, the random number generator is the 
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend 
            context. -1 means using all processors. See Glossary for more details.
    
     Returns:
    ------------
        dictionary:
                keys are: 'test_score' , 'best_accuracy' (training validation score),
                'best_params', 'best_estimator', 'results'
    
    """
    # Start timer
    start = time.time()

    # Create dictioinary for storing results.
    results = {}
    # Instantiate Pipeline object.
    pipe = Pipeline([('scaler', scaler),
                        ('pca', PCA(n_components=n_components,random_state=random_state)),
                        ('clf', estimator(random_state=random_state))])

    # Fit pipeline to training data.                    
    pipe.fit(X_train, y_train)

    # Instantiate RandomizedSearchCV object.
    grid = RandomizedSearchCV(estimator = pipe,
        param_distributions = params,
        n_iter = n_iter,
        scoring = 'accuracy',
        cv = cv, verbose = verbose, n_jobs=n_jobs, return_train_score = True)

    # Fit gridsearch object to training data.
    grid.fit(X_train, y_train)

    # Store Test scores in results dictionary.
    results['test_score'] = grid.score(X_test, y_test)
    results['best_accuracy'] = grid.best_score_
    results['best_params'] = grid.best_params_
    results['best_estimator'] = grid.best_estimator_
    results['results'] = grid.cv_results_

    # End timer
    end = time.time()

    # print concise results if verbosity greater than 0.
    if verbose > 0:
        name = str(estimator).split(".")[-1].split("'")[0]
        print(f'{name} \nBest Score: {grid.best_score_} \nBest Params: {grid.best_params_} ')
        print(f'\nBest Estimator: {grid.best_estimator_}')
        print(f'\nTime Elapsed: {((end - start))/60} minutes')
    
    return results

def compare_pipes(config_dict, X_train, y_train, X_test, y_test, n_components='mle',
                 search='random',scaler=StandardScaler(), n_iter=10, random_state=42,
                  cv=3, verbose=2, n_jobs=-1):
    """
    Runs any number of estimators through pipeline and gridsearch(exhaustive or radomized) with cross validations, 
    can print dataframe with scores, returns dictionary of all results.

    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface. 
            Ex. sklearn.svm.SVC 
    params: dict, or list of dictionaries if using GridSearchcv, cannot pass lists if search='random
            Dictionary with parameters names (string) as keys and distributions or
             lists of parameters to try. Distributions must provide a rvs method for 
             sampling (such as those from scipy.stats.distributions). 
             If a list is given, it is sampled uniformly.
            MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    X_train, y_train, X_test, y_test: 
            training and testing data to fit, test to model
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    search: str, 'random' or 'grid',
            Type of gridsearch to execute, 'random' = RandomizedSearchCV,
            'grid' = GridSearchCV.
    scaler: sklearn.preprocessing class instance,
            MUST BE IN FORM: StandardScaler(), (default=StandardScaler())
    n_iter: int,
            Number of parameter settings that are sampled. n_iter trades off 
            runtime vs quality of the solution.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of 
            possible values instead of scipy.stats distributions. If int, random_state is the 
            seed used by the random number generator; If RandomState instance, random_state 
            is the random number generator; If None, the random number generator is the 
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend 
            context. -1 means using all processors. See Glossary for more details.

    """

    #Start timer
    begin = time.time()
    # CreateDictionary to store results from each grid search. Create list for displaying results.
    compare_dict = {}
    df_list = [['estimator', 'Test Score', 'Best Accuracy Score']]
    # Loop through dictionary instantiate pipeline and grid search on each estimator.
    for k, v in config_dict.items():

        # perform RandomizedSearchCV.
        if search == 'random':

            # Assert params are in correct form, as to not raise error after running search.
            if type (v) == list:
                raise ValueError("'For random search, params must be dictionary, not list ")
            else:
                results = random_pipe(k, v, X_train, y_train, X_test, y_test, n_components, 
                                    scaler, n_iter, random_state, cv, verbose, n_jobs)
        # Perform GridSearchCV.
        elif search == 'grid':
            results = pipe_search(k, v, X_train, y_train, X_test, y_test, n_components, 
                                        scaler, random_state, cv, verbose, n_jobs )

        # Raise error if grid parameter not specified.
        else:
            raise ValueError(f"search expected 'random' or 'grid' instead got{search}")

        # append results to display list and dictionary.
        name = str(k).split(".")[-1].split("'")[0]
        compare_dict[name] = results
        df_list.append([name, results['test_score'], results['best_accuracy']])

    # Display results if verbosity greater than 0.
    finish = time.time()
    if verbose > 0:
        print(f'\nTotal runtime: {((finish - begin)/60)}')
        display(list2df(df_list))
    
    return compare_dict

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.base import clone
import numpy as np
from scipy import sparse
import time

class MetaClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """
    A model stacking classifier for sklearn classifiers. Uses Sklearn API to fit and predict, 
        can be used with PipeLine and other sklearn estimators. Must be passed primary list of estimator(s)
        and secondary(meta) classifier. Secondary model trains a predicts on primary level estimators.

    Parameters:
    _-_-_-_-_-_-_-_-_-

    classifiers : {array-like} shape = [n_estimators]
        list of instantiated sklearn estimators.
    meta_classifier : instatiated sklearn estimator.
        This is the secondary estimator that makes the final prediction based on predicted values
        of classifiers.
    use_probability : bool, (default=False) If True calling fit will train meta_classifier on the predicted probabilities
        instead of predicted class labels.
    double_down : bool, (default=False) If True, calling fit will train meta_classifier on both the primary 
        classifiers predicted lables and the original dataset. Otherwise meta_classifier will only be 
        trained on primary classifier's predicted labels.
    average_probability : bool, (default = False) If True, calling fit will fit the meta_classifier with averaged 
        the probabalities from primiary predictions.
    clones : bool, (default = True), If True, calling fit will fit deep copies of classifiers and meta classifier 
        leaving the original estimators unmodified. False will fit the passed in classifiers directly.  This param 
        is for use with non-sklearn estimators who cannot are not compatible with being cloned.  This may be unecesary
        but I read enough things about it not working to set it as an option for safe measure. It is best to clone.
    verbose : int, (0-2) Sets verbosity level for output while fitting.
    

    Attributes:
    _-_-_-_-_-_-_-_-

    clfs_ : list, fitted classifers (primary classifiers)
    meta_clf_ : estimator, (secondary classifier)
    meta_features_ : predictions from primary classifiers

    Methods:
    8_-_-_-_-_-_-_-_-
    fit(X, y, sample_weight=None): fit entire ensemble with training data, including fitting meta_classifier with meta_data
            params: (See sklearns fit model for any estimator)
                    X : {array-like}, shape = [n_samples, n_features]
                    y : {array-like}, shape =[n_samples]
                    sample_weight : array-like, shape = [n_samples], optional
    fit_transform(X, y=None, fit_params) : Refer to Sklearn docs
    predict(X) : Predict labels
    get_params(params) : get classifier parameters, refer to sklearn class docs
    set_params(params) : set classifier parameters, mostly used internally, can be used to set parameters, refer to sklearn docs.
    score(X, y, sample_weight=None): Get accuracy score
    predict_meta(X): predict meta_features, primarily used to train meta_classifier, but can be used for base ensemeble performance
    predict_probs(X) : Predict label probabilities for X.

    EXAMPLE******************************************EXAMPLE*******************************************EXAMPLE
    EXAMPLE:          # Instantiate classifier objects for base ensemble

                >>>>  xgb = XGBClassifier()   
                >>>>  svc = svm.SVC()
                >>>>  gbc = GradientBoostingClassifier()
                
                      # Store estimators in list

                >>>>  classifiers = [xgb, svc, gbc]  

                    # Instantiate meta_classifier for making final predictions

                >>>>  meta_classifier = LogisticRegression()

                    # instantiate MetaClassifer object and pass classifiers and meta_classifier
                    # Fit model with training data

                >>>>  clf = Metaclassifier(classifiers=classifiers, meta_classifier=meta_classifier)
                >>>>  clf.fit(X_train, y_train)

                    # Check accuracy scores, predict away...

                >>>>  print(f"MetaClassifier Accuracy Score: {clf.score(X_test, y_test)}")
                >>>>  clf.predict(X)
                ---------------------------------------------------------------------------

                fitting 3 classifiers...
                fitting 1/3 classifers...
                ...
                fitting meta_classifier...

                time elapsed: 6.66 minutes
                MetaClassifier Accuracy Score: 99.9   Get it!
    8***********************************************************************************************>
    """


    def __init__(self, classifiers=None, meta_classifier=None, 
                 use_probability=False, double_down=False, 
                 average_probs=False, clones=True, verbose=2):

        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_probability = use_probability
        self.double_down = double_down
        self.average_probs = average_probs
        self.clones = clones
        self.verbose = verbose
        


    def fit(self, X, y, sample_weight=None):

        """
        Fit base classifiers with data and meta-classifier with predicted data from base classifiers.
        
        Parameters:
        .-.-.-.-.-.-.-.-.-.-.
        X : {array-like}, shape =[n_samples, n_features]
            Training data m number of samples and number of features
        y : {array-like}, shape = [n_samples] or [n_samples, n_outputs]
            Target feature values.
        
        Returns:
        .-.-.-.-.-.-.-.-.-.-.
        
        self : object, 
            Fitted MetaClassifier
            
          """
        start = time.time()

        # Make clones of classifiers and meta classifiers to preserve original 
        if self.clones:
            self.clfs_ = clone(self.classifiers)
            self.meta_clf_ = clone(self.meta_classifier)
        else:
            self.clfs_ = self.classifiers
            self.meta_clf_ = self.meta_classifier
        
        if self.verbose > 0:
            print('Fitting %d classifiers' % (len(self.classifiers)))

        # Count for printing classifier count
        n = 1

        for clf in self.clfs_:

            if self.verbose > 1:
                print(f"Fitting classifier {n}/{len(self.clfs_)}")
                n +=1

            if sample_weight is None:
                clf.fit(X ,y)
            else:
                clf.fit(X, y, sample_weight)

        # Get meta_features to fit MetaClassifer   
        meta_features = self.predict_meta(X)

        if verbose >1:
            print("Fitting meta-classifier to meta_features")

        # Assess if X is sparse or not and stack horizontally
        elif sparse.issparse(X):
            meta_features = sparse.hstack((X, meta_features))
        else:
            meta_features = np.hstack((X, meta_features))
        
        # Set attribute
        self.meta_features_ = meta_features

        # Check for sample_weight and fit MetaClassifer to meta_features
        if sample_weight is None:
            self.meta_clf_.fit(meta_features, y)
        else:
            self.meta_clf_.fit(meta_features, y, sample_weight=sample_weight)

        stop = time.time()

        if verbose > 0:
            print(f"Estimators Fit! Time Elapsed: {(stop-start)/60} minutes")
            print("8****************************************>")

        return self



    def predict_meta(self, X):
        
        """
        Predicts on base estimators to get meta_features for MetaClassifier.
        
        Parameters:
        -.-.-.-.-.-.-.-.-
        X : np.array, shape=[n_samples, n_features]

        Returns:
        -.-.-.-.-.-.-.-.-
        meta_features : np.array, shape=[n_samples, n_classifiers]
            the 'new X' for the MetaClassifier to predict with.
        
        """
        # Check parameters and run approriate prediction
        if self.use_probability:

            probs = np.asarray([clf.predict_probs(X) for clf in self.clfs_])

            if self.average_probs:
                preds = np.average(probs, axis=0)

            else:
                preds = np.concatenate(probs, axis=1)

        else:
            preds = np.column_stack([clf.predict(X) for clf in self.clfs_])
        
        return preds

    def predict_probs(self, X):

        """
        Predict probabilities for X
        
        Parameters:
        -.-.-.-.-.-.-.-.-
        X : np.array, shape=[n_samples, n_features]

        Returns:
        -.-.-.-.-.-.-.-.-
        probabilities : array-like,  shape = [n_samples, n_classes] 

        """

        meta_features = self.predict_meta(X)

        if self.double_down == False:
            return self.meta_clf_.predict_probs(meta_features)
        
        elif sparse.issparse(X):
            return self.meta_clf_.predict_probs(sparse.hstack((X, meta_features)))

        else:
            return self.meta_clf_.predict_probs(np.hstack((X, meta_features)))


    def predict(self, X):

        """
        Predicts target values. 
        
        Parameters:
        -.-.-.-.-.-.-.-.-
        X : np.array, shape=[n_samples, n_features]

        Returns:
        -.-.-.-.-.-.-.-.-
        predicted labels : array-like,  shape = [n_samples] or [n_samples, n_outputs] 

        """

        meta_features = self.predict_meta(X)

        if self.double_down == False:
            return self.meta_clf_.predict(meta_features)

        elif sparse.issparse(X):
            return self.meta_clf_.predict(sparse.hstack((X, meta_features)))

        else:
            return self.meta_clf_.predict(np.hstack((X, meta_features)))       

# from sklearn.utils.estimator_checks import check_estimator         


 
# check_estimator(MetaClassifier())
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from sklearn import tree
import xgboost 


def thick_pipe(features, target, n_components,
               classifiers=[
                   LogisticRegression,
                   svm.SVC, 
                   tree.DecisionTreeClassifier, 
                   RandomForestClassifier,
                   AdaBoostClassifier,
                   GradientBoostingClassifier,
                   xgboost.sklearn.XGBClassifier
               ], test_size=.25, split_rand=None, class_rand=None, verbose=False):
    
    """
    Takes features and target, train/test splits and runs each through pipeline,
    outputs accuracy results models and train/test set in dictionary.
    
    Params:
    ------------
    features: pd.Dataframe, variable features
    target: pd.Series, classes/labels
    n_components: int, number of priniciple components, use select_pca() to determine this number
    classifiers: list, classificaation models put in pipeline
    test_size: float, size of test set for test_train_split (default=.25)
    split_rand: int, random_state parameter for test_train_split (default=None)
    class_rand: int, random_state parameter for classifiers (default=None)
    verbose: bool, will print pipline instances as they are created (default=False)
    
    Returns:
    -----------
    dictionary: keys are abbreviated name of model ('LogReg', 'DecTree', 'RandFor', 'SVC'),
    'X_train', 'X_test', 'y_train', 'y_test'. Values are dictionaries with keys for models:
    'accuracy', 'model'. values are: accuracy score,and the classification model.
     values for train/test splits. """
    
    # from bs_ds import list2df
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
    from sklearn import tree
    import xgboost 
    
    
    results = [['classifier', 'score']]
    class_dict = {}
    
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=test_size,
                                                        random_state=split_rand)
    
    for classifier in classifiers:    
    
        pipe = Pipeline([('pca', PCA(n_components=n_components,random_state=class_rand)),
                         ('clf', classifier(random_state=class_rand))])
        
        if verbose:
            print(f'{classifier}:\n{pipe}')
            
        pipe.fit(X_train, y_train)
        
        if classifier == LogisticRegression:
            name = 'LogReg'
        elif classifier == tree.DecisionTreeClassifier:
            name = 'DecTree'
        elif classifier == RandomForestClassifier:
            name = 'RandFor'
        elif classifier == AdaBoostClassifier:
            name = 'AdaBoost'
        elif classifier == GradientBoostingClassifier:
            name = 'GradBoost'
        elif classifier ==  xgboost.sklearn.XGBClassifier:
            name = 'xgb'
        else:
            name = 'SVC'
        
        accuracy = pipe.score(X_test, y_test)
        results.append([name, accuracy])
        class_dict[name] = {'accuracy': accuracy,'model': pipe}
        
    class_dict['X_train'] = X_train
    class_dict['X_test'] = X_test
    class_dict['y_train'] = y_train
    class_dict['y_test'] = y_test
    
    display(list2df(results))
    
    return class_dict


    def check_df_for_columns(df, columns=None):
        """
        Checks df for presence of columns.

        args:
        8**********>
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




    # HTML(f"<style>{CSS}</style>")