# -*- coding: utf-8 -*-
"""A collection of function to change the aesthetics of Pandas DataFrames using CSS, html, and pandas styling."""
# from IPython.display import HTML
# import pandas as pd
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

def make_CSS(show=False):
    CSS="""
        table td{
        text-align: center;
        }
        table th{
        background-color: black;
        color: white;
        font-family:serif;
        font-size:1.2em;
        }
        table td{
        font-size:1.05em;
        font-weight:75;
        }
        table td, th{
        text-align: center;
        }
        table caption{
        text-align: center;
        font-size:1.2em;
        color: black;
        font-weight: bold;
        font-style: italic
        }
    """
    if show==True:
        from pprint import pprint
        pprint(CSS)
    return CSS


# CSS="""
#     .{
#     text-align: center;
#     }
#     th{
#     background-color: black;
#     color: white;
#     font-family:serif;
#     font-size:1.2em;
#     }
#     td{
#     font-size:1.05em;
#     font-weight:75;
#     }
#     td, th{
#     text-align: center;
#     }
#     caption{
#     text-align: center;
#     font-size:1.2em;
#     color: black;
#     font-weight: bold;
#     font-style: italic
#     }
# """
# HTML(f"<style>{CSS}</style>")
# CSS = """
# table.dataframe td, table.dataframe th { /* This is for the borders for columns)*/
#     border: 2px solid black
#     border-collapse:collapse;
#     text-align:center;
# }
# table.dataframe th {
#     /*padding:1em 1em;*/
#     background-color: #000000;
#     color: #ffffff;
#     text-align: center;
#     font-weight: bold;
#     font-size: 12pt
#     font-weight: bold;
#     padding: 0.5em 0.5em;
# }
# table.dataframe td:not(:th){
#     /*border: 1px solid ##e8e8ea;*/
#     /*background-color: ##e8e8ea;*/
#     background-color: gainsboro;
#     text-align: center;
#     vertical-align: middle;
#     font-size:10pt;
#     padding: 0.7em 1em;
#     /*padding: 0.1em 0.1em;*/
# }
# table.dataframe tr:not(:last-child) {
#     border-bottom: 1px solid gainsboro;
# }
# table.dataframe {
#     /*border-collapse: collapse;*/
#     background-color: gainsboro; /* This is alternate rows*/
#     text-align: center;
#     border: 2px solid black;
# }
# table.dataframe th:not(:empty), table.dataframe td{
#     border-right: 1px solid white;
#     text-align: center;
# }
# # """

def html_off():
    from IPython.display import HTML
    return HTML('<style>{}</style>'.format(''))

def html_on(CSS=None, verbose=False):
    """Applies HTML/CSS styling to all dataframes. 'CSS' variable is created by make_CSS() if not supplied.
    Verbose =True will display the default CSS code used. Any valid CSS key: value pair can be passed."""
    from IPython.display import HTML
    if CSS==None:
        CSS = make_CSS()
    if verbose==True:
        from pprint import pprint
        pprint(CSS)

    return HTML(f"<style>{CSS}</style>")#.format(CSS))


def display_side_by_side(*args):
    """Display all input dataframes side by side. Also accept captioned styler df object (df_in = df.style.set_caption('caption')
    Modified from Source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side"""
    from IPython.display import display_html
    import pandas
    html_str=''
    for df in args:
        if type(df) == pandas.io.formats.style.Styler:
            html_str+= '&nbsp;'
            html_str+=df.render()
        else:
            html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
