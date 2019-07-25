# -*- coding: utf-8 -*-

""" A collection of modified tools to visualize the inner-workings of model objects, especially Catboot Models."""
# from sklearn.tree import export_graphviz

def make_activations_model(model,idx_layers_to_show=None, verbose=True):
    """Accepts a Keras image convolution model and exports a new model,
    with just the intermediate activations to plot with plot_activations()."""
    import keras
    import matplotlib.pyplot as plt
    from keras import models
    import numpy as np

    # If no image layer index provided, get all Conv2D and MaxPooling2D layers
    if idx_layers_to_show == None:
        layers_to_show = []

        # Check all layers for appropriate types
        for l,layer in enumerate(model.layers):

            check_type = type(layer)
            if check_type in [keras.layers.convolutional.Conv2D, keras.layers.pooling.MaxPooling2D]:
                layers_to_show.append(layer)

        # Create layer_output s
        layer_outputs = [layer.output for layer in layers_to_show]

    else:
        check_dims = np.shape(idx_layers_to_show)

        # Check if 2 index numbers provided
        if check_dims == 2:
            idx_start = idx_layers_to_show[0]
            idx_end = idx_layers_to_show[1]

            layer_outputs = [layer.output for layer in model.layers[idx_start:idx_end]]# exclude the flatten and dense layers

        elif check_dims == 1:

            layer_outputs = [layer.output for layer in model.layers[idx_layers_to_show]]# exclude the flatten and dense layers

    # Now that we have layer_outputs, lets creat ethe activaiton_model
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    if verbose==True:
        print(activation_model.summary())

    return activation_model

def plot_activations(activations_model, img_tensor, n_cols=16,process=True,colormap='viridis'):
    """Accepts an activations_model from make_activations_model. Plots all channels'
    outputs for every image layer in the model."""
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    # Genearate activations from model
    activations = activations_model.predict(img_tensor)

    # Extract layer names for labels
    layer_names = []
#     for layer in model.layers[:8]:
    for layer in activations_model.layers:
        layer_names.append(layer.name)

    # Calculate the number of rows and columns for the figure
    total_features = sum([a.shape[-1] for a in activations]) # shape[-1] ==number of outputs
    n_rows = math.ceil(total_features / n_cols)

    # creat the figure and plots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols,n_rows*1.2) )

    iteration = 0
    for layer_n, layer_activation in enumerate(activations):
        n_channels = layer_activation.shape[-1]

        for ch_idx in range(n_channels):
            row = iteration // n_cols
            column = iteration % n_cols

            ax = axes[row, column]

            channel_image = layer_activation[0,:,:,ch_idx]

            if process==True:
                """create a z-score of the image"""
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()

                channel_image *= 64
                channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            ax.imshow(channel_image, aspect='auto',cmap=colormap)

            # Remove x and y ticks
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            # Add labels for first channel in layer
            if ch_idx == 0:
                ax.set_title(layer_names[layer_n],fontsize=10)
            iteration +=1

    # After all channels in a layer are finished:
    fig.subplots_adjust(hspace=1.25)
    plt.show()

    return fig, axes


def plot_auc_roc_curve(y_test, y_test_pred):
    """ Takes y_test and y_test_pred from a ML model and uses sklearn roc_curve to plot the AUC-ROC curve."""
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    import matplotlib.pyplot as plt
    auc = roc_auc_score(y_test, y_test_pred[:,1])

    FPr, TPr, _  = roc_curve(y_test, y_test_pred[:,1])
    auc()
    plt.plot(FPr, TPr,label=f"AUC for Classifier:\n{round(auc,2)}" )

    plt.plot([0, 1], [0, 1],  lw=2,linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None,
                          print_matrix=True):
    """Check if Normalization Option is Set to True. If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function."""
    import itertools
    import numpy as np
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

def plot_cat_feature_importances(cb_clf):
    """Accepts a fitted CatBoost classifier model and plots the feature importances as a bar chart.
    Returns the results as a Series."""
    # Plotting Feature Importances
    import pandas as pd
    important_feature_names = cb_clf.feature_names_
    important_feature_scores = cb_clf.feature_importances_

    important_features = pd.Series(important_feature_scores, index = important_feature_names)
    important_features.sort_values().plot(kind='barh')
    return important_features


class Clock(object):
    """A clock meant to be used as a timer for functions using local time.
    Clock.tic() starts the timer, .lap() adds the current laps time to clock._list_lap_times, .toc() stops the timer.
    If user initiializes with verbose =0, only start and final end times are displays.
        If verbose=1, print each lap's info at the end of each lap.
        If verbose=2 (default, display instruction line, return datafarme of results.)
    """

    from datetime import datetime
    from pytz import timezone
    from tzlocal import get_localzone
    from bs_ds import list2df
    # from bs_ds import list2df

    def get_time(self,local=True):
        """Returns current time, in local time zone by default (local=True)."""
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        _now_utc_=datetime.now(timezone('UTC'))
        _now_local_=_now_utc_.astimezone(self._timezone_)
        if local==True:
            time_now = _now_local_

            return time_now#_now_local_
        else:
            return _now_utc_


    def __init__(self, display_final_time_as_minutes=True, verbose=2):

        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        self._strformat_ = []
        self._timezone_ = []
        self._timezone_ = get_localzone()
        self._start_time_ = []
        self._lap_label_ = []
        self._lap_end_time_ = []
        self._verbose_ = verbose
        self._lap_duration_ = []
        self._verbose_ = verbose
        self._prior_start_time_ = []
        self._display_as_minutes_ = display_final_time_as_minutes

        strformat = "%m/%d/%y - %I:%M:%S %p"
        self._strformat_ = strformat

    def mark_lap_list(self, label=None):
        """Used internally, appends the current laps' information when called by .lap()
        self._lap_times_list_ = [['Lap #' , 'Start Time','Stop Time', 'Stop Label', 'Duration']]"""
        import bs_ds as bs
#         print(self._prior_start_time_, self._lap_end_time_)

        if label is None:
            label='--'

        duration = self._lap_duration_.total_seconds()
        self._lap_times_list_.append([ self._lap_counter_ , # Lap #
                                      (self._prior_start_time_).strftime(self._strformat_), # This Lap's Start Time
                                      self._lap_end_time_,#.strftime(self._strformat_), # stop clock time
                                      label,#self._lap_label_, # The Label passed with .lap()
                                      f'{duration:.3f} sec']) # the lap duration


    def tic(self, label=None ):
        "Start the timer and display current time, appends label to the _list_lap_times."
        from datetime import datetime
        from pytz import timezone

        self._start_time_ = self.get_time()
        self._start_label_ = label
        self._lap_counter_ = 0
        self._prior_start_time_=self._start_time_
        self._lap_times_list_=[]

        # Initiate lap counter and list
        self._lap_times_list_ = [['Lap #','Start Time','Stop Time', 'Label', 'Duration']]
        self._lap_counter_ = 0
        self._decorate_ = '--- '
        decorate=self._decorate_
        base_msg = f'{decorate}CLOCK STARTED @: {self._start_time_.strftime(self._strformat_):>{25}}'

        if label == None:
            display_msg = base_msg+' '+ decorate
            label='--'
        else:
            spacer = ' '
            display_msg = base_msg+f'{spacer:{10}} Label: {label:{10}} {decorate}'
        if self._verbose_>0:
            print(display_msg)#f'---- Clock started @: {self._start_time_.strftime(self._strformat_):>{25}} {spacer:{10}} label: {label:{20}}  ----')

    def toc(self,label=None, summary=True):
        """Stop the timer and displays results, appends label to final _list_lap_times entry"""
        if label == None:
            label='--'
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone
        from bs_ds import list2df
        if label is None:
            label='--'

        _final_end_time_ = self.get_time()
        _total_time_ = _final_end_time_ - self._start_time_
        _end_label_ = label

        self._lap_counter_+=1
        self._final_end_time_ = _final_end_time_
        self._lap_label_=_end_label_
        self._lap_end_time_ = _final_end_time_.strftime(self._strformat_)
        self._lap_duration_ = _final_end_time_ - self._prior_start_time_
        self._total_time_ = _total_time_

        decorate=self._decorate_
        # Append Summary Line
        if self._display_as_minutes_ == True:
            total_seconds = self._total_time_.total_seconds()
            total_mins = int(total_seconds // 60)
            sec_remain = total_seconds % 60
            total_time_to_display = f'{total_mins} min, {sec_remain:.3f} sec'
        else:

            total_seconds = self._total_time_.total_seconds()
            sec_remain = round(total_seconds % 60,3)

            total_time_to_display = f'{sec_remain} sec'
        self._lap_times_list_.append(['TOTAL',
                                      self._start_time_.strftime(self._strformat_),
                                      self._final_end_time_.strftime(self._strformat_),
                                      label,
                                      total_time_to_display]) #'Total Time: ', total_time_to_display])

        if self._verbose_>0:
            print(f'--- TOTAL DURATION   =  {total_time_to_display:>{15}} {decorate}')

        if summary:
            self.summary()

    def lap(self, label=None):
        """Records time, duration, and label for current lap. Output display varies with clock verbose level.
        Calls .mark_lap_list() to document results in clock._list_lap_ times."""
        from datetime import datetime
        if label is None:
            label='--'
        _end_time_ = self.get_time()

        # Append the lap attribute list and counter
        self._lap_label_ = label
        self._lap_end_time_ = _end_time_.strftime(self._strformat_)
        self._lap_counter_+=1
        self._lap_duration_ = (_end_time_ - self._prior_start_time_)
        # Now update the record
        self.mark_lap_list(label=label)

        # Now set next lap's new _prior_start
        self._prior_start_time_=_end_time_
        spacer = ' '

        if self._verbose_>0:
            print(f'       - Lap # {self._lap_counter_} @:  \
            {self._lap_end_time_:>{25}} {spacer:{5}} Dur: {self._lap_duration_.total_seconds():.3f} sec.\
            {spacer:{5}}Label:  {self._lap_label_:{20}}')

    def summary(self):
        """Display dataframe summary table of Clock laps"""
        from bs_ds import list2df
        import pandas as pd
        from IPython.display import display
        df_lap_times = list2df(self._lap_times_list_)#,index_col='Lap #')
        df_lap_times.drop('Stop Time',axis=1,inplace=True)
        df_lap_times = df_lap_times[['Lap #','Start Time','Duration','Label']]
        dfs = df_lap_times.style.hide_index().set_caption('Summary Table of Clocked Processes').set_properties(subset=['Start Time','Duration'],**{'width':'140px'})
        display(dfs.set_table_styles([dict(selector='table, th', props=[('text-align', 'center')])]))

