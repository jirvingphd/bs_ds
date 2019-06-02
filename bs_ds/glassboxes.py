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
    """A clock meant to be used as a timer for functions, also displays local clock time.
    Call Clock.tic() to start a timer. Call Clock.toc to end the timer and display time elapsed."""
    from datetime import datetime
    from pytz import timezone
    from tzlocal import get_localzone

    def __init__(self, verbose=2):

        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        _now_utc_ = []
        _now_local_= []
        _now_utc_ = datetime.now(timezone('UTC'))
        _now_local_ = _now_utc_.astimezone(get_localzone())

        strformat = "%m/%d/%y - %I:%M:%S %p"
        if verbose > 0:
            print(f'Clock created at {_now_local_.strftime(strformat)}.')

        if verbose >1:
            print(f'\tClock.tic() to start.\n\tClock.toc() to stop')


        self._start_utc_= []
        self._start_local_ = []

        self._end_utc_ = []
        self._end_local_ = []

        self._timezone_ = []
        self._timezone_ = get_localzone()

        self._label_ = []
        self._verbose_ = verbose

    def tic(self):

        from datetime import datetime
        from pytz import timezone
        _start_utc_ = datetime.now(timezone('UTC'))
        _start_local_= _start_utc_.astimezone(self._timezone_)

        self._start_utc_ = _start_utc_
        self._start_local_=_start_local_

    def toc(self):

        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        _end_utc_=datetime.now(timezone('UTC'))
        _end_local_=_end_utc_.astimezone(self._timezone_)

        _elapsed_ = _end_local_ - self._start_local_

        print(f'\nTime elapsed: {_elapsed_}.')

        self._end_utc_=_end_utc_
        self._end_local_=_end_local_
        self._elapsed_ = _elapsed_
