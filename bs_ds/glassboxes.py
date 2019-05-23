# -*- coding: utf-8 -*-

""" A collection of modified tools to visualize the inner-workings of model objects, especially Catboot Models."""
""" CURRENTLY ATTEMPTING TO MAKE viz_tree WORK FOR CATBOOST CLASSIFIERS"""
from sklearn.externals.six import StringIO
from IPython.display import Image
# from sklearn.tree import export_graphviz
import pydotplus

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


# ###################################### CODE FROM ELSEWHERE #################
# ATTEMPTING TO MAKE viz_tree worth for catboost
# ## FIRST LINKED FUNCTION:
# # https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

# # Calls check_is_fitted to test if model is fitted.
#     # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html

# def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
#     """Perform is_fitted validation for estimator.
#     Checks if the estimator is fitted by verifying the presence of
#     "all_or_any" of the passed attributes and raises a NotFittedError with the
#     given message.
#     Parameters
#     ----------
#     estimator : estimator instance.
#         estimator instance for which the check is performed.
#     attributes : attribute name(s) given as string or a list/tuple of strings
#         Eg.:
#             ``["coef_", "estimator_", ...], "coef_"``
#     msg : string
#         The default error message is, "This %(name)s instance is not fitted
#         yet. Call 'fit' with appropriate arguments before using this method."
#         For custom messages if "%(name)s" is present in the message string,
#         it is substituted for the estimator name.
#         Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
#     all_or_any : callable, {all, any}, default all
#         Specify whether all or any of the given attributes must exist.
#     Returns
#     -------
#     None
#     Raises
#     ------
#     NotFittedError
#         If the attributes are not found.
#     """
#     if msg is None:
#         msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
#                 "appropriate arguments before using this method.")

#     if not hasattr(estimator, 'fit'):
#         raise TypeError("%s is not an estimator instance." % (estimator))

#     if not isinstance(attributes, (list, tuple)):
#         attributes = [attributes]

#     if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
#         raise NotFittedError(msg % {'name': type(estimator).__name__})

# class NotFittedError(ValueError, AttributeError):
#     """Exception class to raise if estimator is used before fitting.
#     This class inherits from both ValueError and AttributeError to help with
#     exception handling and backward compatibility.
#     Examples
#     --------
#     >>> from sklearn.svm import LinearSVC
#     >>> from sklearn.exceptions import NotFittedError
#     >>> try:
#     ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
#     ... except NotFittedError as e:
#     ...     print(repr(e))
#     ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
#     NotFittedError('This LinearSVC instance is not fitted yet'...)
#     .. versionchanged:: 0.18
#        Moved from sklearn.utils.validation.
#     """


# # Uses _DOTTreeExporter # - from export.py https://github.com/scikit-learn/scikit-learn/blob/612a04e4e44d5fb02661c1e03069218197877f2b/sklearn/tree/export.py
# # class _BaseTreeExporter(object):
# #     def __init__(self, max_depth=None, feature_names=None,
# #                  class_names=None, label='all', filled=False,
# #                  impurity=True, node_ids=False,
# #                  proportion=False, rotate=False, rounded=False,
# #                  precision=3, fontsize=None):
# #         self.max_depth = max_depth
# #         self.feature_names = feature_names
# #         self.class_names = class_names
# #         self.label = label
# #         self.filled = filled
# #         self.impurity = impurity
# #         self.node_ids = node_ids
# #         self.proportion = proportion
# #         self.rotate = rotate
# #         self.rounded = rounded
# #         self.precision = precision
# #         self.fontsize = fontsize

# #     def get_color(self, value):
# #         # Find the appropriate color & intensity for a node
# #         if self.colors['bounds'] is None:
# #             # Classification tree
# #             color = list(self.colors['rgb'][np.argmax(value)])
# #             sorted_values = sorted(value, reverse=True)
# #             if len(sorted_values) == 1:
# #                 alpha = 0
# #             else:
# #                 alpha = ((sorted_values[0] - sorted_values[1])
# #                          / (1 - sorted_values[1]))
# #         else:
# #             # Regression tree or multi-output
# #             color = list(self.colors['rgb'][0])
# #             alpha = ((value - self.colors['bounds'][0]) /
# #                      (self.colors['bounds'][1] - self.colors['bounds'][0]))
# #         # unpack numpy scalars
# #         alpha = float(alpha)
# #         # compute the color as alpha against white
# #         color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
# #         # Return html color code in #RRGGBB format
# #         return '#%2x%2x%2x' % tuple(color)

# #     def get_fill_color(self, tree, node_id):
# #         # Fetch appropriate color for node
# #         if 'rgb' not in self.colors:
# #             # Initialize colors and bounds if required
# #             self.colors['rgb'] = _color_brew(tree.n_classes[0])
# #             if tree.n_outputs != 1:
# #                 # Find max and min impurities for multi-output
# #                 self.colors['bounds'] = (np.min(-tree.impurity),
# #                                          np.max(-tree.impurity))
# #             elif (tree.n_classes[0] == 1 and
# #                   len(np.unique(tree.value)) != 1):
# #                 # Find max and min values in leaf nodes for regression
# #                 self.colors['bounds'] = (np.min(tree.value),
# #                                          np.max(tree.value))
# #         if tree.n_outputs == 1:
# #             node_val = (tree.value[node_id][0, :] /
# #                         tree.weighted_n_node_samples[node_id])
# #             if tree.n_classes[0] == 1:
# #                 # Regression
# #                 node_val = tree.value[node_id][0, :]
# #         else:
# #             # If multi-output color node by impurity
# #             node_val = -tree.impurity[node_id]
# #         return self.get_color(node_val)

# #     def node_to_str(self, tree, node_id, criterion):
# #         # Generate the node content string
# #         if tree.n_outputs == 1:
# #             value = tree.value[node_id][0, :]
# #         else:
# #             value = tree.value[node_id]

# #         # Should labels be shown?
# #         labels = (self.label == 'root' and node_id == 0) or self.label == 'all'

# #         characters = self.characters
# #         node_string = characters[-1]

# #         # Write node ID
# #         if self.node_ids:
# #             if labels:
# #                 node_string += 'node '
# #             node_string += characters[0] + str(node_id) + characters[4]

# #         # Write decision criteria
# #         if tree.children_left[node_id] != _tree.TREE_LEAF:
# #             # Always write node decision criteria, except for leaves
# #             if self.feature_names is not None:
# #                 feature = self.feature_names[tree.feature[node_id]]
# #             else:
# #                 feature = "X%s%s%s" % (characters[1],
# #                                        tree.feature[node_id],
# #                                        characters[2])
# #             node_string += '%s %s %s%s' % (feature,
# #                                            characters[3],
# #                                            round(tree.threshold[node_id],
# #                                                  self.precision),
# #                                            characters[4])

# #         # Write impurity
# #         if self.impurity:
# #             if isinstance(criterion, _criterion.FriedmanMSE):
# #                 criterion = "friedman_mse"
# #             elif not isinstance(criterion, str):
# #                 criterion = "impurity"
# #             if labels:
# #                 node_string += '%s = ' % criterion
# #             node_string += (str(round(tree.impurity[node_id], self.precision))
# #                             + characters[4])

# #         # Write node sample count
# #         if labels:
# #             node_string += 'samples = '
# #         if self.proportion:
# #             percent = (100. * tree.n_node_samples[node_id] /
# #                        float(tree.n_node_samples[0]))
# #             node_string += (str(round(percent, 1)) + '%' +
# #                             characters[4])
# #         else:
# #             node_string += (str(tree.n_node_samples[node_id]) +
# #                             characters[4])

# #         # Write node class distribution / regression value
# #         if self.proportion and tree.n_classes[0] != 1:
# #             # For classification this will show the proportion of samples
# #             value = value / tree.weighted_n_node_samples[node_id]
# #         if labels:
# #             node_string += 'value = '
# #         if tree.n_classes[0] == 1:
# #             # Regression
# #             value_text = np.around(value, self.precision)
# #         elif self.proportion:
# #             # Classification
# #             value_text = np.around(value, self.precision)
# #         elif np.all(np.equal(np.mod(value, 1), 0)):
# #             # Classification without floating-point weights
# #             value_text = value.astype(int)
# #         else:
# #             # Classification with floating-point weights
# #             value_text = np.around(value, self.precision)
# #         # Strip whitespace
# #         value_text = str(value_text.astype('S32')).replace("b'", "'")
# #         value_text = value_text.replace("' '", ", ").replace("'", "")
# #         if tree.n_classes[0] == 1 and tree.n_outputs == 1:
# #             value_text = value_text.replace("[", "").replace("]", "")
# #         value_text = value_text.replace("\n ", characters[4])
# #         node_string += value_text + characters[4]

# #         # Write node majority class
# #         if (self.class_names is not None and
# #                 tree.n_classes[0] != 1 and
# #                 tree.n_outputs == 1):
# #             # Only done for single-output classification trees
# #             if labels:
# #                 node_string += 'class = '
# #             if self.class_names is not True:
# #                 class_name = self.class_names[np.argmax(value)]
# #             else:
# #                 class_name = "y%s%s%s" % (characters[1],
# #                                           np.argmax(value),
# #                                           characters[2])
# #             node_string += class_name

# #         # Clean up any trailing newlines
# #         if node_string.endswith(characters[4]):
# #             node_string = node_string[:-len(characters[4])]

# #         return node_string + characters[5]


# class _DOTTreeExporter(_BaseTreeExporter):
#     def __init__(self, out_file=SENTINEL, max_depth=None,
#                  feature_names=None, class_names=None, label='all',
#                  filled=False, leaves_parallel=False, impurity=True,
#                  node_ids=False, proportion=False, rotate=False, rounded=False,
#                  special_characters=False, precision=3):

#         super().__init__(
#             max_depth=max_depth, feature_names=feature_names,
#             class_names=class_names, label=label, filled=filled,
#             impurity=impurity,
#             node_ids=node_ids, proportion=proportion, rotate=rotate,
#             rounded=rounded,
#             precision=precision)
#         self.leaves_parallel = leaves_parallel
#         self.out_file = out_file
#         self.special_characters = special_characters

#         # PostScript compatibility for special characters
#         if special_characters:
#             self.characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>',
#                                '>', '<']
#         else:
#             self.characters = ['#', '[', ']', '<=', '\\n', '"', '"']

#         # validate
#         if isinstance(precision, Integral):
#             if precision < 0:
#                 raise ValueError("'precision' should be greater or equal to 0."
#                                  " Got {} instead.".format(precision))
#         else:
#             raise ValueError("'precision' should be an integer. Got {}"
#                              " instead.".format(type(precision)))

#         # The depth of each node for plotting with 'leaf' option
#         self.ranks = {'leaves': []}
#         # The colors to render each node with
#         self.colors = {'bounds': None}

#     def export(self, decision_tree):
#         # Check length of feature_names before getting into the tree node
#         # Raise error if length of feature_names does not match
#         # n_features_ in the decision_tree
#         if self.feature_names is not None:
#             if len(self.feature_names) != decision_tree.n_features_:
#                 raise ValueError("Length of feature_names, %d "
#                                  "does not match number of features, %d"
#                                  % (len(self.feature_names),
#                                     decision_tree.n_features_))
#         # each part writes to out_file
#         self.head()
#         # Now recurse the tree and add node & edge attributes
#         if isinstance(decision_tree, _tree.Tree):
#             self.recurse(decision_tree, 0, criterion="impurity")
#         else:
#             self.recurse(decision_tree.tree_, 0,
#                          criterion=decision_tree.criterion)

#         self.tail()

#     def tail(self):
#         # If required, draw leaf nodes at same depth as each other
#         if self.leaves_parallel:
#             for rank in sorted(self.ranks):
#                 self.out_file.write(
#                     "{rank=same ; " +
#                     "; ".join(r for r in self.ranks[rank]) + "} ;\n")
#         self.out_file.write("}")

#     def head(self):
#         self.out_file.write('digraph Tree {\n')

#         # Specify node aesthetics
#         self.out_file.write('node [shape=box')
#         rounded_filled = []
#         if self.filled:
#             rounded_filled.append('filled')
#         if self.rounded:
#             rounded_filled.append('rounded')
#         if len(rounded_filled) > 0:
#             self.out_file.write(
#                 ', style="%s", color="black"'
#                 % ", ".join(rounded_filled))
#         if self.rounded:
#             self.out_file.write(', fontname=helvetica')
#         self.out_file.write('] ;\n')

#         # Specify graph & edge aesthetics
#         if self.leaves_parallel:
#             self.out_file.write(
#                 'graph [ranksep=equally, splines=polyline] ;\n')
#         if self.rounded:
#             self.out_file.write('edge [fontname=helvetica] ;\n')
#         if self.rotate:
#             self.out_file.write('rankdir=LR ;\n')

#     def recurse(self, tree, node_id, criterion, parent=None, depth=0):
#         if node_id == _tree.TREE_LEAF:
#             raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

#         left_child = tree.children_left[node_id]
#         right_child = tree.children_right[node_id]

#         # Add node with description
#         if self.max_depth is None or depth <= self.max_depth:

#             # Collect ranks for 'leaf' option in plot_options
#             if left_child == _tree.TREE_LEAF:
#                 self.ranks['leaves'].append(str(node_id))
#             elif str(depth) not in self.ranks:
#                 self.ranks[str(depth)] = [str(node_id)]
#             else:
#                 self.ranks[str(depth)].append(str(node_id))

#             self.out_file.write(
#                 '%d [label=%s' % (node_id, self.node_to_str(tree, node_id,
#                                                             criterion)))

#             if self.filled:
#                 self.out_file.write(', fillcolor="%s"'
#                                     % self.get_fill_color(tree, node_id))
#             self.out_file.write('] ;\n')

#             if parent is not None:
#                 # Add edge to parent
#                 self.out_file.write('%d -> %d' % (parent, node_id))
#                 if parent == 0:
#                     # Draw True/False labels if parent is root node
#                     angles = np.array([45, -45]) * ((self.rotate - .5) * -2)
#                     self.out_file.write(' [labeldistance=2.5, labelangle=')
#                     if node_id == 1:
#                         self.out_file.write('%d, headlabel="True"]' %
#                                             angles[0])
#                     else:
#                         self.out_file.write('%d, headlabel="False"]' %
#                                             angles[1])
#                 self.out_file.write(' ;\n')

#             if left_child != _tree.TREE_LEAF:
#                 self.recurse(tree, left_child, criterion=criterion,
#                              parent=node_id, depth=depth + 1)
#                 self.recurse(tree, right_child, criterion=criterion,
#                              parent=node_id, depth=depth + 1)

#         else:
#             self.ranks['leaves'].append(str(node_id))

#             self.out_file.write('%d [label="(...)"' % node_id)
#             if self.filled:
#                 # color cropped nodes grey
#                 self.out_file.write(', fillcolor="#C0C0C0"')
#             self.out_file.write('] ;\n' % node_id)

#             if parent is not None:
#                 # Add edge to parent
#                 self.out_file.write('%d -> %d ;\n' % (parent, node_id))


# def export_graphviz(decision_tree, out_file=None, max_depth=None,
#                     feature_names=None, class_names=None, label='all',
#                     filled=False, leaves_parallel=False, impurity=True,
#                     node_ids=False, proportion=False, rotate=False,
#                     rounded=False, special_characters=False, precision=3):
#     """Export a decision tree in DOT format.
#     This function generates a GraphViz representation of the decision tree,
#     which is then written into `out_file`. Once exported, graphical renderings
#     can be generated using, for example::
#         $ dot -Tps tree.dot -o tree.ps      (PostScript format)
#         $ dot -Tpng tree.dot -o tree.png    (PNG format)
#     The sample counts that are shown are weighted with any sample_weights that
#     might be present.
#     Read more in the :ref:`User Guide <tree>`.
#     Parameters
#     ----------
#     decision_tree : decision tree classifier
#         The decision tree to be exported to GraphViz.
#     out_file : file object or string, optional (default=None)
#         Handle or name of the output file. If ``None``, the result is
#         returned as a string.
#         .. versionchanged:: 0.20
#             Default of out_file changed from "tree.dot" to None.
#     max_depth : int, optional (default=None)
#         The maximum depth of the representation. If None, the tree is fully
#         generated.
#     feature_names : list of strings, optional (default=None)
#         Names of each of the features.
#     class_names : list of strings, bool or None, optional (default=None)
#         Names of each of the target classes in ascending numerical order.
#         Only relevant for classification and not supported for multi-output.
#         If ``True``, shows a symbolic representation of the class name.
#     label : {'all', 'root', 'none'}, optional (default='all')
#         Whether to show informative labels for impurity, etc.
#         Options include 'all' to show at every node, 'root' to show only at
#         the top root node, or 'none' to not show at any node.
#     filled : bool, optional (default=False)
#         When set to ``True``, paint nodes to indicate majority class for
#         classification, extremity of values for regression, or purity of node
#         for multi-output.
#     leaves_parallel : bool, optional (default=False)
#         When set to ``True``, draw all leaf nodes at the bottom of the tree.
#     impurity : bool, optional (default=True)
#         When set to ``True``, show the impurity at each node.
#     node_ids : bool, optional (default=False)
#         When set to ``True``, show the ID number on each node.
#     proportion : bool, optional (default=False)
#         When set to ``True``, change the display of 'values' and/or 'samples'
#         to be proportions and percentages respectively.
#     rotate : bool, optional (default=False)
#         When set to ``True``, orient tree left to right rather than top-down.
#     rounded : bool, optional (default=False)
#         When set to ``True``, draw node boxes with rounded corners and use
#         Helvetica fonts instead of Times-Roman.
#     special_characters : bool, optional (default=False)
#         When set to ``False``, ignore special characters for PostScript
#         compatibility.
#     precision : int, optional (default=3)
#         Number of digits of precision for floating point in the values of
#         impurity, threshold and value attributes of each node.
#     Returns
#     -------
#     dot_data : string
#         String representation of the input tree in GraphViz dot format.
#         Only returned if ``out_file`` is None.
#         .. versionadded:: 0.18
#     Examples
#     --------
#     >>> from sklearn.datasets import load_iris
#     >>> from sklearn import tree
#     >>> clf = tree.DecisionTreeClassifier()
#     >>> iris = load_iris()
#     >>> clf = clf.fit(iris.data, iris.target)
#     >>> tree.export_graphviz(clf) # doctest: +ELLIPSIS
#     'digraph Tree {...

#     FROM https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
#     """

#     check_is_fitted(decision_tree, 'tree_')
#     own_file = False
#     return_string = False
#     try:
#         if isinstance(out_file, str):
#             out_file = open(out_file, "w", encoding="utf-8")
#             own_file = True

#         if out_file is None:
#             return_string = True
#             out_file = StringIO()

#         exporter = _DOTTreeExporter(
#             out_file=out_file, max_depth=max_depth,
#             feature_names=feature_names, class_names=class_names, label=label,
#             filled=filled, leaves_parallel=leaves_parallel, impurity=impurity,
#             node_ids=node_ids, proportion=proportion, rotate=rotate,
#             rounded=rounded, special_characters=special_characters,
#             precision=precision)
#         exporter.export(decision_tree)

#         if return_string:
#             return exporter.out_file.getvalue()

#     finally:
#         if own_file:
#             out_file.close()


# ####


# # Pydotplus
# # https://pydotplus.readthedocs.io/reference.html
# def graph_from_dot_data(data):
#     """Load graph as defined by data in DOT format.

#     The data is assumed to be in DOT format. It will
#     be parsed and a Dot class will be returned,
#     representing the graph.
#     """

#     return parser.parse_dot_data(data)

# def parse_dot_data(data):
#     global top_graphs

#     top_graphs = list()

#     if PY3:
#         if isinstance(data, bytes):
#             # this is extremely hackish
#             try:
#                 idx = data.index(b'charset') + 7
#                 while data[idx] in b' \t\n\r=':
#                     idx += 1
#                 fst = idx
#                 while data[idx] not in b' \t\n\r];,':
#                     idx += 1
#                 charset = data[fst:idx].strip(b'"\'').decode('ascii')
#                 data = data.decode(charset)
#             except:
#                 data = data.decode('utf-8')
#     else:
#         if data.startswith(codecs.BOM_UTF8):
#             data = data.decode('utf-8')

#     try:

#         graphparser = graph_definition()

#         if pyparsing_version >= '1.2':
#             graphparser.parseWithTabs()

#         tokens = graphparser.parseString(data)

#         if len(tokens) == 1:
#             return tokens[0]
#         else:
#             return [g for g in tokens]

#     except ParseException:
#         err = sys.exc_info()[1]
#         print(err.line)
#         print(" " * (err.column - 1) + "^")
#         print(err)
#         return None


