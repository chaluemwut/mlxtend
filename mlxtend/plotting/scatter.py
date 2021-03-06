import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import cycle

def category_scatter(x, y, label_col, data,
            markers='sxo^v',
            colors=('blue', 'green', 'red', 'purple', 'gray', 'cyan'),
            alpha=0.7, markersize=20.0):

    """
    Scatter plot to plot categories in different colors/markerstyles.
    
    Parameters
    ----------
    x : str or int
      DataFrame column name of the x-axis values or
      integer for the numpy ndarray column index.
    
    y : str
      DataFrame column name of the y-axis values or
      integer for the numpy ndarray column index
    
    data : Pandas DataFrame object or NumPy ndarray.
    
    markers : str
      Markers that are cycled through the label category.
    
    colors : tuple 
      Colors that are cycled through the label category.

    alpha : float (default: 0.7)
      Parameter to control the transparency.

    markersize : float (default : 20.0)
      Parameter to control the marker size.
      
    Returns
    ---------
    None
    
    """
    color_gen = cycle(colors)
    marker_gen = cycle(markers)

    if isinstance(data, pd.DataFrame):
        for i in (x, y, label_col):
            assert(isinstance(i, str))
        frame = True
        labels = np.unique(data.loc[:, label_col])
    
    elif isinstance(data, np.ndarray):
        for i in (x, y, label_col):
            assert(isinstance(i, int))
        frame = False
        labels = np.unique(data[:, label_col])
        
    else:
        raise ValueError('df must be pandas.DataFrame or numpy.ndarray object')

    for lab in labels:
        
        if frame:
            x_dat = data.loc[data.loc[:, label_col] == lab, x]
            y_dat = data.loc[data.loc[:, label_col] == lab, y]
        else:
            x_dat = data[data[:, label_col] == lab, x]
            y_dat = data[data[:, label_col] == lab, y]            
        
        plt.scatter(x_dat, 
                y_dat, 
                c=next(color_gen),
                marker=next(marker_gen),
                label=lab,
                alpha=alpha,
                s=markersize)
    return None


