from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt	
from math import floor, sqrt, ceil


# function from https://github.com/ymlai87416/PythonPlayground/blob/master/Python%20notebook/Machine%20learning%20by%20andrew%20ng/machine-learning-ex3/Assignment3.ipynb
def displayData(X):
    
    m = X.shape[0]
    plt.figure()
    
    if(m == 1):
        tmp = X[0,:].reshape(20,20, order='F')
        plt.imshow(tmp, cmap='gray_r')
    else:
        display_rows = floor(sqrt(m))
        display_cols = ceil(m / display_rows)
        # set up array
        fig, axarr = plt.subplots(nrows=display_rows, ncols=display_cols,
                                  figsize=(10,10))

        # loop over randomly drawn numbers
        for ii in range(display_rows):
            for jj in range(display_cols):
                tmp = X[ii*display_cols+jj,:].reshape(20,20, order='F')
                axarr[ii,jj].imshow(tmp, cmap='gray_r')
                plt.setp(axarr[ii,jj].get_xticklabels(), visible=False)
                plt.setp(axarr[ii,jj].get_yticklabels(), visible=False)

        fig.subplots_adjust(hspace=0, wspace=0)