#!/usr/bin/env python
import numpy
import matplotlib.pyplot as plt
from ss_plotting import plot_utils
from matplotlib import cm as cm
import ss_plotting.colors as colors

def plot_hexbin(series,
         series_labels = None,
         plot_xlabel = None, 
         plot_xlim = None,
         plot_ylabel = None, 
         plot_ylim = None,
         plot_title = None,
         fontsize=8,
         vmax = None,
         vmin = None,
         linewidths = None,
         savefile = None,
         savefile_size = (3.4, 1.5),
         show_plot = True,
         gridsize = 20,
         series_C = None,
         reduce_C_function = numpy.mean,
         color_map = cm.jet,
         x_scale = 'linear',
         y_scale = 'linear',
         mincnt = None,
         series_edgecolors = None,
         marginals = False):
    """
    Plot yvals as a function of xvals
    @param series List of (xvals, yvals) for each series- each of these will be plotted in a different color
    @param plot_xlabel The label for the x-axis - if None no label is printed
    @param plot_xlim The limits for the x-axis - if None these limits are selected automatically
    @param plot_ylabel The label for the y-axis - if None no label is printed
    @param plot_ylim The limits for the y-axis - if None these limits are selected automatically
    @param plot_title A title for the plot - if None no title is printed
    @param fontsize The size of font for all labels
    @param linewidths The width of the lines in the plot
    @param savefile The path to save the plot to, if None plot is not saved
    @param savefile_size The size of the saved plot
    @param show_plot If True, display the plot on the screen via a call to plt.show()
    
    @param x_scale set to log or linear for the x axis
    @param y_scale set to log or linear for the y axis
    @param cmap the color of hexbin
    @param series_C In hexbin, 'C' is optional--it maps values to x-y coordinates; 
                    if 'C' is None (default) then the result is a pure 2D histogram
    @param reduce_C_function    
        Could be numpy.mean, .sum, .count_nonzero, .max
        Function of one argument that reduces all the values in a bin to a single number
    @param mincnt If not None, only display cells with more than mincnt number of points in the cell
    @param gridsize how many hexagons in one line
    @return fig, ax The figure and axis the plot was created in
    """

    # Validate
    if series is None or len(series) == 0:
        raise ValueError('No data series')
    num_series = len(series)

    if series_labels is None:
        series_labels = [None for s in series]
    if len(series_labels) != num_series:
        raise ValueError('You must define a label for every series')

    fig, ax = plt.subplots()
    plot_utils.configure_fonts(fontsize=fontsize)

    xvals = series[0]
    yvals = numpy.array(series[1])
    if series_C == None:
        c = None
    else:
        c = series_C
    if series_edgecolors == None:
        edgecolors = None
    else:
        edgecolors = colors.get_plot_color(series_edgecolors)


    #http://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set/2371812#2371812
    # http://matplotlib.org/examples/color/colormaps_reference.html
    # if 'bins=None', then color of each hexagon corresponds directly to its count
    # 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then
    # the result is a pure 2D histogram
    # hb = plt.hexbin(x, y, C=None, gridsize=gridsize, marginals=True, cmap=cm.jet, bins=None)
    # hb = plt.hexbin(x, y, C=None, marginals=True, cmap=plt.get_cmap('YlOrBr'), bins=None)
    # r = ax.hexbin(xvals, yvals, C=None, marginals=True, cmap=plt.get_cmap('YlOrBr'), bins=None)


    if plot_xlim is None or plot_ylim is None:
        plot_xlim = [min(xvals),max(xvals)]
        plot_ylim = [min(yvals),max(yvals)]
    if c == None:
        im = ax.hexbin(xvals, yvals, 
            C=None,
            cmap=color_map,
            bins=None,
            gridsize = gridsize,
            xscale = 'linear', 
            yscale = 'linear',
            norm=None, 
            vmin=vmin, 
            vmax=vmax,
            alpha=None, 
            linewidths=linewidths, 
            edgecolors=edgecolors,
            extent=[plot_xlim[0], plot_xlim[1], plot_ylim[0], plot_ylim[1]],
            mincnt=None, 
            marginals=marginals)
    else:
        im = ax.hexbin(xvals, yvals, 
            C=c,
            cmap=color_map,
            bins=None,
            gridsize = gridsize,
            xscale = 'linear', 
            yscale = 'linear',
            norm=None, 
            vmin=vmin, 
            vmax=vmax,
            alpha=None, 
            linewidths=linewidths, 
            edgecolors=edgecolors,
            reduce_C_function = reduce_C_function, 
            mincnt=None, 
            marginals=marginals)

    cb = fig.colorbar(im,ax=ax)
    cb.set_label('count')

    # plt.axis([min(x)-1., max(x)+1., min(y)-0.1, max(y)+0.1])

    plt.axis([plot_xlim[0], plot_xlim[1], plot_ylim[0], plot_ylim[1]])
    
    if plot_ylabel is not None:
        plt.ylabel(plot_ylabel)
    if plot_xlabel is not None:
        plt.xlabel(plot_xlabel)
    if plot_title is not None:
        plt.title(plot_title)


    # Make the axis pretty
    plot_utils.simplify_axis(ax)

    # Save the file
    if savefile is not None:
        plot_utils.output(fig, savefile, savefile_size,
                          fontsize=fontsize)
    # Show
    if show_plot:
        plt.show()

    return im
    
