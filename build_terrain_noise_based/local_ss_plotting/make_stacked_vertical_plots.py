#!/usr/bin/env python
import numpy
import matplotlib.pyplot as plt
from ss_plotting import plot_utils
import ss_plotting.colors as colors

def plot_stacked_vertical_bar_graph(
    series, series_colors,
    series_labels=None,
    series_color_emphasis=None, 
    series_errs = None,
    series_err_colors = None,
    series_padding=0.0,
    series_use_labels=False,
    series_style=None,
    plot_xlabel = None,
    plot_ylabel = None, 
    plot_yinvert = False,
    plot_title = None,
    category_labels = None,
    category_ticks= True,
    category_padding = 0.25,
    barwidth=0.35,
    xpadding=0.,
    fontsize=8,
    legend_fontsize=8,
    legend_location='best',
    legend_title = None,
    legend_reverse_label = False,
    legend_bbox_to_anchor = None,
    legend_pos_rel_plot = None,
    legend_title_pos = None,
    legend_labelspacing = None,
    savefile = None,
    savefile_size = (3.4, 1.5),
    show_plot = True):
    """
    Plot a bar graph 
    @param series List of data for each series - each of these will be plotted in a different color
      Each series should have the same number of elements. 
    @param series_labels List of labels for each series - same length as series
    @param series_colors List of colors for each series - same length as series
    @param series_color_emphasis List of booleans, one for each series, indicating whether the series
       color should be bold - if None no series is bold
    @param series_errs The error values for each series - if None no error bars are plotted
    @param series_err_colors The colors for the error bars for each series, if None black is used
    @param plot_xlabel The label for the x-axis - if None no label is printed
    @param plot_ylabel The label for the y-axis - if None no label is printed
    @param plot_title A title for the plot - if None no title is printed
    @param category_labels The labels for each particular category in the histogram
    @param category_ticks If true, also place a tick at each category
    @param category_padding Fraction of barwidth (0 - 1) - distance between categories
    @param barwidth The width of each bar
    @param xpadding The padding between the first bar and the left axis and the last bar and the right axis
    @param fontsize The size of font for all labels
    @param legend_fontsize The size of font for the legend labels
    @param legend_location The location of the legend, if None no legend is included
    @param savefile The path to save the plot to, if None plot is not saved
    @param savefile_size The size of the saved plot
    @param show_plot If True, display the plot on the screen via a call to plt.show()
    @return fig, ax The figure and axis the plot was created in
    """

    # Validate
    if series is None or len(series) == 0:
        raise ValueError('No data series')
    num_series = len(series)

    if len(series_colors) != num_series:
        raise ValueError('You must define a color for every series')
        
    if series_labels is None:
        series_labels = [None for l in range(num_series)]
    if len(series_labels) != num_series:
        raise ValueError('You must define a label for every series')

    if series_errs is None:
        series_errs = [None for _ in series]
    if len(series_errs) != num_series:
        raise ValueError('series_errs is not None. Must provide error value for every series.')

    if series_err_colors is None:
        series_err_colors = ['black' for _ in series]
    if len(series_err_colors) != num_series:
        raise ValueError('Must provide an error bar color for every series')

    if series_color_emphasis is None:
        series_color_emphasis = [False for _ in series]
    if len(series_color_emphasis) != num_series:
        raise ValueError('The emphasis list must be the same length as the series_colors list')

    if series_use_labels and len(series[0]) > 1:
        raise ValueError('Only series containing one category may be labeled.')

    if series_style is None:
        series_style = [dict() for _ in series]

    fig, ax = plt.subplots()
    plot_utils.configure_fonts(fontsize=fontsize, legend_fontsize=legend_fontsize)

    if plot_yinvert:
      ax.invert_yaxis()

    spacing = category_padding*barwidth
    num_categories = len(series[0])
    category_width = spacing+barwidth
    # position for each bar 
    index = numpy.array(range(num_categories)) * category_width + xpadding
    # print index

    y_offset = numpy.array([0.0] * num_categories)

    for idx in range(num_series):
        # x_offset = idx * (barwidth + series_padding)
        x_offset = 0.

        style = dict(
            label = series_labels[idx],
            color = colors.get_plot_color(
                series_colors[idx], emphasis=series_color_emphasis[idx]),
            ecolor = colors.get_plot_color(series_err_colors[idx]),
            linewidth = 0,
        )
        style.update(series_style[idx])

        
        r = ax.bar(
            left = index + x_offset,
            height = series[idx],
            width = barwidth,
            bottom=y_offset,
            yerr = series_errs[idx],
            **style
        )
        y_offset = y_offset + series[idx]
    
    # Label the plot
    if plot_ylabel is not None:
        ax.set_ylabel(plot_ylabel)
    if plot_xlabel is not None:
        ax.set_xlabel(plot_xlabel)
    if plot_title is not None:
        ax.set_title(plot_title)
    
    # Add category tick marks
    if series_use_labels:
        indices = numpy.arange(num_series)
        ticks = xpadding + indices * (barwidth + series_padding) + 0.5 * barwidth
        
        ax.set_xticks(ticks)

        if series_labels is not None:
            ax.set_xticklabels(series_labels)
    elif category_ticks:
        ticks = index + .5*barwidth

        ax.set_xticks(ticks)

        if category_labels is not None:
            ax.set_xticklabels(category_labels)
    else:
        ax.set_xticks([])

    # Set the x-axis limits
    lims = [ 0,
             2 * xpadding
             + num_categories * (category_width + (num_series - 1) * series_padding)
             - spacing
    ]
    ax.set_xlim(lims)

    # Legend
    if legend_location is not None:

        if legend_pos_rel_plot == 'right':
            # http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend_location = 'center left'
            legend_bbox_to_anchor = (1, 0.5)
        # # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # http://matplotlib.org/1.3.1/users/legend_guide.html
        l = ax.legend(title=legend_title, labelspacing=legend_labelspacing,\
            frameon=False, loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)
        # http://matplotlib.org/1.3.1/users/legend_guide.html
        if legend_reverse_label:
            handles, labels = ax.get_legend_handles_labels()
            l = ax.legend(handles[::-1], labels[::-1],\
                title=legend_title, labelspacing=legend_labelspacing,\
                frameon=False,loc=legend_location,\
                bbox_to_anchor=legend_bbox_to_anchor)
        if legend_title_pos is not None:
            l.get_title().set_position(legend_title_pos)


    # Make the axis pretty
    plot_utils.simplify_axis(ax)

    # Save the file
    if savefile is not None:
        plot_utils.output(fig, savefile, savefile_size,
                          fontsize=fontsize, 
                          legend_fontsize=legend_fontsize
                          )
        
    # Show
    if show_plot:
        plt.show()

    return fig, ax

