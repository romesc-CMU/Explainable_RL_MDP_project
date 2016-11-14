from local_ss_plotting import plot_utils
import local_ss_plotting.colors as ss_colors
# from ss_plotting import plot_utils
import matplotlib.pyplot as plt


class Drawer:
    def __init__(self):
        self.colors = {}
        self.colors['blue_original'] = 'blue'
        self.colors['blue'] = ss_colors.get_plot_color('blue',False)
        self.colors['blue_emp'] = ss_colors.get_plot_color('blue',True)
        self.colors['red'] = ss_colors.get_plot_color('red',False)
        self.colors['red_emp'] = ss_colors.get_plot_color('red',True)
        self.colors['grey'] = ss_colors.get_plot_color('grey',False)
        self.colors['green'] = ss_colors.get_plot_color('green',False)
        self.colors['pink'] = ss_colors.get_plot_color('pink',False)
        self.colors['orange'] = ss_colors.get_plot_color('orange',False)
        self.colors['purple'] = ss_colors.get_plot_color('purple',False)
        self.colors['yellow'] = ss_colors.get_plot_color('yellow',False)
        

    def show_plt(self, fig, file_name=None, show=False, savefile_size=(8.,6.), fontsize=8, legend_fontsize=8):
        plot_utils.configure_fonts(fontsize=fontsize, legend_fontsize=legend_fontsize)
        if show == True:
            plt.show()
        if file_name != None:
            plot_utils.output(fig, file_name, savefile_size,\
                    fontsize=fontsize, legend_fontsize=legend_fontsize)

    def get_colors(self):
        return self.colors