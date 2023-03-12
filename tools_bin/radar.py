"""
https://towardsdatascience.com/how-to-create-and-visualize-complex-radar-charts-f7764d0f3652
"""
import numpy as np
import math
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import random

import textwrap


class ComplexRadar():
    """
    Create a complex radar chart with different scales for each variable
    Parameters
    ----------
    fig : figure object
        A matplotlib figure object to add the axes on
    variables : list
        A list of variables
    ranges : list
        A list of tuples (min, max) for each variable
    n_ring_levels: int, defaults to 5
        Number of ordinate or ring levels to draw
    show_scales: bool, defaults to True
        Indicates if we the ranges for each variable are plotted
    format_cfg: dict, defaults to None
        A dictionary with formatting configurations
    """
    def __init__(self, fig, variables, ranges, n_ring_levels=5, show_scales=True, format_cfg=None):
        
        # Default formatting
        self.format_cfg = {
            # Axes
            # https://matplotlib.org/stable/api/figure_api.html
            'axes_args': {},
            # Tick labels on the scales
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rgrids.html
            'rgrid_tick_lbls_args': {'fontsize':8},
            # Radial (circle) lines
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html
            'rad_ln_args': {},
            # Angle lines
            # https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
            'angle_ln_args': {},
            # Include last value (endpoint) on scale
            'incl_endpoint':False,
            # Variable labels (ThetaTickLabel)
            'theta_tick_lbls':{'va':'top', 'ha':'center'},
            'theta_tick_lbls_txt_wrap':15,
            'theta_tick_lbls_brk_lng_wrds':False,
            'theta_tick_lbls_pad':25,
            # Outer ring
            # https://matplotlib.org/stable/api/spines_api.html
            'outer_ring':{'visible':True, 'color':'#d6d6d6'}
        }
        
        if format_cfg is not None:
            self.format_cfg = { k:(format_cfg[k]) if k in format_cfg.keys() else (self.format_cfg[k]) 
                 for k in self.format_cfg.keys()}        
        
        
        # Calculate angles and create for each variable an axes
        # Consider here the trick with having the first axes element twice (len+1)
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.2,0.2,0.6,0.6], 
                             polar=True,
                             label = "axes{}".format(i),
                             **self.format_cfg['axes_args']) for i in range(len(variables)+1)]
        
        # Ensure clockwise rotation (first variable at the top N)
        for ax in axes:
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)
        
        # Writing the ranges on each axes
        for i, ax in enumerate(axes):

            # Here we do the trick by repeating the first iteration
            j = 0 if (i==0 or i==1) else i-1
            ax.set_ylim(*ranges[j])
            # Set endpoint to True if you like to have values right before the last circle
            grid = np.linspace(*ranges[j], num=n_ring_levels, 
                               endpoint=self.format_cfg['incl_endpoint'])
            gridlabel = ["{}".format(round(x,2)) for x in grid]
            gridlabel[0] = "" # remove values from the center
            lines, labels = ax.set_rgrids(grid, 
                                          labels=gridlabel, 
                                          angle=angles[j],
                                          **self.format_cfg['rgrid_tick_lbls_args']
                                         )
            
            ax.set_ylim(*ranges[j])
            ax.spines["polar"].set_visible(False)
            ax.grid(visible=False)
            
            if show_scales == False:
                ax.set_yticklabels([])

        # Set all axes except the first one unvisible
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            
        # Setting the attributes
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax1 = axes[1]
        self.plot_counter = 0
        
        
        # Draw (inner) circles and lines
        self.ax.yaxis.grid(**self.format_cfg['rad_ln_args'])
        # Draw outer circle
        self.ax.spines['polar'].set(**self.format_cfg['outer_ring'])
        # Draw angle lines
        self.ax.xaxis.grid(**self.format_cfg['angle_ln_args'])

        # ax1 is the duplicate of axes[0] (self.ax)
        # Remove everything from ax1 except the plot itself
        self.ax1.axis('off')
        self.ax1.set_zorder(9)
        
        # Create the outer labels for each variable
        l, text = self.ax.set_thetagrids(angles, labels=variables)
        
        # Beautify them
        labels = [t.get_text() for t in self.ax.get_xticklabels()]
        labels = ['\n'.join(textwrap.wrap(l, self.format_cfg['theta_tick_lbls_txt_wrap'], 
                                          break_long_words=self.format_cfg['theta_tick_lbls_brk_lng_wrds'])) for l in labels]
        self.ax.set_xticklabels(labels, **self.format_cfg['theta_tick_lbls'])
        
        for t,a in zip(self.ax.get_xticklabels(),angles):
            if a == 0:
                t.set_ha('center')
            elif a > 0 and a < 180:
                t.set_ha('left')
            elif a == 180:
                t.set_ha('center')
            else:
                t.set_ha('right')

        self.ax.tick_params(axis='both', pad=self.format_cfg['theta_tick_lbls_pad'])


    def _scale_data(self, data, ranges):
        """Scales data[1:] to ranges[0]"""
        # for each dataset
        # for d, (y1, y2) in zip(data[1:], ranges[1:]):
        #     assert (y1 <= d <= y2) or (y2 <= d <= y1)
        x1, x2 = ranges[0]
        d = data[0]
        sdata = [d]
        # for each dataset
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if np.isnan(d):
                d = y1
            sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
        return sdata
        
    def plot(self, data, *args, **kwargs):
        """Plots a line"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
        self.plot_counter = self.plot_counter+1
    
    def fill(self, data, *args, **kwargs):
        """Plots an area"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
        
    def use_legend(self, *args, **kwargs):
        """Shows a legend"""
        self.ax1.legend(*args, **kwargs)
    
    def set_title(self, title, pad=25, **kwargs):
        """Set a title"""
        self.ax.set_title(title,pad=pad, **kwargs)


# data = load_wine(as_frame=True)['data']
# pipe = make_pipeline(StandardScaler(),
#                     KMeans(init="k-means++", n_clusters=3, 
#                             n_init=4, random_state=0)
#                     )

# data['cluster'] = pipe.fit_predict(data)
# result = data.groupby(['cluster']).mean()


# min_max_per_variable = data.describe().T[['min', 'max']]
# min_max_per_variable['min'] = min_max_per_variable['min'].apply(lambda x: int(x))
# min_max_per_variable['max'] = min_max_per_variable['max'].apply(lambda x: math.ceil(x))

# variables = result.columns # dataset
# ranges = list(min_max_per_variable.itertuples(index=False, name=None))   
# import ipdb; ipdb.set_trace()


# ["VOC to non-VOC", "COCO to LVIS", "COCO to UVO", "COCO to Objects365"]
# data = pd.DataFrame(
#     {
#         "VOC to non-VOC": {"OLN": 26.9, "LDET": 27.4, "GGN": 28.7, "Previous SOTA": 33.7, "SWORD (This Work)": 35.9},
#         "COCO to LVIS":   {"OLN": np.nan, "LDET": np.nan, "GGN": 20.4, "Previous SOTA": 20.4, "SWORD (This Work)": 24.8},
#         "COCO to UVO (All)":   {"OLN": np.nan, "LDET": 40.4, "GGN": 43.4, "Previous SOTA": 43.4, "SWORD (This Work)": 53.0},
#         "COCO to UVO (Novel)": {"OLN": np.nan, "LDET": 30.5, "GGN": np.nan, "Previous SOTA": 30.5, "SWORD (This Work)": 43.5},
#         "COCO to Objects365 (All)": {"OLN": np.nan, "LDET": 41.4, "GGN": np.nan, "Previous SOTA": 41.4, "SWORD (This Work)": 51.9},
#         "COCO to Objects365 (Novel)": {"OLN": np.nan, "LDET": 36.8, "GGN": np.nan, "Previous SOTA": 36.8, "SWORD (This Work)": 45.9},
#     }
# )

# the empty value is 0
data = pd.DataFrame(
    {
        "Object Detection": {"Cascade Mask R-CNN": 58.7, "SeqTR": np.nan, "Unicorn": np.nan,  "VMT": np.nan, "UNINEXT": 60.6},
        "Instance Segmentation":  {"Cascade Mask R-CNN": 50.9, "SeqTR": np.nan, "Unicorn": np.nan,  "VMT": np.nan, "UNINEXT": 51.8},
        "REC":  {"Cascade Mask R-CNN": np.nan, "SeqTR": 87.0, "Unicorn": np.nan,  "VMT": np.nan, "UNINEXT": 92.6},
        "RES": {"Cascade Mask R-CNN": np.nan, "SeqTR": 71.7, "Unicorn": np.nan,  "VMT": np.nan, "UNINEXT": 82.2},
        "SOT": {"Cascade Mask R-CNN": np.nan, "SeqTR": np.nan, "Unicorn": 68.5,  "VMT": np.nan, "UNINEXT": 72.2},
        "VOS": {"Cascade Mask R-CNN": np.nan, "SeqTR": np.nan, "Unicorn": 69.2,  "VMT": np.nan, "UNINEXT": 81.8},
        "MOT": {"Cascade Mask R-CNN": np.nan, "SeqTR": np.nan, "Unicorn": 41.2,  "VMT": np.nan, "UNINEXT": 44.2},
        "MOTS": {"Cascade Mask R-CNN": np.nan, "SeqTR": np.nan, "Unicorn": 29.6,  "VMT": 28.7, "UNINEXT": 35.7},
        "VIS": {"Cascade Mask R-CNN": np.nan, "SeqTR": np.nan, "Unicorn": np.nan,  "VMT": 59.7, "UNINEXT": 66.9},
        "R-VOS": {"Cascade Mask R-CNN": np.nan, "SeqTR": np.nan, "Unicorn": np.nan,  "VMT": np.nan, "UNINEXT": 70.1},
    }
)


# data = pd.DataFrame(
#     {
#         "VOC to non-VOC": {"Deformable-DETR": 13.5, "OLN": 26.9, "LDET": 27.4, "GGN": 28.7,  "SWORD (This Work)": 35.9},
#         "COCO to LVIS":   {"Deformable-DETR": 16.4, "OLN": 0, "LDET": 0, "GGN": 20.4, "SWORD (This Work)": 24.8},
#         "COCO to UVO (All)":   {"Deformable-DETR": 50.3, "OLN": 42.1, "LDET": 40.4, "GGN": 43.4,  "SWORD (This Work)": 53.0},
#         "COCO to UVO (Novel)": {"Deformable-DETR": 37.9, "OLN": 34.7, "LDET": 30.5, "GGN": 0,  "SWORD (This Work)": 43.5},
#         "COCO to Objects365 (All)": {"Deformable-DETR": 48.7, "OLN": 43.1, "LDET": 41.4, "GGN": 0,  "SWORD (This Work)": 51.9},
#         "COCO to Objects365 (Novel)": {"Deformable-DETR": 40.1, "OLN": 38.9, "LDET": 36.8, "GGN": 0,  "SWORD (This Work)": 45.9},
#     }
# )

# NOTE: this is somewhat tricky
min_max_per_variable = data.describe().T[['min', 'max']]
min_max_per_variable['min'] = min_max_per_variable['min'].apply(lambda x: int(x)-10)
min_max_per_variable['max'] = min_max_per_variable['max'].apply(lambda x: math.ceil(x+1))

variables = data.columns # dataset
ranges = list(min_max_per_variable.itertuples(index=False, name=None))   

# import ipdb; ipdb.set_trace()


format_cfg = {
    'rad_ln_args': {'visible':True, 'linewidth':2},
    'outer_ring': {'visible': True, 'linewidth':1.5},
    'rgrid_tick_lbls_args': {'family': 'serif', 'fontsize':15, 'fontweight': 'semibold'}, 
    'theta_tick_lbls': {'family': 'serif', 'fontsize':17, 'fontweight': 'semibold'},
    'theta_tick_lbls_pad':15
}

# "family":  ['cursive', 'fantasy', 'monospace', 'sans', 'sans serif', 'sans-serif', 'serif']
# 

fig = plt.figure(figsize=(10, 10), dpi=200)
radar = ComplexRadar(fig, variables, ranges, n_ring_levels=4, show_scales=True, format_cfg=format_cfg)

# custom_colors = ['#f9b4ab', '#264e70', '#679186']
# for g,c in zip(result.index, custom_colors):
#     # import ipdb; ipdb.set_trace()
#     radar.plot(result.loc[g].values, label=f"cluster {g}", color=c)
#     radar.fill(result.loc[g].values, alpha=0.5, color=c)

custom_colors = ['#FFD700', '#FF7F00', '#228B22', '#00BFFF', '#EE7AE9']
for g, c in zip(data.index, custom_colors):
    # import ipdb; ipdb.set_trace()
    radar.plot(data.loc[g].values, label=f"{g}", alpha=0.65, color=c, linewidth=2.5)
    radar.fill(data.loc[g].values, alpha=0.25, color=c)

# radar.set_title("Radar chart solution with different scales",pad=50)
# radar.use_legend(loc='lower left', bbox_to_anchor=(0.15, -0.25),ncol=radar.plot_counter)
plt.savefig("test.png")
plt.close(fig)


# plt.show() 