import pandas as pd
import numpy as np
from astropy.convolution import convolve, Box1DKernel
from echelle_fork.echelle import echelle, plot_echelle
import bokeh
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, RadioButtonGroup, Button, LinearColorMapper
from bokeh.models.ranges import DataRange1d
from bokeh.events import DoubleTap
from bokeh.plotting import figure, output_file, show
from bokeh.transform import factor_cmap, factor_mark
from bokeh.io import curdoc
from bokeh.colors import RGB
from matplotlib import cm


#===========================================================
# Data
#===========================================================

# Read in csv
ps_df = pd.read_csv("data/KIC9773821_PS.csv")

# Prepare numpy array data
freq = ps_df.freq.to_numpy()
pows = ps_df.pows.to_numpy()

# Constants we already know about the star
Dnu = 8.10
numax = 102
fmin = numax - 30
fmax = numax + 30

# Magic maths
amp = np.sqrt(pows)
df=freq[1] - freq[0]
smooth = .1/df  # in muHz
amp=convolve(amp, Box1DKernel(smooth))

# x-ticks, y-ticks, and 2d array z as image
xn, yn, z = echelle(freq, amp, Dnu,
    fmin=fmin, fmax=fmax)

# Get rid of half the data (too much for bokeh!)
#yn = yn[::2]
#z = z[::2]


# Change to period
period = 1e6/freq # convert muHz to seconds
DP = 194.0 # first guess at period spacing
pmin=1e6/fmax
pmax=1e6/fmin

# echelle p
ixn, iyn, iz = echelle(period[::-1], amp[::-1], DP, fmin=pmin, fmax=pmax)


# Data source for echelle with frequency
source_f = ColumnDataSource(data=dict(z=[z]))

# Data source of echelle with period
source_p = ColumnDataSource(data=dict(z=[iz]))

# Tuple of (x,y) for l node labelling
l_points = ColumnDataSource(data=dict(x=[], y=[], ix=[], iy=[], l=[]))

#===========================================================
# Plot setup
#===========================================================

# Tools and tooltips
tools = "pan,box_select,save,reset,wheel_zoom"
tooltips = [("x", "$x"), ("y", "$y"), ("value", "@z")]# What is displayed

# Dimensions of figure
width=480
height=600

# Create plots
echelle_f = figure(tooltips=tooltips,
            tools=tools,
            plot_width=width, plot_height=height)
echelle_f.x_range.range_padding = echelle_f.y_range.range_padding = 0
echelle_f.xaxis.axis_label = f"Frequency mod {Dnu:.2f} (muHz)"
echelle_f.yaxis.axis_label = f"Frequency (muHz)"

echelle_p = figure(tooltips=tooltips,
            tools=tools,
            plot_width=width, plot_height=height)
echelle_p.x_range.range_padding = echelle_p.y_range.range_padding = 0
echelle_p.xaxis.axis_label = f"Period mod {DP:.2f} (s)"
echelle_p.yaxis.axis_label = f"Period (s)"

# More intense should have more color
palette = list(bokeh.palettes.Blues256)
palette.reverse()


# output_file("image.html", title="image.py example")
# bokeh.io.show(p, inputs)  # open a browser


# Echelle frequency image - must give a vector of image(s)
width = xn.max()-xn.min()
height = yn.max()-yn.min()
f_image = echelle_f.image(image='z', x=xn.min(), y=yn.min(), 
    dw=width, dh=height, 
    palette=palette, level="image", source=source_f)
echelle_f.xgrid.visible = echelle_f.ygrid.visible = False


# Echelle period image
width = ixn.max()-ixn.min()
height = iyn.max()-iyn.min()

# Flipped image means starting position (to the left) has more positive value
p_image = echelle_p.image(image='z', x=width+ixn.min(), y=height+iyn.min(),
    dw=width, dh=height,
    palette=palette, level="image", source=source_p)
echelle_p.xgrid.visible = echelle_p.ygrid.visible = False

# Override auto_calculated range to center diagram
echelle_p.x_range = DataRange1d(range_padding=0.0, start=width + ixn.min(), end=ixn.min(), flipped=True)
echelle_p.y_range = DataRange1d(range_padding=0.0, start=height + iyn.min(), end=iyn.min(), flipped=True)


# Mode labels and styles
L_MODES = ['l=0', 'l=1', 'l=2']
MARKERS = ['square', 'circle', 'triangle']

# Echelle frequency labels
echelle_f.scatter(x='x', y='y', source=l_points, legend_field='l',
    size=18, fill_alpha=1,
    marker=factor_mark('l', MARKERS, L_MODES),
    color=factor_cmap('l', 'Category10_3', L_MODES))

# Echelle period labels
echelle_p.scatter(x='ix', y='iy', source=l_points, legend_field='l',
    size=18, fill_alpha=1,
    marker=factor_mark('l', MARKERS, L_MODES),
    color=factor_cmap('l', 'Category10_3', L_MODES))

#===========================================================
# Data update
#===========================================================

# Selections for l modes
l_modes = RadioButtonGroup(labels=["l=0", "l=1", "l=2"], active=0, aspect_ratio="auto")

# SLider for Dnu guesses
partition_f = Slider(title="Dnu (mu Hz)", value=Dnu, start=Dnu - 5, end=Dnu + 5, step=0.05)

# SLider for DT guesses
partition_p = Slider(title="DT (s)", value=DP, start=DP - 10, end=DP + 10, step=0.05)

# Button to undo
undo_label = Button(label="Remove last label", button_type="success")


def update_period_data(attr, old, new):
    """
    Update echelle diagram
    """
    global p_image
    # Get current slider values
    width = partition_p.value

    # Get new echelle
    ixn, iyn, iz = echelle(period[::-1], amp[::-1], width, 
        fmin=pmin, fmax=pmax)

    # Update to source
    source_p.data.update(dict(z=[iz]))

    # Echelle period image
    width = ixn.max()-ixn.min()
    height = iyn.max()-iyn.min()

    # Flipped image means starting position (to the left) has more positive value
    prev_image = p_image
    
    p_image = echelle_p.image(image='z', x=width+ixn.min(), y=height+iyn.min(),
        dw=width, dh=height,
        palette=palette, level="image", source=source_p)

    prev_image.visible = False
    # Override auto_calculated range to center diagram
    echelle_p.xaxis.axis_label = f"Period mod {width:.2f} (s)"
    echelle_p.x_range.start = width + ixn.min()
    echelle_p.x_range.end = ixn.min()
    echelle_p.y_range.start = height + iyn.min()
    echelle_p.y_range.end = iyn.min()

    print(f"DT changed to {new}")


def update_freq_data(attr, old, new):
    """
    Update echelle diagram
    """
    global f_image
    # Get current slider values
    width = partition_f.value

    # Get new echelle
    xn, yn, z = echelle(freq, amp, width,
        fmin=fmin, fmax=fmax)

    # Update to source
    source_f.data.update(dict(z=[z]))

    # Echelle frequency image - must give a vector of image(s)
    width = xn.max()-xn.min()
    height = yn.max()-yn.min()

    f_image.visible = False
    f_image = echelle_f.image(image='z', x=xn.min(), y=yn.min(), 
        dw=width, dh=height, 
        palette=palette, level="image", source=source_f)
    echelle_f.xgrid.visible = echelle_f.ygrid.visible = False

    echelle_f.xaxis.axis_label = f"Frequency mod {width:.2f} (muHz)"
    echelle_f.x_range.start = xn.min()
    echelle_f.x_range.end = xn.max()
    echelle_f.y_range.start = yn.min()
    echelle_f.y_range.end = yn.max()

    print(f"Dnu changed to {new}")


def add_frequency_point(event):
    """
    Add a l mode label by double clicking on frequency diagram
    """
    # Get points upon double click
    x = event.x
    y = event.y

    # Add points to mode's data and update
    x_vals = l_points.data['x']
    y_vals = l_points.data['y']
    ix_vals = l_points.data['ix']
    iy_vals = l_points.data['iy']
    l_vals = l_points.data['l']
    l = str(l_modes.active)

    # Add to list and update
    x_vals.append(x)
    y_vals.append(y)
    ix_vals.append(1e6/x)
    iy_vals.append(1e6/y)
    l_vals.append(f"l={l}")

    l_points.data.update(dict(x=x_vals, y=y_vals, 
        ix=ix_vals, iy=iy_vals, l=l_vals))

    print(f"Added ({x:.1f}, {y:.1f}) and ({1e6/x:.1f}, {1e6/y:.1f}) to {l} mode")


def remove_point(event):
    """
    Remove the last label
    """
    x_vals = l_points.data['x']
    ix_vals = l_points.data['ix']
    iy_vals = l_points.data['iy']    
    y_vals = l_points.data['y']
    l_vals = l_points.data['l']

    if len(x_vals) > 0:
        x = x_vals.pop(-1)
        y = y_vals.pop(-1)
        ix = ix_vals.pop(-1)
        iy = iy_vals.pop(-1)
        l = l_vals.pop(-1)
        print(f"Removed ({x:.1f}, {y:.1f}) to {l} mode")

        l_points.data.update(dict(x=x_vals, y=y_vals, 
            ix=ix_vals, iy=iy_vals, l=l_vals))


# Interactive changes
partition_f.on_change('value', update_freq_data)
partition_p.on_change('value', update_period_data)
echelle_f.on_event(DoubleTap, add_frequency_point)
undo_label.on_click(remove_point)
inputs = column(partition_f, partition_p)


#===========================================================
# Finalize
#===========================================================

print(f"xn.min()={xn.min()}")
print(f"xn.max()={xn.max()}")
print(f"yn.min()={yn.min()}")
print(f"yn.max()={yn.max()}")
print(f"ixn.min()={ixn.min()}")
print(f"ixn.max()={ixn.max()}")
print(f"iyn.min()={iyn.min()}")
print(f"iyn.max()={iyn.max()}")

curdoc().add_root(column(row(echelle_f, echelle_p), inputs, row(l_modes, undo_label)))
curdoc().title = "Interactive Echelle Labelling"
#===========================================================