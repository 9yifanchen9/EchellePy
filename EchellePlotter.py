import numpy as np
import pandas as pd
from astropy.convolution import convolve, Box1DKernel
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.pyplot as plt

class EchellePlotter:
    def __init__(self,     
        freq, power,
        Dnu_min, Dnu_max, fmin=0, fmax=None,
        step=None, cmap="BuPu", fig=None, ax=None,
        interpolation=None, smooth=False, smooth_filter_width=50.0,
        scale=None, return_coords=False, **kwargs):

        self.Dnu = (Dnu_min + Dnu_max) / 2.0
        self.fmin = fmin
        self.fmax = fmax

        # Styles for labels
        self.colors = {0: "blue", 1: "green", 2: "orange"}
        self.markers = {0: "s", 1: "^", 2: "o"}

        if Dnu_max < Dnu_min:
            raise ValueError("Maximum range can not be less than minimum")

        if smooth_filter_width < 1:
            raise ValueError("The smooth filter width can not be less than 1!")

        # Smoothen power spectrum
        if smooth:
            self.power = smooth_power(power, smooth_filter_width)

        self.freq = freq
        self.power = power

        # Compute echelle
        self.x, self.y, self.z = echelle(freq, self.power, self.Dnu, 
            sampling=1, fmin=fmin, fmax=fmax)

        # Scale image intensities
        if scale is "sqrt":
            self.z = np.sqrt(self.z)
        elif scale is "log":
            self.z = np.log10(self.z)
        self.scale = scale

        # Create subplot(s)
        if ax == None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = ax.figure
            self.ax = ax

        plt.subplots_adjust(left=0.3, bottom=0.25)

        # Step in x-axis as median difference
        if step is None:
            step = np.median(np.diff(self.freq))

        self.step = step

        # A list of l-mode values (used for removing points)
        self.l_labels = []

        self.legend_labels = []

        # Create widgets
        self.create_Dnu_slider(Dnu_min, Dnu_max, step)
        self.create_label_radio_buttons()
        self.create_image(self.x, self.y, self.z, cmap, interpolation)
        self.create_remove_label_button()

        # List of 3 arrays, with index being l-mode label
        # e.g. label_[1][2] is 3rd frequency for l=1 mode label
        self.f_labels = [[],[],[]]

        # 3 scatter plots corresponding to l-mode labels
        self.create_label_scatters()
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def create_label_scatters(self):
        """ Scatter plots for l-mode labelling """
        self.scatters = [None, None, None]


    def create_Dnu_slider(self, Dnu_min, Dnu_max, step):
        """Create Slider that adjusts Dnu"""
        # Frequency adjust axes
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])

        # Value format string
        valfmt = "%1." + str(len(str(step).split(".")[-1])) + "f"

        # Slider and event calls
        self.slider = Slider(
            axfreq,
            u"\u0394\u03BD",
            Dnu_min,
            Dnu_max,
            valinit=(Dnu_max + Dnu_min) / 2.0,
            valstep=step,
            valfmt=valfmt,
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.slider.on_changed(self.update)

    def create_label_radio_buttons(self):
        """
        Create RadioButtons to select labelling
        """
        # axcolor = 'lightgoldenrodyellow'
        axcolor = 'white'
        rax = plt.axes([0.03, 0.7, 0.15, 0.15], facecolor=axcolor)
        self.radio_l = RadioButtons(rax, ('-', 'l=0', 'l=1', 'l=2'))

    def create_remove_label_button(self):
        """Create Button to remove last label"""
        axcolor = 'white'
        rax = plt.axes([0.03, 0.3, 0.15, 0.10], facecolor=axcolor)
        self.remove_button = Button(rax, "Undo")
        self.remove_button.on_clicked(self.remove_previous_label)

    def create_image(self, x, y, z, cmap, interpolation):
        """Create the echelle image"""
        self.line = self.ax.imshow(
            z,
            aspect="auto",
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin="lower",
            cmap=cmap,
            interpolation=interpolation,
        )

        self.ax.set_xlabel(u"Frequency mod \u0394\u03BD")
        self.ax.set_ylabel("Frequency")

    def show(self):
        """Show plot using plt.show()"""
        plt.show()

    def update(self, Dnu):
        """Updates echelle diagram given new Dnu"""
        self.Dnu = Dnu

        # Calculate new data
        self.x, self.y, self.z = echelle(self.freq, self.power, self.Dnu, sampling=1, fmin=self.fmin, fmax=self.fmax)
        if self.scale != None:
            if self.scale == "sqrt":
                self.z = np.sqrt(self.z)
            elif self.scale == "log":
                self.z = np.log10(self.z)
        self.line.set_array(self.z)

        # Set new visible range for x and y axes
        self.line.set_extent((self.x.min(), self.x.max(), self.y.min(), self.y.max()))
        self.ax.set_xlim(0, self.Dnu)

        # Shift labelled points accordingly
        self.update_labels()

        # Render
        self.fig.canvas.blit(self.ax.bbox)

    def update_labels(self):
        """Update the labels to new echelle diagram coordinates"""
        # Replot
        l = 0
        while l < len(self.f_labels):
            self.update_scatter(l)
            l += 1

    def on_key_press(self, event):
        """Key press to shift slider left and right"""
        if event.key == "left":
            new_Dnu = self.slider.val - self.slider.valstep
        elif event.key == "right":
            new_Dnu = self.slider.val + self.slider.valstep
        else:
            new_Dnu = self.slider.val

        self.slider.set_val(new_Dnu)
        self.update(new_Dnu)

    def on_click(self, event):
        """Clicking event"""
        # Only add label click in the echelle, and selected l mode to label
        click_in_plot = event.inaxes == self.line.axes
        l_mode = self.radio_l.value_selected
        if not click_in_plot or l_mode == "-": return

        # l mode integer
        l = int(l_mode.replace("l=",""))

        # Coordinate of clicks on the image array
        x, y = event.xdata, event.ydata
        
        y_inc = self.y[1] - self.y[0]


        # Find the nearest x and y value in self.x and self.y
        nearest_x_index = (np.abs(self.x-x)).argmin()

        # Find y that are just below our cursor
        nearest_y = self.y[self.y-y < 0].max()

        self.add_point(self.x[nearest_x_index], nearest_y, l)

    def add_point(self, x, y, l):
        """Add point to the plot and update scatter"""
        f = self.coord_to_freq(x, y)
        self.f_labels[l].append(f)
        self.l_labels.append(l)
        self.update_scatter(l)

    def remove_previous_label(self, event):
        """Removes the previously added point"""
        if len(self.l_labels) == 0:
            return

        l = self.l_labels.pop()
        self.f_labels[l].pop()
        self.update_scatter(l)

    def update_scatter(self, l):
        """Updates scatter plot"""
        no_label_no_scatter = len(self.f_labels[l]) == 0 and self.scatters[l] == None
        no_label_has_scatter = len(self.f_labels[l]) == 0 and self.scatters[l] != None
        has_label_no_scatter = len(self.f_labels[l]) != 0 and self.scatters[l] == None

        if no_label_no_scatter:
            return

        # No labels for the mode anymore
        elif no_label_has_scatter:
            self.scatters[l].remove()
            self.scatters[l] = None
            self.legend_labels.remove(f"l={l}")

        # First label of the mode
        elif has_label_no_scatter:
            # Update scatter plots based on l_labels
            color = self.colors[l]
            marker = self.markers[l]
            label = f"l={l}"

            x_labels, y_labels = self.get_coords(l, self.Dnu)

            self.scatters[l] = self.ax.scatter(x_labels, y_labels, 
                s=50, marker=marker, label=label, color=color)
            
            # Set new visible range for x and y axes
            self.line.set_extent((self.x.min(), self.x.max(), self.y.min(), self.y.max()))
            self.ax.set_xlim(0, Dnu)            

            label = f"l={l}"
            if label not in self.legend_labels:
                self.legend_labels.append(f"l={l}")
        else:
            x_labels, y_labels = self.get_coords(l, self.Dnu)
            self.scatters[l].set_offsets(np.c_[x_labels, y_labels])
        
        self.ax.legend(self.legend_labels)
        self.fig.canvas.draw()


    def freq_to_period(self, freq):
        """From frequency (muHz) to period (s)"""
        return 1e6/freq

    def period_to_freq(self, period):
        """From period (s) to frequency (muHz)"""
        return 1e6/period

    def period_to_coord(self, period, DP):
        """Period to coordinate on the Echelle period"""
        y_inc = self.y[1] - self.y[0]
        x = (freq - self.x.min()) % Dnu# freq mod Dnu is x-coordinate
        y = self.y[self.y < freq].max()# The bin just below the frequency is y-coordinate
        return x, y

    def get_coords(self, l, Dnu):
        xs = []
        ys = []
        for f in self.f_labels[l]:
            x, y = self.freq_to_coord(f, Dnu)
            xs.append(x)
            ys.append(y)

        return xs, ys

    def freq_to_coord(self, freq, Dnu):
        """Frequency to coordinate on the Echelle"""
        y_inc = self.y[1] - self.y[0]
        x = (freq - self.x.min()) % Dnu# freq mod Dnu is x-coordinate
        # The bin just below the frequency, plus half increment (for labelling in middle) 
        y = self.y[self.y < freq].max() + 0.5*(self.y[1] - self.y[0])
        return x, y

    def coord_to_freq(self, x, y):
        return y + x

    def get_legend_labels(self):
        """Get legend labels based on whether there is data for each legend"""
        return [f"l={l}" for l in range(len(self.f_labels)) if len(self.f_labels[l]) != 0]


def echelle(freq, power, Dnu, fmin=0.0, fmax=None, offset=0.0, sampling=0.1):
    """Calculates the echelle diagram. Use this function if you want to do
    some more custom plotting.

    Parameters
    ----------
    freq : array-like
        Frequency values
    power : array-like
        Power values for every frequency
    Dnu : float
        Value of deltanu
    fmin : float, optional
        Minimum frequency to calculate the echelle at, by default 0.
    fmax : float, optional
        Maximum frequency to calculate the echelle at. If none is supplied,
        will default to the maximum frequency passed in `freq`, by default None
    offset : float, optional
        An offset to apply to the echelle diagram, by default 0.0

    Returns
    -------
    array-like
        The x, y, and z values of the echelle diagram.
    """
    if fmax == None:
        fmax = freq[-1]

    # Apply offset
    fmin = fmin - offset
    fmax = fmax - offset
    freq = freq - offset

    # Quality of life checks
    if fmin <= 0.0:
        fmin = 0.0
    else:
        # Make sure it partitions exactly
        fmin = fmin - (fmin % Dnu)

    # trim data
    index = (freq >= fmin) & (freq <= fmax)
    trimx = freq[index]

    # median interval width
    samplinginterval = np.median(trimx[1:-1] - trimx[0:-2]) * sampling

    # Fixed sampling interval x values
    xp = np.arange(fmin, fmax + Dnu, samplinginterval)

    # Interpolant (approximation) for xp values (given the frequency and power)
    yp = np.interp(xp, freq, power)

    # Number of stacks and Number of elements in each stack
    n_stack = int((fmax - fmin) / Dnu)
    n_element = int(Dnu / samplinginterval)

    # Number of rows for each datapoint (elongate graph)
    morerow = 2

    # Array of length = number of stacks
    arr = np.arange(1, n_stack) * Dnu

    # Double the size (due to 2 rows?)
    arr2 = np.array([arr, arr])

    # y-values of each stack - reshape 2 stacks
    yn = np.reshape(arr2, len(arr) * 2, order="F")

    # Ending values - Insert 0 in beginning and append number of stacks * Dnu, plus offsets
    yn = np.insert(yn, 0, 0.0)
    yn = np.append(yn, n_stack * Dnu) + fmin + offset

    # x values of partition
    xn = np.arange(1, n_element + 1) / n_element * Dnu

    # image as 2D array
    z = np.zeros([n_stack * morerow, n_element])

    # Add yp values to rows of image
    for i in range(n_stack):
        for j in range(i * morerow, (i + 1) * morerow):
            # Multiple rows of the same data
            z[j, :] = yp[n_element * (i) : n_element * (i + 1)]

    return xn, yn, z

if __name__ == "__main__":
    # Read in csv
    lc_df = pd.read_csv("data/KIC9773821_LC.csv")
    ps_df = pd.read_csv("data/KIC9773821_PS.csv")

    # Prepare numpy array data
    time = lc_df.time.to_numpy()
    flux = lc_df.flux.to_numpy()

    freq = ps_df.freq.to_numpy()
    pows = ps_df.pows.to_numpy()

    amp = np.sqrt(pows)
    df=freq[1]-freq[0]
    smooth = .1/df  # in muHz
    amp=convolve(amp, Box1DKernel(smooth))
    

    Dnu = 8.10 # large frequency separation (muHz)
    fmin = 102 - 40
    fmax = 102 + 40

    e = EchellePlotter(freq, amp, Dnu-3, Dnu+3, step=.05, 
        fmin=fmin, fmax=fmax, return_coords=True)
    e.show()
