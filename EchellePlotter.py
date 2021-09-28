import numpy as np
import pandas as pd
from astropy.convolution import convolve, Box1DKernel
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
import json
import tkinter
from tkinter.filedialog import askopenfilename


class EchellePlotter:
    def __init__(self,     
        freq, power,
        Dnu_min, Dnu_max, fmin=0, fmax=None, step=None,
        period=None, period_power=None, pmin=None, pmax=None, DP_min=None, DP_max=None, pstep=None,
        cmap="BuPu", colors={}, markers={}, plot_line=[], size=50, 
        figsize=(6.4, 4.8), dpi=100,
        interpolation=None, smooth=False, smooth_filter_width=50.0, scale=None):
    #==================================================================
    # Class attributes and argument checks
    #==================================================================
        self.Dnu = (Dnu_min + Dnu_max) / 2.0
        self.fmin = fmin
        self.fmax = fmax
        self.scale = scale
        self.plot_line = plot_line
        self.size = size

        if period is not None and period_power is not None:
            self.plot_period = True
        else:
            self.plot_period = False

        if self.plot_period:
            if DP_min == None or DP_max == None:
                raise Exception("Must provide DP_min and DP_max when plotting period echelle")
            if pstep == None:
                raise Exception("Step of period echelle slider (pstep) must be provided")

            self.DP = (DP_min + DP_max) / 2.0

        # Styles for labels            
        self.colors = {0: "blue", 1: "red", 2: "orange"}
        self.colors.update(colors)

        self.markers = {0: "s", 1: "^", 2: "o"}
        self.markers.update(markers)
            
        if Dnu_max < Dnu_min:
            raise ValueError("Maximum range for Dnu can not be less than minimum")

        if self.plot_period:
            if DP_max < DP_min:
                raise ValueError("Maximum range for DP can not be less than minimum")

        if smooth_filter_width < 1:
            raise ValueError("The smooth filter width can not be less than 1!")

        self.freq = freq
        self.power = power

        if self.plot_period:
            self.period = period
            self.p_power = period_power

    #==================================================================
    # Data preparation
    #==================================================================
        if smooth:
            self.power = EchellePlotter.smooth_power(self.power, smooth_filter_width)

        self.update_echelle()
        if self.plot_period:
            # Minimum frequency is maximum period
            if pmin == None:
                self.pmin = self.period[0]
            else:
                if pmin < self.period[0]:
                    raise Exception("pmin provided exceeds the range of given period")
                self.pmin = pmin

            if pmax == None:
                self.pmax = self.period[-1]
            else:
                if pmax > self.period[-1]:
                    raise Exception("pmax provided exceeds the range of given period")
                self.pmax = pmax

            self.update_period_echelle()

    #==================================================================
    # Plotting
    #==================================================================
        # Create subplot(s)
        if self.plot_period:
            self.fig, self.axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
            self.ax = self.axs[0]
            self.pax = self.axs[1]

            # Set period y labels to the right
            self.pax.yaxis.set_label_position("right")
            self.pax.yaxis.tick_right()

        else:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)

        if self.plot_period:
            plt.subplots_adjust(left=0.20, right=0.85, bottom=0.25, wspace=0.05)
        else:
            plt.subplots_adjust(left=0.20, bottom=0.25)

        # Step in x-axis as median difference
        if step is None:
            step = np.median(np.diff(self.freq))
        self.step = step

        # Create widgets
        self.image = self.create_image(self.ax, self.x, self.y, self.z, 
            cmap, interpolation,
            xlabel=u"Frequency mod \u0394\u03BD", ylabel="Frequency (\u03BCHz)")
        self.create_Dnu_slider(Dnu_min, Dnu_max, step)

        if self.plot_period:
            # Invert axis for period plot
            self.pax.invert_yaxis()

            self.pimage = self.create_image(self.pax, self.px, self.py, self.pz, 
                cmap, interpolation,
                xlabel=u"Period mod \u0394P", ylabel="Period (s)")

            self.set_pextent()
            self.create_DP_slider(DP_min, DP_max, pstep)

    #==================================================================
    # Labelling point and saving
    #==================================================================
        self.create_label_radio_buttons()
        # self.create_remove_label_button()

        # A list of l-mode values (used for removing points)
        self.labels = 0

        # Dictionary of lists, with index being l-mode label
        # e.g. label_[1][2] is 3rd frequency for l=1 mode label
        self.f_labels = {0:[], 1:[], 2:[]}

        # 3 scatter plots corresponding to l-mode labels
        self.create_label_scatters()
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.create_save_button()
        self.create_load_button()


#======================================================================
# Echelle diagram functionality
#======================================================================
    def create_image(self, ax, x, y, z, cmap, interpolation, xlabel="", ylabel=""):
        """Create the echelle image"""
        image = ax.imshow(
            z,
            aspect="auto",
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin="lower",
            cmap=cmap,
            interpolation=interpolation,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return image

    def create_Dnu_slider(self, Dnu_min, Dnu_max, step):
        """Create Slider that adjusts Dnu"""
        # Frequency adjust axes
        if self.plot_period:
            axfreq = plt.axes([0.20, 0.1, 0.25, 0.03])
        else:
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

    def create_DP_slider(self, DP_min, DP_max, step):
        """Create Slider that adjusts Dnu"""
        # Period adjust axes
        axperiod = plt.axes([0.60, 0.1, 0.25, 0.03])

        # Value format string
        valfmt = "%1." + str(len(str(step).split(".")[-1])) + "f"

        # Slider and event calls
        self.pslider = Slider(
            axperiod,
            u"\u0394P",
            DP_min,
            DP_max,
            valinit=(DP_min + DP_max) / 2.0,
            valstep=step,
            valfmt=valfmt,
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.pslider.on_changed(self.pupdate)

    def update_echelle(self):
        """Get new period echelle"""
        self.x, self.y, self.fmap, self.z = EchellePlotter.echelle(self.freq, self.power, self.Dnu, 
            sampling=1, fmin=self.fmin, fmax=self.fmax)
        # Scale image intensities
        if self.scale == "sqrt":
            self.z = np.sqrt(self.z)
        elif self.scale == "log":
            self.z = np.log10(self.z)

    def update_period_echelle(self):
        """Get new period echelle"""
        # Ascending period to feed into echelle
        self.px, self.py, self.pmap, self.pz = EchellePlotter.echelle(self.period, self.p_power, 
            self.DP, sampling=1, fmin=self.pmin, fmax=self.pmax)

        # Flip the y-axis
        self.pz = np.flip(self.pz, 0)

        # Scale image intensities
        if self.scale == "sqrt":
            self.pz = np.sqrt(self.pz)
        elif self.scale == "log":
            self.pz = np.log10(self.pz)

    def update(self, Dnu, shift=False):
        """Updates frequency echelle diagram given new Dnu"""
        self.Dnu = Dnu
        self.update_echelle()
        self.image.set_array(self.z)

        if shift:
            self.set_extent()

        # Shift labelled points accordingly
        self.update_labels()

        # Render
        self.fig.canvas.blit(self.ax.bbox)

    def pupdate(self, DP, shift=False):
        """Updates period echelle diagram given new DP"""
        self.DP = DP

        # Calculate new data
        self.update_period_echelle()
        self.pimage.set_array(self.pz)

        # Set new visible range for x and y axes (reverted)
        # self.pimage.set_extent((self.px.max(), self.px.min(), self.py.max(), self.py.min()))
        # self.pax.set_xlim(self.DP, 0)

        if shift:
            self.set_pextent()

        # Shift labelled points accordingly
        self.update_labels()

        # Render
        self.fig.canvas.blit(self.pax.bbox)

    def on_key_press(self, event):
        """Key press to shift slider left and right
        or to access tool bar faster"""
        key = event.key.lower()
        if key == "left" or key == "right":
            if key == "left":
                new_Dnu = self.slider.val - self.slider.valstep
            else:
                new_Dnu = self.slider.val + self.slider.valstep
            self.slider.set_val(new_Dnu)
            self.update(new_Dnu, shift=True)

        elif self.plot_period and (key == "h" or key == "l"):
            if key == "h":
                new_DP = self.pslider.val - self.pslider.valstep
            elif key == "l":
                new_DP = self.pslider.val + self.pslider.valstep

            self.pslider.set_val(new_DP)
            self.pupdate(new_DP, shift=True)

        # Select remove tool
        elif key == 'r':
            self.radio_l.set_active(4)

        # Select l mode label tool
        elif key.isdigit():
            self.radio_l.set_active(int(key)+1)

        # Unselect tool
        elif key == 'escape':
            self.radio_l.set_active(0)

#======================================================================
# Point and line Labelling
#======================================================================
    def create_label_scatters(self):
        """ Scatter plots for l-mode labelling """
        self.scatters = [None, None, None]

        if self.plot_period:
            self.pscatters = [None, None, None]

    def create_label_radio_buttons(self):
        """Create RadioButtons to select labelling"""
        axcolor = 'white'
        rax = plt.axes([0.02, 0.7, 0.08, 0.15], facecolor=axcolor)
        self.radio_l = RadioButtons(rax, ('', '$\ell=0$', '$\ell=1$', '$\ell=2$', 'remove'))

    def create_save_button(self):
        """Press to call export_points"""
        ax_save = plt.axes([0.02, 0.4, 0.08, 0.08])
        self.save_button = Button(
            ax_save,
            "Save",)
        self.save_button.on_clicked(self.save_button_clicked)

    def create_load_button(self):
        """Press to chose file and call import_points"""
        ax_save = plt.axes([0.02, 0.2, 0.08, 0.08])
        self.load_button = Button(
            ax_save,
            "Load",)
        self.load_button.on_clicked(self.load_button_clicked)

    def create_remove_label_button(self):
        """Create Button to remove last label [NOT USED YET]"""
        axcolor = 'white'
        rax = plt.axes([0.02, 0.4, 0.08, 0.15], facecolor=axcolor)
        self.undo_point_button = Button(rax, "\u27f2 Undo")
        self.undo_point_button.on_clicked(self.remove_previous_label)

    def on_click(self, event):
        """Clicking event"""
        # Only add label click in the echelle, and selected l mode to label
        click_in_f_plot = event.inaxes == self.ax

        if self.plot_period:
            click_in_p_plot = event.inaxes == self.pax
        else:
            click_in_p_plot = False

        l_mode = self.radio_l.value_selected

        if not (click_in_f_plot or click_in_p_plot) or l_mode == "":
            return

        # Coordinate of clicks on the image array
        x, y = event.xdata, event.ydata

        # Remove current selected frequency
        if l_mode == "remove":
            if click_in_f_plot:
                self.remove_point(x, y)

            elif click_in_p_plot:
                self.remove_point(x, y, period=True)

        elif click_in_f_plot:
            l = self.get_l_mode_choice(l_mode)
            self.add_point(x, y, l)


        elif click_in_p_plot:
            # Point labelling
            l = self.get_l_mode_choice(l_mode)
            self.add_point(x, y, l, period=True)

    def add_point(self, x, y, l, period=False):
        """Add point to the plot and update scatter"""
        if period:
            # Find the left x and down y value in self.x and self.y
            nearest_x = self.px[self.px-x < 0].max()
            nearest_y = self.py[self.py-y < 0].max()
            f = self.coord2freq(nearest_x, nearest_y, period=True)

        else:
            # Find the nearest x and y value in self.px and self.py
            nearest_x = self.x[self.x-x < 0].max()
            nearest_y = self.y[self.y-y < 0].max()
            f = self.coord2freq(nearest_x, nearest_y)

        self.f_labels[l].append(f)
        self.update_scatter(l)
        self.labels += 1

    def remove_point(self, x0, y0, period=False):
        """Removes point with given frequency"""
        if self.labels == 0:
            return

        # Find the label close to this frequency
        min_diff = 1000

        min_l = 0
        min_i = 0
        for l, ls in self.f_labels.items():
            for i, freq in enumerate(ls):
                # Calculate Euclidean distance on plot
                if period:
                    x, y = self.p2coord(self.f2p(freq))
                else:
                    x, y = self.f2coord(freq)

                diff = np.sqrt((x-x0)**2 + (y-y0)**2)
                if diff < min_diff:
                    min_diff = diff
                    min_l = l
                    min_i = i

        # Does not have corresponding frequency label
        if (not period and min_diff > 0.1) or (period and min_diff > 100):
            return

        # Remove label
        self.f_labels[min_l].pop(min_i)
        self.update_scatter(min_l)
        self.labels -= 1

    def set_extent(self):
        """Set new visible range for x and y axes of frequency echelle"""
        self.image.set_extent((self.x.min(), self.x.max(), self.y.min(), self.y.max()))
        # self.ax.set_xlim(0, self.Dnu)

    def set_pextent(self):
        """Set new visible range for y axis of period echelle"""
        # Extent and xlim match inverted axes
        self.pimage.set_extent((self.px.min(), self.px.max(), self.py.max(), self.py.min()))
        # self.pax.set_xlim(0, self.DP)

    def update_labels(self):
        """Update the labels to new echelle diagram coordinates"""
        # Replot
        l = 0
        while l < len(self.f_labels):
            self.update_scatter(l)
            l += 1

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
            
            if self.plot_period:
                self.pscatters[l].remove()
                self.pscatters[l] = None


        # First label of the mode
        elif has_label_no_scatter:
            # Update scatter plots based on l value
            color = self.colors[l]
            marker = self.markers[l]
            label = f"l={l}"

            # Get frequency coordinates
            x_labels, y_labels = self.get_coords(self.f_labels[l], self.Dnu)
            self.scatters[l] = self.ax.scatter(x_labels, y_labels, 
                s=self.size, marker=marker, label=label, color=color)

            # Period coordinates
            if self.plot_period:
                # Plot Line
                if l in self.plot_line:
                    # Connect dots in ascending frequency order
                    ordered_f = np.sort(self.f_labels[l])
                    x_line, y_line = self.get_pcoords(np.array(ordered_f), self.DP)

                    # Only gets plot when expanded as tuple
                    self.pscatters[l], = self.pax.plot([],[], "--",
                        color=self.colors[l], marker=self.markers[l], label=label)
                    self.update_line(l)

                # Scatter plot
                else:
                    px_labels, py_labels = self.get_pcoords(self.f_labels[l], self.DP)
                    self.pscatters[l] = self.pax.scatter(px_labels, py_labels, 
                        s=50, marker=marker, label=label, color=color)

        else:
            x_labels, y_labels = self.get_coords(self.f_labels[l], self.Dnu)
            self.scatters[l].set_offsets(np.c_[x_labels, y_labels])
            
            if self.plot_period:
                if l in self.plot_line:
                    self.update_line(l)
                else:                
                    px_labels, py_labels = self.get_pcoords(self.f_labels[l], self.DP)
                    self.pscatters[l].set_offsets(np.c_[px_labels, py_labels])

        # Draw legend if there are labels
        self.ax.legend()
        if self.plot_period:
            self.pax.legend()

        self.fig.canvas.draw()

    def update_line(self, l):
        """Update line"""
        ordered_f = np.sort(self.f_labels[l])
        x_line, y_line = self.get_pcoords(np.array(ordered_f), self.DP)
        self.pscatters[l].set_data(x_line, y_line)
        self.fig.canvas.draw()

    def get_l_mode_choice(self, choice):
        if choice == '$\ell=0$':
            return 0
        elif choice == '$\ell=1$':
            return 1
        elif choice == '$\ell=2$':
            return 2
        return -1

    def get_coords(self, freqs, Dnu):
        """From frequency to frequency coordinates"""
        xs = []
        ys = []
        for f in freqs:
            x, y = self.f2coord(f)
            xs.append(x)
            ys.append(y)

        return xs, ys
    
    def get_pcoords(self, freqs, DP):
        """From frequency to period coordinates"""
        xs = []
        ys = []
        for f in freqs:
            p = self.f2p(f)
            x, y = self.p2coord(p)
            xs.append(x)
            ys.append(y)

        return xs, ys

    def f2coord(self, freq):
        """Frequency to coordinate on the Echelle"""
        r, c = EchellePlotter.unravel_nearest_index(self.fmap, freq)
        # To line up with the middle of the pixel
        # Increment both x and y by half the sampling width
        x_inc = self.x[1] - self.x[0]
        y_inc = self.y[1] - self.y[0]
        return self.x[c] + 0.5*x_inc, self.y[r] + 0.5*y_inc

    def p2coord(self, period):
        """Period to coordinate on the Echelle"""
        r, c = EchellePlotter.unravel_nearest_index(self.pmap, period)
        # To line up with the middle of the pixel
        # Increment both x and y by half the sampling width        
        x_inc = self.px[1] - self.px[0]
        y_inc = self.py[1] - self.py[0]
        return self.px[c] + 0.5*x_inc, self.py[r] + 0.5*y_inc

    def coord2freq(self, x, y, period=False):
        if period:
            return self.p2f(y + x)

        return x + y

#======================================================================
# Utilities
#======================================================================
    def show(self):
        """Show plot using plt.show()"""
        plt.show()

    def savefig(self, *args, **kwargs):
        """Save plot using plt.savefig()"""
        plt.savefig(*args, **kwargs)

    def f2p(self, freq):
        """From frequency (muHz) to period (s)"""
        return 1e6/freq

    def p2f(self, period):
        """From period (s) to frequency (muHz)"""
        return 1e6/period

    def unravel_nearest_index(array, value):
        """Returns the indices as tuples where value is closest in array"""
        array = np.asarray(array)
        array = np.abs(array - value)
        idx = np.unravel_index(np.argmin(array, axis=None), array.shape)
        return idx

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

        # Take a range of frequencies separated by the samplinginterval
        # and use linear interpolation to estimate power values of those frequencies
        xp = np.arange(fmin, fmax + Dnu, samplinginterval)
        yp = np.interp(xp, freq, power)

        # Number of stacks and Number of elements in each stack
        n_stack = int((fmax - fmin) / Dnu)
        n_element = int(Dnu / samplinginterval)

        # image as 2D array
        freqs = np.zeros([n_stack, n_element])
        pows = np.zeros([n_stack, n_element])

        # Add yp values to rows of image
        for i in range(n_stack):
            freqs[i, :] = xp[n_element * (i) : n_element * (i + 1)]
            pows[i, :] = yp[n_element * (i) : n_element * (i + 1)]

        # Construct x-y coordinates and provide endpoints (last value + increment)
        xn = freqs[0]-freqs[0,0]
        xn = np.append(xn, xn[-1] + xn[1]-xn[0])
        yn = freqs[:,0]
        yn = np.append(yn, yn[-1] + yn[1]-yn[0])

        return xn, yn, freqs, pows

    def smooth_power(power, smooth_filter_width):
        """Smooths the input power array with a Box1DKernel from astropy

        Parameters
        ----------
        power : array-like
            Array of power values
        smooth_filter_width : float
            filter width

        Returns
        -------
        array-like
            Smoothed power
        """
        return convolve(power, Box1DKernel(smooth_filter_width))

    def load_button_clicked(self, events):
        """Choose file to load label points"""
        tkinter.Tk().withdraw()
        filename = askopenfilename()
        self.import_points(filename)
        for l in range(3):
            self.update_scatter(l)

    def import_points(self, filename):
        """Read f_labels from json file"""
        # File chooser was cancelled
        if filename == "":
            return

        with open(filename, "r") as f:
            data = json.load(f)

        # Assign values to the dictionary
        # f_labels have integers as key, but json reads keys as strings
        self.f_labels[0] = data['0']
        self.f_labels[1] = data['1']
        self.f_labels[2] = data['2']

    def save_button_clicked(self, events):
        """Wrapper for export_points when button clicked"""
        self.export_points()

        tkinter.Tk().withdraw()
        tkinter.messagebox.showinfo("EchellePlotter", "Points saved!")

    def export_points(self, filename="labelled_points.json"):
        """Export all labelled points to a json file"""
        with open(filename, "w") as f:
            json.dump(self.f_labels, f)

if __name__ == "__main__":
    #-----------------------------------
    # Produce power spectrum
    #-----------------------------------
    ps_df = pd.read_csv("data/11502092_PS.csv", sep='\t', names=['freq', 'amp'])
    freq = np.array(ps_df.freq)
    amp = np.array(ps_df.amp)

    Dnu = 5 # large frequency separation (muHz)
    fmin = 15
    fmax = 45

    # Change to period
    DP = 295.6 # period spacing

    #-----------------------------------
    # Produce periodogram
    #-----------------------------------
    period_df = pd.read_csv("data/11502092_Period.csv")
    equal_period = np.array(period_df.period)
    equal_period_amp = np.array(period_df.amp)

    e = EchellePlotter(freq, amp, Dnu_min=Dnu-3, Dnu_max=Dnu+3, step=.05,
        fmin=fmin, fmax=fmax,
        period=equal_period, period_power=equal_period_amp,
        DP_min=DP-10, DP_max=DP+10, pstep=0.1,
        colors={0:"red", 1:"blue", 2:"red"},
        markers={0: "o", 1:"^", 2:"s"},
        plot_line=[1])
    e.show()


