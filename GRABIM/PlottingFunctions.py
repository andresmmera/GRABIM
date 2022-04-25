# Functions for ladder circuit analysis

from GRABIM import LadderCircuitAnalysis as Ladder_ckt
from GRABIM import TwoPortMatrixConversions as TP
import numpy as np

# Plot (external libraries)
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import Legend, LegendItem
from bokeh.models import Arrow, NormalHead
from bokeh.models import ColumnDataSource, LabelSet, Text

def Plot_S21_S11_dB(ZS, ZL, code, x, freq, plotw = 0):
    # Calculate T and S parameters
    T = Ladder_ckt.get_ABCD_Network(ZS, ZL, code, x, freq); # ABCD matrix calculation
    S = TP.TtoS(T, ZS, ZL); # S-parameter matrix conversion

    plot = figure(plot_width=800, plot_height=400, title='Response')
    
    if (plotw == 1):
        plot.line(freq*2*np.pi, 20*np.log10(np.abs(S[:, 0, 0])), line_width=2, color="navy", legend_label="S11")
        plot.line(freq*2*np.pi, 20*np.log10(np.abs(S[:, 1, 0])), line_width=2, color="red", legend_label="S21")
        plot.xaxis.axis_label = 'w (rad/s)';
    else:
        plot.line(freq*1e-6, 20*np.log10(np.abs(S[:, 0, 0])), line_width=2, color="navy", legend_label="S11")
        plot.line(freq*1e-6, 20*np.log10(np.abs(S[:, 1, 0])), line_width=2, color="red", legend_label="S21")
        plot.xaxis.axis_label = 'frequency (MHz)';
        
    plot.yaxis.axis_label = 'Response (dB)';
    plot.legend.location = 'bottom_right';
    show(plot)
    
    
def Plot_S11_nu(ZS, ZL, code, x, freq):    
    
    # Calculate the reflection coefficient
    rho = Ladder_ckt.get_Input_Reflection_Coeff(ZS, ZL, code, x, freq)
    
    plot = figure(plot_width=800, plot_height=400, title='Reflection coefficient')

    plot.line(freq*1e-6, np.abs(rho), line_width=2, color="navy", legend_label="S11")
    plot.xaxis.axis_label = 'frequency (MHz)';
    plot.yaxis.axis_label = 'Reflection coefficient';
    plot.legend.location = 'bottom_right';
    show(plot)