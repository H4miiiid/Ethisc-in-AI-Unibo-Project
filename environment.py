import digitalhub as dh

try:
    from local import pio_renderer
except ImportError:
    ### DEFAULT CONFIGURATION
    pio_renderer = 'iframe_connected'
