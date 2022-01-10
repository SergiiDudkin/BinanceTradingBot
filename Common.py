import math
import sqlite3
from Queue import Queue
from time import time
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import socket
import json
import platform


# Classes
class Epoch(object):
    """Calculates an epoch from timestamp and stores reference timestamp with period"""
    def __init__(self, reftime, tinth):
        try: self.refts = int(datetime.strptime(reftime, '%Y-%m-%d %H:%M:%S').strftime('%s'))
        except: self.refts = (datetime.strptime(reftime, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1, 3)).total_seconds()
        self.avedur = tinth * 3600 # Auto trading period

    def __call__(self, ts=None):
        return int(((int(time()) if ts is None else ts) - self.refts) // self.avedur)


# Base classes
class DbUpdate(object):
    """Base class containing function db_update"""
    def db_update(self, commit=True):
        """Transfers records from python queue to sql database"""
        while self.sqlqueue.qsize():
            table_data, param_tuple = self.sqlqueue.get()
            self.cur.execute('INSERT INTO {} VALUES ({});'.format(table_data, ('?, ' * len(param_tuple))[:-2]), param_tuple) # Insert raw
        if commit is True: self.con.commit()


class Destroyer(object):
    """Defines destroyer for GUI"""
    def destroyer(self):
        self.save_sql(commit=True)
        self.quit()
        self.destroy()


class Rsu(object):
    """Defines refresh, save_sql and update_plot function for GUI"""
    def rsu(self):
        self.refresh()
        self.save_sql()
        self.update_plot()


class SockCli(object):
    """Defines functions for socket client"""
    def sockinit(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(5)
        self.server_address = ('localhost', 10001)

    def get_shared_data(self, *args):
        sent = self.sock.sendto(json.dumps(args), self.server_address)
        return json.loads(self.sock.recvfrom(4096)[0])


# Subclasses
class CustomizedNavigationToolbar2Tk(NavigationToolbar2Tk, object):
    """NavigationToolbar2Tk was subclassed to fix problems with home view and preserve zoom view from regular updating."""
    def home(self):
        """Called when homeview button is clicked on. Restores home view"""
        super(CustomizedNavigationToolbar2Tk, self).home() # Inherit

        # Set limits
        self.canvas.figure.axes[0].set_xlim(*self.canvas.figure.axes[0].mem_xlim)
        self.canvas.figure.axes[0].set_ylim(*self.canvas.figure.axes[0].mem_ylim)
        self.canvas.figure.axes[1].set_ylim(*self.canvas.figure.axes[1].mem_ylim)

        adjust_margins(self.canvas.figure) # Fix figure margins
        self.zoom_flag = False # Reset zoom_flag

    def press_zoom(self, event):
        """Called when zooming to rectangle. Sets zoom_flag which preserves zoom from regular updating."""
        super(CustomizedNavigationToolbar2Tk, self).press_zoom(event) # Inherit
        self.zoom_flag = True


# Functions
def adjust_margins(fig, event=None, l=0.9, b=0.25, r=0.04, t=0.05):
    """Adjusts margins of matplotlib figure to be constant regardless of resize"""
    if hasattr(fig, 'left_margin'): l = fig.left_margin
    width, height = fig.get_size_inches()
    fig.subplots_adjust(left=l/width, bottom=b/height, right=1-r/width, top=1-t/height)

def sci_round(num, sig_fig=1, ndigits=1e6):
    """Round with precision sig_fig and max digits after coma ndigits"""
    return 0.0 if not num else round(num, min(sig_fig - int(math.floor(math.log10(abs(num) / 2.0))) - 1, ndigits))

def floor_str(num, prec):
    """Round function for Binance"""
    strnum = '{:.8f}'.format(num)
    return strnum[:strnum.index('.')+1+prec]

OP_SYS = platform.system()


if __name__ == '__main__':
    # epoch = Epoch('2020-7-22 0:14:0', 0.25)
    # print('refts {} {}'.format(epoch.refts, datetime.fromtimestamp(epoch.refts)))
    a = sci_round(None)
    print(a)

