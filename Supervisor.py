#!/usr/bin/env python
from Tkinter import *
import ttk
import socket
import json
from threading import Thread, Lock
from time import sleep, time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from Scrapers import ScrapeTPV, ScrapeFiat, SingleBrowser, PATH
from Common import Epoch, CustomizedNavigationToolbar2Tk, Destroyer, Rsu, adjust_margins, sci_round
from TradingAlgorithms import AmplitudePhaseAlgo, APAlgoStub, PerfectTradeAlgo, PTAlgoStub


class Shared(object):
    """Data shared between multiple processes"""
    def __init__(self, coins):
        self.prices = dict.fromkeys(coins)
        self._price_locks = {key: Lock() for key in coins}

        self.usdrate = None
        self._usdrate_lock = Lock()

        algotypes = ('AP', 'PT', 'both')
        self.decisions = {algotype: dict.fromkeys(coins) for algotype in algotypes}
        self._dec_locks = {algotype: {coin: Lock() for coin in coins} for algotype in algotypes}

    def ping(self): return True

    # Coin prices
    def set_price(self, coin, price, lpu_time):
        with self._price_locks[coin]: self.prices[coin] = (price, lpu_time)

    def get_price(self, coin):
        with self._price_locks[coin]: return self.prices[coin]

    # USD rate
    def set_usdrate(self, usdrate):
        with self._usdrate_lock: self.usdrate = usdrate

    def get_usdrate(self):
        with self._usdrate_lock: return self.usdrate

    # Decisions (sell/buy)
    def set_decision(self, coin, algotype, decision):
        with self._dec_locks[algotype][coin]: self.decisions[algotype][coin] = (decision, int(time()))

    def get_decision(self, coin, algotype):
        with self._dec_locks[algotype][coin]: return self.decisions[algotype][coin]


class ServerThread(Thread, object):
    """docstring for ThreadedTrading"""
    def __init__(self, coins):
        super(ServerThread, self).__init__()
        self.shared_obj = Shared(coins)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('localhost', 10001))

    def run(self):
        while True:
            data, address = self.sock.recvfrom(4096)
            req = json.loads(data)
            func = getattr(self.shared_obj, req[0]) # Get desired function
            sent = self.sock.sendto(json.dumps(func(*req[1:])), address) # Pass arguments to the function, execute it and send the response


class SupervisorApp(Tk, Rsu, Destroyer):
    """docstring for SupervisorApp"""
    def __init__(self, forder_hist, folder_db, coins_params):
        super(SupervisorApp, self).__init__()
        coins = tuple([coin for coin, _, _ in coins_params])

        # Window setup
        self.protocol("WM_DELETE_WINDOW", self.destroyer)
        self.title('SUPERVISOR')
        self.geometry('1000x600')
        self.resizable(True, True)
        
        # Create widgets
            # Menu bar
        self.var_w = IntVar(self, 0)
        menubar = Menu(self, relief=FLAT, font=('Helvetica', 10), bg='gray88')
        viewmenu = Menu(menubar, tearoff=0)
        for idx, coin in enumerate(coins): viewmenu.add_radiobutton(label=coin, variable=self.var_w, value=idx, command=self.switch_view)
        menubar.add_cascade(label="View", menu=viewmenu)
        self.config(menu=menubar)

            # Frames
        main_frame = Frame(self)
        main_frame.pack(padx=4, side=LEFT, fill=BOTH, expand=True)

        tabs_frame = Frame(main_frame)
        tabs_frame.pack(side=LEFT, fill=Y)

            # Global data frame
        cmc_frame = ttk.LabelFrame(tabs_frame, labelwidget=Label(tabs_frame, text='Global Data', font=('Times', 10, 'italic')), labelanchor=N)
        cmc_frame.pack(padx=2, pady=2, fill=X)

        self.lbl_cut = Label(cmc_frame)
        self.lbl_cut.grid(row=0, column=0, padx=2, pady=2, sticky=E)
        self.lbl_cui = Label(cmc_frame)
        self.lbl_cui.grid(row=0, column=1, padx=2, pady=2, sticky=W)

        Label(cmc_frame, text='Price, USD').grid(row=1, column=0, padx=2, pady=2, sticky=E)
        self.lbl_cmc = Label(cmc_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_cmc.grid(row=1, column=1, padx=2, pady=2, sticky=W)

        cmc_frame.grid_columnconfigure(0, weight=1)
        cmc_frame.grid_columnconfigure(1, weight=0)

            # AlgoAP frame
        aap_frame = ttk.LabelFrame(tabs_frame, labelwidget=Label(tabs_frame, text='Algorithm AP', font=('Times', 10, 'italic')), labelanchor=N)
        aap_frame.pack(padx=2, pady=2, fill=X)

        Label(aap_frame, text='Rank * 1000').grid(row=0, column=0, padx=2, pady=2, sticky=E)
        self.lbl_apr = Label(aap_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_apr.grid(row=0, column=1, padx=2, pady=2, sticky=W)
                
        Label(aap_frame, text='Decision').grid(row=1, column=0, padx=2, pady=2, sticky=E)
        self.lbl_apd = Label(aap_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_apd.grid(row=1, column=1, padx=2, pady=2, sticky=W)

        aap_frame.grid_columnconfigure(0, weight=1, pad=10)
        aap_frame.grid_columnconfigure(1, weight=0)

            # AlgoPT frame
        apt_frame = ttk.LabelFrame(tabs_frame, labelwidget=Label(tabs_frame, text='Algorithm PT', font=('Times', 10, 'italic')), labelanchor=N)
        apt_frame.pack(padx=2, pady=2, fill=X)

        Label(apt_frame, text='Rank * 1000').grid(row=0, column=0, padx=2, pady=2, sticky=E)
        self.lbl_ptr = Label(apt_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_ptr.grid(row=0, column=1, padx=2, pady=2, sticky=W)
                
        Label(apt_frame, text='Decision').grid(row=1, column=0, padx=2, pady=2, sticky=E)
        self.lbl_ptd = Label(apt_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_ptd.grid(row=1, column=1, padx=2, pady=2, sticky=W)

        apt_frame.grid_columnconfigure(0, weight=1, pad=10)
        apt_frame.grid_columnconfigure(1, weight=0)

            # Fiat frame
        fiat_frame = ttk.LabelFrame(tabs_frame, labelwidget=Label(tabs_frame, text='Fiat Rates', font=('Times', 10, 'italic')), labelanchor=N)
        fiat_frame.pack(padx=2, pady=2, fill=X)

        Label(fiat_frame, text='USD-UAH rate').grid(row=0, column=0, padx=2, pady=2, sticky=E)
        self.lbl_dh = Label(fiat_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_dh.grid(row=0, column=1, padx=2, pady=2, sticky=W)

            # Plots frame
        plots_frame = Frame(main_frame)
        plots_frame.pack(side=LEFT, fill=BOTH, expand=True)

        globplot_frame = ttk.LabelFrame(plots_frame, labelwidget=Label(plots_frame, text='Historical data', font=('Times', 10, 'italic')), labelanchor=N)
        globplot_frame.pack(padx=2, pady=2, fill=BOTH, expand=True)

                # Matplotlib
        self.fig, (self.ax0, self.ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        self.data0 = self.ax0.plot([], [], color='b', linewidth=1)[0]
        self.data1 = self.ax1.fill_betweenx([], [], color='dimgrey')
        self.ax0.set_ylabel('Price, USD', fontsize='small')
        self.ax1.set_ylabel('24h Vol, billions USD', fontsize='small')
        self.ax0.label_outer()
        self.ax1.label_outer()
        self.canvas = FigureCanvasTkAgg(self.fig, master=globplot_frame)
        self.canvas.figure.left_margin = 0.8
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, padx=3, fill=BOTH, expand=1)
        self.toolbar = CustomizedNavigationToolbar2Tk(self.canvas, globplot_frame)
        self.toolbar.zoom_flag = False
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.canvas.mpl_connect("resize_event", lambda event:adjust_margins(self.fig,event,l=0.7))

        # Launch the server thread
        self.srv_thr = ServerThread(coins)
        self.srv_thr.daemon = True
        self.srv_thr.start()

        # Initialize instances
        self.epoch = Epoch(reftime, tinth)
        self.insts = []
        for coin, paramfile_algoap, paramfile_algopt in coins_params:
            tpv = ScrapeTPV(coin, forder_hist, folder_db)
            algoap = AmplitudePhaseAlgo(paramfile_algoap, coin, tpv.p_s, folder_db) if paramfile_algoap else APAlgoStub()
            algopt = PerfectTradeAlgo(paramfile_algopt, coin, folder_db, tpv.p_s) if paramfile_algopt else PTAlgoStub()
            tpv.cut_data()
            self.set_shared_data(tpv, algoap, algopt, coin)
            self.insts.append((tpv, algoap, algopt, coin))
        self.fiat = ScrapeFiat(folder_db)
        self.srv_thr.shared_obj.set_usdrate(self.fiat.get_usduah_rate())

        last_epoch = max([self.epoch(inst[0].t_s[-1]) for inst in self.insts])
        for tpv, algoap, algopt, _ in self.insts:
            if self.epoch(tpv.t_s[-1]) < last_epoch:
                tpv.update()
                algoap.append(tpv.p_s[-1:])
                algopt(tpv.p_s[-1:])
        self.threads = [Thread()] * len(coins) # Thread stub for switch_view()
        self.pending_w = None # ID of pending switch_view() call
        self.switch_view()
        self.rsu()

        print('! Epoch {}, {}'.format(last_epoch, datetime.fromtimestamp(self.tpv.t_s[-1])))
        sleeptime = max(0, self.epoch.refts + (last_epoch + 1) * self.epoch.avedur + timeshift - time())
        print('sleeptime', sleeptime)
        self.after(int(sleeptime * 1000), self.scraping_loop_part_a)

    def scraping_loop_part_a(self):
        """HTTP requests are executed here"""
        print('Epoch {}, {}'.format(self.epoch(), datetime.fromtimestamp(time())))
        self.threads = [Thread(target=lambda args=item:self.tpv_loopfunc(*args)) for item in self.insts] + [Thread(target=self.fiat_loopfunc)]
        for thread in self.threads: thread.start() # Start all threads
        self.after(1000, self.scraping_loop_part_b)
 
    def scraping_loop_part_b(self):
        """Part of loop for GUI functions"""
        if any([thread.is_alive() for thread in self.threads]): self.after(1000, self.scraping_loop_part_b)
        else:
            self.rsu()
            sleeptime = self.epoch.avedur - (time() - self.epoch.refts - timeshift) % self.epoch.avedur
            print('sleeptime', sleeptime)
            self.after(int(sleeptime * 1000), self.scraping_loop_part_a)

    def tpv_loopfunc(self, tpv, algoap, algopt, coin):
        """HTTP requests, algorithms, data sharing"""
        tpv.update()
        algoap.append(tpv.p_s[-1:])
        algopt(tpv.p_s[-1:])
        if tpv.next_update <= time(): algoap.replace(tpv.replace())
        self.set_shared_data(tpv, algoap, algopt, coin)

    def fiat_loopfunc(self): self.srv_thr.shared_obj.set_usdrate(self.fiat.get_usduah_rate())

    def set_shared_data(self, tpv, algoap, algopt, coin):
        """Data sharing via socket"""
        self.srv_thr.shared_obj.set_price(coin, tpv.p_s[-1], tpv.t_s[-1])
        self.srv_thr.shared_obj.set_decision(coin, 'AP', algoap.trade_decision())
        self.srv_thr.shared_obj.set_decision(coin, 'PT', algopt.decision)
        self.srv_thr.shared_obj.set_decision(coin, 'both', algoap.decision and algopt.decision)

    # Matplotlib functions
    def on_key_press(self, event):
        key_press_handler(event, self.canvas, self.toolbar)

    def update_plot(self):
        time = np.array([datetime.fromtimestamp(t) for t in self.tpv.t_s])
        self.data0.set_xdata(time)
        self.data0.set_ydata(self.tpv.p_s)
        self.data1.remove()
        self.data1 = self.ax1.fill_between(time, self.tpv.v_s * 1e-9, linewidth=0, color='dimgrey')

        # Calc limits
        y_min, y_max = np.min(self.tpv.p_s), np.max(self.tpv.p_s)
        y_margin = (y_max - y_min) * 0.05
        self.ax0.mem_xlim = (time[0], time[-1] + timedelta(days=1))
        self.ax0.mem_ylim = (y_min - y_margin, y_max + y_margin)
        vols_srtd = np.sort(self.tpv.v_s)
        minvol, maxvol = vols_srtd[0], vols_srtd[-1]
        slope = (maxvol - minvol) / self.tpv.minsize
        elbow_idx = np.argmax(minvol + slope * np.arange(self.tpv.minsize) * 2 - vols_srtd)
        elbow_val = vols_srtd[elbow_idx]
        use_elbow = (elbow_val / maxvol) * (1 - float(elbow_idx) / self.tpv.minsize) <= 10e-3
        self.ax1.mem_ylim = (0, (elbow_val if use_elbow else np.max(self.tpv.v_s)) * 1.05e-9)

        # Set view
        if self.toolbar.zoom_flag is False: # If not zooming,
            self.ax0.set_xlim(self.ax0.mem_xlim)
            self.ax0.set_ylim(self.ax0.mem_ylim)
            self.ax1.set_ylim(self.ax1.mem_ylim)

        self.canvas.draw()

    # Appearance funcs
    def refresh(self):
        usd_uah_rate = self.fiat.true_usdrate
        price = self.tpv.p_s[-1]

        rank = self.algoap.rank
        decision = self.algoap.decision
        self.lbl_cmc['text'] = sci_round(price, 5)
        self.lbl_apr['text'] = 'None' if rank is None else round(rank * 1000, 5)
        self.lbl_apd['text'] = 'None' if decision is None else ('Buy' if decision is True else 'Sell')

        rankpt = self.algopt.rank
        decisionpt = self.algopt.decision
        self.lbl_ptr['text'] = 'None' if rankpt is None else round(rankpt * 1000, 5)
        self.lbl_ptd['text'] = 'None' if decisionpt is None else ('Buy' if decisionpt is True else 'Sell')

        self.lbl_dh['text'] = round(usd_uah_rate, 2)

    # Button calls
    def switch_view(self):
        if self.pending_w is not None: self.after_cancel(self.pending_w)
        view_idx = self.var_w.get()

        if self.threads[view_idx].is_alive(): self.pending_w = self.after(500, self.switch_view) # If the data is under processing, wait and call later
        else:
            # Change flags and current object
            self.view_idx = view_idx
            self.toolbar.zoom_flag = False
            self.tpv, self.algoap, self.algopt, self.coin = self.insts[self.view_idx]
    
            # Ubdate displaying of graphical and numerical data
            self.lbl_cut['text'] = self.coin
            self.lbl_cui['image'] = self.tpv.tkimage
            self.update_plot()
            self.refresh()

    # Other functions
    def save_sql(self, commit=True):
        for tpv, algoap, algopt, _ in self.insts: 
            tpv.db_update(commit)
            algoap.db_update(commit)
            algopt.db_update(commit)

    def destroyer(self):
        if any([thread.is_alive() for thread in self.threads]): self.after(1000, self.destroyer)
        else:
            browser = SingleBrowser(executable_path=PATH)
            browser.quit()
            super(SupervisorApp, self).destroyer()


if __name__ == '__main__':
    # Settings
    coins_params = (('ETH', 'ParamsAlgoAP/ETH/params24.npz', 'ParamsAlgoPT/ETH/params3.npz'),
                    ('XMR', 'ParamsAlgoAP/XMR/params25.npz', None),
                    ('USDT', None, None),
                    ('BTC', 'ParamsAlgoAP/BTC/params26.npz', 'ParamsAlgoPT/BTC/params5.npz')
    )
    reftime = '2020-7-22 0:14:0'
    tinth = 0.25
    timeshift = 1

    # Input files
    forder_hist = 'HistoricalData/'
    folder_db = 'DataBases/'

    # Fix ImportError: Failed to import _strptime because the import lockis held by another thread.
    datetime.strptime('20200101','%Y%m%d') # Dummy call

    gui = SupervisorApp(forder_hist, folder_db, coins_params)
    gui.mainloop()
