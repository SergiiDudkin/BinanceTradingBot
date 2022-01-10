#!/usr/bin/env python
from Tkinter import *
import ttk
from Queue import Queue
import sqlite3
from threading import Thread
from time import sleep, time
from datetime import datetime, timedelta
import numpy as np
from bisect import bisect
import requests
import functools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from Common import DbUpdate, Epoch, CustomizedNavigationToolbar2Tk, Destroyer, Rsu, SockCli, adjust_margins, sci_round


def requestdecor(func):
    name = func.__name__
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(3):
            if i != 0: sleep(2)
            try:
                res = func(*args, **kwargs)
                res.raise_for_status()
                return res.json()
            except Exception as e: print('{} failed #{}. {}'.format(name, i, repr(e)))
    return wrapper

@requestdecor
def req_deals(symbol): return requests.get('https://btc-trade.com.ua/api/deals/{}'.format(symbol))

@requestdecor
def req_buy_list(symbol): return requests.get('https://btc-trade.com.ua/api/trades/buy/{}'.format(symbol))

@requestdecor
def req_sell_list(symbol): return requests.get('https://btc-trade.com.ua/api/trades/sell/{}'.format(symbol))

@requestdecor
def req_trades(symbol, limit=500): return requests.get('https://api.binance.com/api/v1/trades?limit={}&symbol={}'.format(limit, symbol))

@requestdecor
def req_depts(symbol, limit=500): return requests.get('https://api.binance.com/api/v1/depth?limit={}&symbol={}'.format(limit, symbol))


class BinanceMarket(DbUpdate):
    """BINANCE follower"""
    def __init__(self, symbol, sums, folder):
        self.symbol = symbol
        self.sums = sums
        self.folder = folder
        self.lensum = len(self.sums)
        self.new_deals = [] # Deals of the ongoing minute
        mulprstrt = (', p{} REAL' * self.lensum).format(*range(self.lensum))
        self.mulprstr = (', p{}' * self.lensum).format(*range(self.lensum))

        # SQL
        self.sqlqueue = Queue()
        self.con = sqlite3.connect('{}Binance_{}.db'.format(self.folder, self.symbol), detect_types=sqlite3.PARSE_DECLTYPES) # Create or connect database
        self.cur = self.con.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS deals (id INTEGER PRIMARY KEY, unixtime INT, average REAL, mean REAL, max REAL, min REAL, first REAL, last REAL, volume REAL);')
        self.cur.execute('CREATE TABLE IF NOT EXISTS spreads_buy (id INTEGER PRIMARY KEY, unixtime REAL, maxbuy REAL{});'.format(mulprstrt))
        self.cur.execute('CREATE TABLE IF NOT EXISTS spreads_sell (id INTEGER PRIMARY KEY, unixtime REAL, minsell REAL{});'.format(mulprstrt))

        # Fetch data from DBs
        self.last_deal_id = None
        self.date_lim = int(time() - timedelta(days=5).total_seconds())
        deals = self.cur.execute('SELECT unixtime, average, volume FROM deals WHERE unixtime > {}'.format(self.date_lim)).fetchall()
        self.agg_utsv, self.agg_prices, self.agg_vols = map(list, zip(*deals)) if deals else ([], [], [])
        self.last_minute = (self.agg_utsv[-1] - 30) / 60 + 1 if self.agg_utsv else 0
        self.spreads_b = np.array(zip(*self.cur.execute('SELECT unixtime{} FROM spreads_buy WHERE unixtime > {} ORDER BY unixtime'.format(self.mulprstr, self.date_lim)).fetchall()), dtype=np.float64)
        self.spreads_s = np.array(zip(*self.cur.execute('SELECT unixtime{} FROM spreads_sell WHERE unixtime > {} ORDER BY unixtime'.format(self.mulprstr, self.date_lim)).fetchall()), dtype=np.float64)
        if self.spreads_b.size == 0: self.spreads_b = np.empty((self.lensum + 1, 0))
        if self.spreads_s.size == 0: self.spreads_s = np.empty((self.lensum + 1, 0))
        self.max_buy = self.cur.execute('SELECT maxbuy FROM spreads_buy WHERE id = (SELECT MAX(id) FROM deals);').fetchone()
        self.min_sell = self.cur.execute('SELECT minsell FROM spreads_sell WHERE id = (SELECT MAX(id) FROM deals);').fetchone()

    def get_deals(self):
        curr_deals = req_trades(self.symbol)
        if curr_deals and self.last_deal_id is not None and curr_deals[0]['id'] > self.last_deal_id: curr_deals = req_trades(self.symbol, 1000) # Request more deals if there is no overlaping

        if curr_deals:
            # Ensure the data is sorted
            id_s = np.array([deal['id'] for deal in curr_deals])
            is_sorted = np.all(np.diff(id_s) >= 0)
            if not is_sorted:
                curr_deals.sort(key=lambda x: x['id'])
                id_s = np.array([deal['id'] for deal in curr_deals])
    
            
            for deal in curr_deals[0 if self.last_deal_id is None else bisect(id_s, self.last_deal_id):]: # Avoid overlaping. Start only from new deals.
                deal_sec = float(deal['time'] / 1000.0)
                deal_minute = int(deal_sec // 60)
                if deal_minute > self.last_minute:
                    if self.new_deals:
                        arr_deals = zip(*self.new_deals)
                        
                        unixtime = self.last_minute * 60 + 30
                        average = np.average(arr_deals[0], weights=arr_deals[1])
                        mean = np.mean(arr_deals[0])
                        max_ = np.max(arr_deals[0])
                        min_ = np.min(arr_deals[0])
                        first = arr_deals[0][0]
                        last = arr_deals[0][-1]
                        volume = np.sum(arr_deals[1]) / 2.0
                        
                        self.agg_utsv.append(unixtime) # Apend new time
                        self.agg_prices.append(average) # Apend new price
                        self.agg_vols.append(volume) # Append new volume
                        st_idx = np.searchsorted(self.agg_utsv, self.date_lim)
                        self.agg_utsv = self.agg_utsv[st_idx:]
                        self.agg_prices = self.agg_prices[st_idx:]
                        self.agg_vols = self.agg_vols[st_idx:]
                        
                        self.sqlqueue.put(('deals (unixtime, average, mean, max, min, first, last, volume)', (unixtime, average, mean, max_, min_, first, last, volume)))

                        self.new_deals = []
                    self.last_minute = deal_minute
                self.new_deals.append((float(deal['price']), float(deal['quoteQty'])))

            self.last_deal_id = curr_deals[-1]['id']
            self.last_deal_price = float(curr_deals[-1]['price'])
        return self.last_deal_price

    def calc_spreads(self, prices, amounts):
        return np.interp(self.sums, np.append(0, np.cumsum(amounts)), np.append(0, np.cumsum(prices * amounts)), right=np.nan) / self.sums

    def get_orders(self):
        res = req_depts(self.symbol)
        if res:
            curr_time = int(time())
    
            # Buy
            buylist = res['bids']
    
                # Ensure the data is sorted
            is_sorted = np.all(np.diff([float(item[0]) for item in buylist]) <= 0)
            if not is_sorted: buylist.sort(key=lambda x: x[0], reverse=True)
            
                # Extract arrays
            prices_b = np.array([float(item[0]) for item in buylist])
            amounts_b = np.array([float(item[1]) for item in buylist])
    
            self.max_buy = prices_b[np.nonzero(amounts_b)[0][0]] # Take the 1st price value with nonzero amount
            spreads_b = self.calc_spreads(prices_b, amounts_b)
            self.spreads_b = np.hstack((self.spreads_b, np.append(curr_time, spreads_b)[:, np.newaxis]))[:, np.searchsorted(self.spreads_b[0], self.date_lim):]
    
            self.sqlqueue.put(('spreads_buy (unixtime, maxbuy{})'.format(self.mulprstr), np.concatenate(([curr_time], [self.max_buy], spreads_b))))
    
            # Sell
            selllist = res['asks']
    
                # Ensure the data is sorted
            is_sorted = np.all(np.diff([float(item[0]) for item in selllist]) >= 0)
            if not is_sorted: selllist.sort(key=lambda x: x[0])
            
                # Extract arrays
            prices_s = np.array([float(item[0]) for item in selllist])
            amounts_s = np.array([float(item[1]) for item in selllist])
    
            self.min_sell = prices_s[np.nonzero(amounts_s)[0][0]] # Take the 1st price value with nonzero amount
            spreads_s = self.calc_spreads(prices_s, amounts_s)
            self.spreads_s = np.hstack((self.spreads_s, np.append(curr_time, spreads_s)[:, np.newaxis]))[:, np.searchsorted(self.spreads_s[0], self.date_lim):]
    
            self.sqlqueue.put(('spreads_sell (unixtime, minsell{})'.format(self.mulprstr), np.concatenate(([curr_time], [self.min_sell], spreads_s))))
    
        return self.max_buy, self.min_sell

    def get_dbs(self):
        self.date_lim = int(time() - timedelta(days=5).total_seconds())
        self.get_orders()
        self.get_deals()


class UkrMarket(DbUpdate):
    """BTC-Trade follower"""
    def __init__(self, symbol, sums, folder):
        self.symbol = symbol
        self.sums = sums
        self.folder = folder
        self.lensum = len(self.sums)
        self.agg_utimes = []
        self.agg_amnts = []
        mulprstrt = (', p{} REAL' * self.lensum).format(*range(self.lensum))
        self.mulprstr = (', p{}' * self.lensum).format(*range(self.lensum))

        # SQL
        self.sqlqueue = Queue()
        self.con = sqlite3.connect('{}BtcTradeUa_{}.db'.format(self.folder, self.symbol), detect_types=sqlite3.PARSE_DECLTYPES) # Create or connect database
        self.cur = self.con.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS deals (id INTEGER PRIMARY KEY, id_s INT, unixtime INT, type TEXT, amnt_base REAL, amnt_trade REAL, price REAL, user TEXT, order_id TEXT);')
        self.cur.execute('CREATE TABLE IF NOT EXISTS spreads_buy (id INTEGER PRIMARY KEY, unixtime INT, maxbuy REAL{});'.format(mulprstrt))
        self.cur.execute('CREATE TABLE IF NOT EXISTS spreads_sell (id INTEGER PRIMARY KEY, unixtime INT, minsell REAL{});'.format(mulprstrt))

        # Fetch data from database
        last_rec = self.cur.execute('SELECT id_s FROM deals WHERE id = (SELECT MAX(id) FROM deals);').fetchone()
        self.last_deal_id = None if last_rec is None else last_rec[0]
        self.date_lim = int(time() - timedelta(days=10).total_seconds())
        deals = self.cur.execute('SELECT unixtime, amnt_base, price FROM deals WHERE unixtime > {}'.format(self.date_lim)).fetchall()
        self.deals = np.array(zip(*deals), dtype=np.float64) if deals else np.empty((3, 0))
        self.aggregate_deals(deals)
        self.spreads_b = np.array(zip(*self.cur.execute('SELECT unixtime{} FROM spreads_buy WHERE unixtime > {} ORDER BY unixtime'.format(self.mulprstr, self.date_lim)).fetchall()), dtype=np.float64)
        self.spreads_s = np.array(zip(*self.cur.execute('SELECT unixtime{} FROM spreads_sell WHERE unixtime > {} ORDER BY unixtime'.format(self.mulprstr, self.date_lim)).fetchall()), dtype=np.float64)
        if self.spreads_b.size == 0: self.spreads_b = np.empty((self.lensum + 1, 0))
        if self.spreads_s.size == 0: self.spreads_s = np.empty((self.lensum + 1, 0))
        self.last_deal_price = self.cur.execute('SELECT price FROM deals WHERE id = (SELECT MAX(id) FROM deals);').fetchone()
        self.max_buy = self.cur.execute('SELECT maxbuy FROM spreads_buy WHERE id = (SELECT MAX(id) FROM deals);').fetchone()
        self.min_sell = self.cur.execute('SELECT minsell FROM spreads_sell WHERE id = (SELECT MAX(id) FROM deals);').fetchone()

    def get_deals(self):
        curr_deals = req_deals(self.symbol)

        if curr_deals: # If there were some deals last time
            curr_deals = curr_deals[::-1] # Reverse order
            
            # Ensure the data is sorted
            id_s = np.array([deal['id'] for deal in curr_deals])
            is_sorted = np.all(np.diff(id_s) >= 0)
            if not is_sorted:
                curr_deals.sort(key=lambda x: x['id'])
                id_s = np.array([deal['id'] for deal in curr_deals])

            new_deals = []
            for deal in curr_deals[0 if self.last_deal_id is None else bisect(id_s, self.last_deal_id):]: # Avoid overlaping. Start only from new deals.
                new_deals.append((float(deal['unixtime']), float(deal['amnt_base']), float(deal['price'])))
                param_tuple = (deal['id'], int(deal['unixtime']), deal['type'], deal['amnt_base'], deal['amnt_trade'], deal['price'], deal['user'], deal['order_id'])
                self.sqlqueue.put(('deals (id_s, unixtime, type, amnt_base, amnt_trade, price, user, order_id)', param_tuple))

            self.aggregate_deals(new_deals)
            newdeals = np.array(zip(*new_deals))
            if len(new_deals): self.deals = np.hstack((self.deals, newdeals))[:, np.searchsorted(self.deals[0], self.date_lim):]
            self.last_deal_id = curr_deals[-1]['id']
            self.last_deal_price = float(curr_deals[-1]['price'])

        return self.last_deal_price

    def aggregate_deals(self, new_deals):
        if not new_deals: return
        curr_t = self.agg_utimes[-1] if self.agg_utimes else None
        for unixtime, amnt_base, price in new_deals:
            if curr_t == unixtime: self.agg_amnts[-1] += amnt_base / 2.0
            else:
                curr_t = unixtime
                self.agg_utimes.append(unixtime)
                self.agg_amnts.append(amnt_base / 2.0)
        st_idx = np.searchsorted(self.agg_utimes, self.date_lim)
        self.agg_utimes = self.agg_utimes[st_idx:]
        self.agg_amnts = self.agg_amnts[st_idx:]

    def calc_spreads(self, prices, amounts):
        return np.interp(self.sums, np.append(0, np.cumsum(amounts)), np.append(0, np.cumsum(prices * amounts)), right=np.nan) / self.sums

    def get_orders_buy(self):
        res = req_buy_list(self.symbol)
        if res is None: return self.max_buy
        buylist = res['list']
        curr_time = int(time())

        # Ensure the data is sorted
        is_sorted = np.all(np.diff([float(item['price']) for item in buylist]) <= 0)
        if not is_sorted: buylist.sort(key=lambda x: x['price'], reverse=True)
    
        # Extract arrays
        prices_b = np.array([float(item['price']) for item in buylist])
        amounts_b = np.array([float(item['currency_trade']) for item in buylist])

        self.max_buy = prices_b[np.nonzero(amounts_b)[0][0]] # Take the 1st price value with nonzero amount
        spreads_b = self.calc_spreads(prices_b, amounts_b)
        self.spreads_b = np.hstack((self.spreads_b, np.append(curr_time, spreads_b)[:, np.newaxis]))[:, np.searchsorted(self.spreads_b[0], self.date_lim):]

        self.sqlqueue.put(('spreads_buy (unixtime, maxbuy{})'.format(self.mulprstr), np.concatenate(([curr_time], [self.max_buy], spreads_b))))

        return self.max_buy

    def get_orders_sell(self):
        res = req_sell_list(self.symbol)
        if res is None: return self.min_sell
        selllist = res['list']
        curr_time = int(time())
    
        # Ensure the data is sorted
        is_sorted = np.all(np.diff([float(item['price']) for item in selllist]) >= 0)
        if not is_sorted: selllist.sort(key=lambda x: x['price'])
        
        # Extract arrays
        prices_s = np.array([float(item['price']) for item in selllist])
        amounts_s = np.array([float(item['currency_trade']) for item in selllist])

        self.min_sell = prices_s[np.nonzero(amounts_s)[0][0]] # Take the 1st price value with nonzero amount
        spreads_s = self.calc_spreads(prices_s, amounts_s)
        self.spreads_s = np.hstack((self.spreads_s, np.append(curr_time, spreads_s)[:, np.newaxis]))[:, np.searchsorted(self.spreads_s[0], self.date_lim):]

        self.sqlqueue.put(('spreads_sell (unixtime, minsell{})'.format(self.mulprstr), np.concatenate(([curr_time], [self.min_sell], spreads_s))))

        return self.min_sell

    def get_dbs(self):
        self.date_lim = int(time() - timedelta(days=5).total_seconds())
        self.get_deals()
        self.get_orders_buy()
        self.get_orders_sell()


class GUI(Tk, Destroyer, Rsu, SockCli):
    """docstring for GUI"""
    def __init__(self, mp, folder):
        super(GUI, self).__init__()
        self.view_idx = 0

        # Absorb papameters
        self.mp = mp
        self.folder = folder

        # Window setup
        self.protocol("WM_DELETE_WINDOW", self.destroyer)
        self.title('Market Monitor')
        self.geometry('1000x600')
        self.resizable(True, True)
        
        # Create widgets
            # Menu bar
        self.var_w = IntVar(self, 0)
        menubar = Menu(self, relief=FLAT, font=('Helvetica', 10), bg='gray88')
        viewmenu = Menu(menubar, tearoff=0)
        for idx, (cu0, cu1, market, _) in enumerate(mp): viewmenu.add_radiobutton(label='{} {}-{}'.format(market, cu0, cu1), variable=self.var_w, value=idx, command=self.switch_view)
        viewmenu.add_separator()
        self.var_h = BooleanVar(self, False) # Menu checkbutton variable
        viewmenu.add_checkbutton(label='Hide plot', onvalue=True, offvalue=False, variable=self.var_h, command=self.plot_hide_show)
        menubar.add_cascade(label="View", menu=viewmenu)
        self.config(menu=menubar)

            # Frames
        main_frame = Frame(self)
        main_frame.pack(padx=4, side=LEFT, fill=BOTH, expand=True)

        tabs_frame = Frame(main_frame)
        tabs_frame.pack(side=LEFT, fill=Y)

            # Market frame
        self.ticklw = Label(tabs_frame, text='{} {}-{}'.format(mp[self.view_idx][2], mp[self.view_idx][0], mp[self.view_idx][1]), font=('Times', 10, 'italic'))
        btctradeua_frame = ttk.LabelFrame(tabs_frame, labelwidget=self.ticklw, labelanchor=N)
        btctradeua_frame.pack(padx=2, pady=2, fill=X)
        Label(btctradeua_frame, text='UAH').grid(row=0, column=1, padx=2, pady=2)
        Label(btctradeua_frame, text='USD').grid(row=0, column=2, padx=2, pady=2)
        
                # Last deal
        Label(btctradeua_frame, text='Last deal').grid(row=1, column=0, padx=2, pady=2, sticky=W)
        self.lbl_ldh = Label(btctradeua_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_ldh.grid(row=1, column=1, padx=2, pady=2)
        self.lbl_ldd = Label(btctradeua_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_ldd.grid(row=1, column=2, padx=2, pady=2)
        
                # Max buy
        Label(btctradeua_frame, text='Max buy').grid(row=2, column=0, padx=2, pady=2, sticky=W)
        self.lbl_mbh = Label(btctradeua_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_mbh.grid(row=2, column=1, padx=2, pady=2)
        self.lbl_mbd = Label(btctradeua_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_mbd.grid(row=2, column=2, padx=2, pady=2)
        
                # Min sell
        Label(btctradeua_frame, text='Min sell').grid(row=3, column=0, padx=2, pady=2, sticky=W)
        self.lbl_msh = Label(btctradeua_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_msh.grid(row=3, column=1, padx=2, pady=2)
        self.lbl_msd = Label(btctradeua_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_msd.grid(row=3, column=2, padx=2, pady=2)

                # Spread
        Label(btctradeua_frame, text='Spread, %').grid(row=4, column=0, padx=2, pady=2)
        self.lbl_spr = Label(btctradeua_frame, text='', background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_spr.grid(row=4, column=1, padx=2, pady=2, columnspan=2)

            # Plots frame
        self.plots_frame = Frame(main_frame)
        self.plots_frame.pack(side=LEFT, fill=BOTH, expand=True)
        globplot_frame = ttk.LabelFrame(self.plots_frame, labelwidget=Label(self.plots_frame, text='Historical data', font=('Times', 10, 'italic')), labelanchor=N)
        globplot_frame.pack(padx=2, pady=2, fill=BOTH, expand=True)

                # Matplotlib
        self.lensum = len(mp[self.view_idx][3])
        color = plt.cm.viridis(np.linspace(0, 1, self.lensum, endpoint=True))
        self.fig, (self.ax0, self.ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})

        self.ax0.plot([], [], color='r', linewidth=1, zorder=100, label='deals')
        for i in range(self.lensum): self.ax0.plot([], [], color=color[i], linewidth=1.0, label=mp[self.view_idx][3][i])
        for i in range(self.lensum): self.ax0.plot([], [], color=color[i], linewidth=1.0)
        self.ax1.plot([], [], linestyle='', markersize=1, marker='.', mec='dimgrey')[0]

        self.ax0.set_ylabel('Price, USD', fontsize='small')
        self.ax1.set_ylabel('Amounts, USD', fontsize='small')
        self.ax0.label_outer()
        self.ax1.label_outer()
        self.ax0.legend(loc='best', fancybox=False, fontsize='xx-small')
        self.canvas = FigureCanvasTkAgg(self.fig, master=globplot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, padx=3, fill=BOTH, expand=1)
        self.toolbar = CustomizedNavigationToolbar2Tk(self.canvas, globplot_frame)
        self.toolbar.zoom_flag = False
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.canvas.mpl_connect("resize_event", lambda event:adjust_margins(self.fig,event,l=0.9))

        # Prepare for monitor_loop
        self.pending_w = None
        self.market_objs = [BinanceMarket(item[0] + item[1], item[3], self.folder) if item[2] == 'Binance' else UkrMarket((item[0] + '_' + item[1]).lower(), item[3], self.folder) for item in mp] # Initialize market and pair objects
        self.sockinit() # Launch the client
        self.curr_obj = self.market_objs[self.view_idx] # Set the current object (to be displayed)
        self.commit_epoch = Epoch(reftime='2020-7-22 0:7:0', tinth=0.25)
        self.last_epoch = self.commit_epoch()
        self.monitor_loop_part_a() # Enter the monitoring loop

    def monitor_loop_part_a(self):
        self.threads = [Thread(target=market_obj.get_dbs) for market_obj in self.market_objs] # Do http requests in parallel threads
        for thread in self.threads: thread.start() # Start each thread
        self.after(1000, self.monitor_loop_part_b)

    def monitor_loop_part_b(self):
        if any([thread.is_alive() for thread in self.threads]): self.after(1000, self.monitor_loop_part_b)
        else:
            print('Time {}'.format(datetime.fromtimestamp(int(time()))))
            self.rsu() # Refresh numerical data, save SQL and update plot
    
            # Save SQL every 15 min
            curr_epoch = self.commit_epoch()
            if curr_epoch != self.last_epoch:
                self.last_epoch = curr_epoch
                self.save_sql(commit=True)
    
            sleeptime = 60 - (time() % 60)
            self.after(int(sleeptime * 1000), self.monitor_loop_part_a)

    # Matplotlib functions
    def on_key_press(self, event): key_press_handler(event, self.canvas, self.toolbar)

    def update_plot(self):
        """Set fresh data, update view"""
        if self.var_h.get(): return # If 'Hide plot' is checked, return

        # Set data
        if self.mp[self.view_idx][2] == 'Binance':
            time_deals = [datetime.fromtimestamp(t) for t in self.curr_obj.agg_utsv]
            deals = self.curr_obj.agg_prices
            time_amounts = time_deals
            amounts = self.curr_obj.agg_vols
        elif self.mp[self.view_idx][2] == 'BTCTradeUA':
            time_deals = [datetime.fromtimestamp(t) for t in self.curr_obj.deals[0]]
            deals = self.curr_obj.deals[2]
            time_amounts = [datetime.fromtimestamp(t) for t in self.curr_obj.agg_utimes]
            amounts = self.curr_obj.agg_amnts

            # Deals
        self.ax0.lines[0].set_xdata(time_deals)
        self.ax0.lines[0].set_ydata(deals)

            # Amounts
        self.ax1.lines[0].set_xdata(time_amounts)
        self.ax1.lines[0].set_ydata(amounts)

            # Buy orders
        time_sprears_s = [datetime.fromtimestamp(t) for t in self.curr_obj.spreads_b[0]]
        for i in range(1, self.lensum + 1):
            self.ax0.lines[i].set_xdata(time_sprears_s)
            self.ax0.lines[i].set_ydata(self.curr_obj.spreads_b[i])

            # Sell orders
        time_spreads_b = [datetime.fromtimestamp(t) for t in self.curr_obj.spreads_s[0]]
        for i in range(1, self.lensum + 1):
            self.ax0.lines[i+self.lensum].set_xdata(time_spreads_b)
            self.ax0.lines[i+self.lensum].set_ydata(self.curr_obj.spreads_s[i])

        # Calc limits
        y_min, y_max = np.nanmin(self.curr_obj.spreads_b[-1]), np.nanmax(self.curr_obj.spreads_s[-1])
        try: y_min, y_max = min(y_min, np.nanmin(deals)), max(y_max, np.nanmax(deals))
        except ValueError: pass
        y_margin = (y_max - y_min) * 0.05
        self.ax0.mem_xlim = (time_spreads_b[0], time_spreads_b[-1] + timedelta(hours=4))
        self.ax0.mem_ylim = (y_min - y_margin, y_max + y_margin)
        try: self.ax1.mem_ylim = (0, np.max(self.ax1.lines[0].get_ydata()) * 1.05)
        except ValueError: pass

        # Set view
        if self.toolbar.zoom_flag is False: # If not zooming,
            self.ax0.set_xlim(self.ax0.mem_xlim)
            self.ax0.set_ylim(self.ax0.mem_ylim)
            self.ax1.set_ylim(self.ax1.mem_ylim)

        self.canvas.draw()

    # Appearance funcs
    def plot_hide_show(self):
        if self.var_h.get(): self.plots_frame.forget() # Hide plot if the box is checked
        else:
            self.plots_frame.pack(side=LEFT, fill=BOTH, expand=True) # Show the plot
            if not self.threads[self.var_w.get()].is_alive(): self.update_plot()

    def refresh(self):
        last_deal = self.curr_obj.last_deal_price
        max_buy = self.curr_obj.max_buy
        min_sell = self.curr_obj.min_sell
        spread = (min_sell - max_buy) * 200 / (min_sell + max_buy)

        if self.mp[self.view_idx][2] == 'Binance':
            # UAH
            try:
                usd_uah_rate = self.get_shared_data('get_usdrate')
                self.lbl_ldh['text'] = sci_round(last_deal * usd_uah_rate, 5)
                self.lbl_mbh['text'] = sci_round(max_buy * usd_uah_rate, 5)
                self.lbl_msh['text'] = sci_round(min_sell * usd_uah_rate, 5)
            except:
                self.lbl_ldh['text'] = 'Err'
                self.lbl_mbh['text'] = 'Err'
                self.lbl_msh['text'] = 'Err'
            
            # USD
            self.lbl_ldd['text'] = sci_round(last_deal, 5)
            self.lbl_mbd['text'] = sci_round(max_buy, 5)
            self.lbl_msd['text'] = sci_round(min_sell, 5)

        elif self.mp[self.view_idx][2] == 'BTCTradeUA':
            # UAH
            self.lbl_ldh['text'] = sci_round(last_deal, 5)
            self.lbl_mbh['text'] = sci_round(max_buy, 5)
            self.lbl_msh['text'] = sci_round(min_sell, 5)

            # USD
            try:
                usd_uah_rate = self.get_shared_data('get_usdrate')
                self.lbl_ldd['text'] = sci_round(last_deal / usd_uah_rate, 5)
                self.lbl_mbd['text'] = sci_round(max_buy / usd_uah_rate, 5)
                self.lbl_msd['text'] = sci_round(min_sell / usd_uah_rate, 5)
            except:
                self.lbl_ldd['text'] = 'Err'
                self.lbl_mbd['text'] = 'Err'
                self.lbl_msd['text'] = 'Err'
            
        self.lbl_spr['text'] = sci_round(spread, 2)

    # Button calls
    def switch_view(self):
        if self.pending_w is not None: self.after_cancel(self.pending_w)
        view_idx = self.var_w.get()

        if self.threads[view_idx].is_alive(): self.pending_w = self.after(500, self.switch_view) # If the data is under processing, wait and call later
        else:
            # Change flags and current object
            self.view_idx = view_idx
            self.toolbar.zoom_flag = False
            self.curr_obj = self.market_objs[self.view_idx]
    
            # Edit labels
            for i in range(self.lensum): self.ax0.lines[i+1].set_label(self.curr_obj.sums[i]) # Change legend labels
            self.ax0.legend(loc='best', fancybox=False, fontsize='xx-small')
            market = self.mp[self.view_idx][2]
            if market == 'Binance': laber_currency = 'USD'
            elif market == 'BTCTradeUA': laber_currency = 'UAH'
            self.ax0.set_ylabel('Price, {}'.format(laber_currency))
            self.ax1.set_ylabel('Amounts, {}'.format(laber_currency))
            self.ticklw['text'] = '{} {}-{}'.format(market, mp[self.view_idx][0], mp[self.view_idx][1])
    
            # Ubdate displaying of graphical and numerical data
            self.update_plot()
            self.refresh()

    # Other functions
    def save_sql(self, commit=False):
        for market_obj in self.market_objs: market_obj.db_update(commit=commit)


if __name__ == '__main__':
    # Settings
    mp = (('ETH', 'UAH', 'BTCTradeUA', (0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16)),
          ('USDT', 'UAH', 'BTCTradeUA', (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)),
          ('ETH', 'USDT', 'Binance', (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)),
          ('XMR', 'USDT', 'Binance', (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))
    )
    folder = 'DataBases/'

    gui = GUI(mp, folder)
    gui.mainloop()

















            # # a = np.nanmin(self.curr_obj.spreads_b[-1])
            # # b = np.nanmin(self.curr_obj.agg_prices)
            # # y_min = min(a, b)
            # print('spreads_b {}, agg_prices {}'.format(self.curr_obj.spreads_b[-1].size, len(self.curr_obj.agg_prices)))
            # try:
            #     y_min = min(np.nanmin(self.curr_obj.spreads_b[-1]), np.nanmin(self.curr_obj.agg_prices))
            # except:
            #     a = np.nanmin(self.curr_obj.spreads_b[-1])
            #     b = np.nanmin(self.curr_obj.agg_prices)
            #     y_min = min(a, b)
            # y_max = max(np.nanmax(self.curr_obj.spreads_s[-1]), np.nanmax(self.curr_obj.agg_prices))


        # self.pending_h = None

        # if self.pending_h is not None: self.after_cancel(self.pending_h) # Cancel the pending function call

            # if self.threads[self.var_w.get()].is_alive(): self.pending_h = self.after(500, self.plot_hide_show) # If the data is under processing, wait and call later,
            # else: self.update_plot() # else update the plot immediately.
