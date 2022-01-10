#!/usr/bin/env python
from time import sleep, time
from Tkinter import *
import ttk
import tkMessageBox
from PIL import ImageTk, Image
from Queue import Queue
import sqlite3
from datetime import datetime
import functools
from binance.client import Client
from binance.exceptions import BinanceAPIException
import sys, traceback
from Common import DbUpdate, Epoch, Destroyer, Rsu, SockCli, sci_round, floor_str, OP_SYS


def clientdecor(func, attempts=10):
    """Decorator of binance.client.Client methods to handle connection issues and exceptions"""
    name = func.__name__
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(attempts):
            if i != 0: sleep(5)
            try: return func(*args, **kwargs)
            except Exception as e:
                # print('{} failed #{}. {}'.format(name, i, repr(e)))
                # print('An exception of type {0} occurred. Arguments:\n{1!r}'.format(type(e).__name__, e.args))
                # traceback.print_exc()
                print('\n{} failed #{}'.format(name, i))
                if args: print(args)
                if kwargs: print(kwargs)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(traceback.format_exception_only(exc_type, exc_value))
    return wrapper


class Login(Tk, object):
    """Login window. Used to input credentials"""
    def __init__(self):
        super(Login, self).__init__()
        self.client = None

        # Window setup
        if OP_SYS == 'Linux': self.tk.call('wm', 'iconphoto', self._w, PhotoImage(file='Images/Markets/binance.png'))
        elif OP_SYS == 'Windows': self.iconbitmap('Images/Markets/binance.ico')
        self.resizable(False, False)
        self.title('Enter your credentials')

        Label(self, text='Api key:').grid(row=0, column=0, padx=2, pady=2, sticky=W)
        self.et_api_key = Entry(self, width=64)
        self.et_api_key.grid(row=0, column=1, padx=2, pady=2, sticky=E)
        self.et_api_key.focus_set()

        Label(self, text='Api secret:').grid(row=1, column=0, padx=2, pady=2, sticky=W)
        self.et_api_secret = Entry(self, width=64)
        self.et_api_secret.grid(row=1, column=1, padx=2, pady=2, sticky=E)

        Button(self, text='Ok', width=6, command=self.ok).grid(row=2, column=0, padx=2, pady=2, columnspan=2, sticky=E)

    def ok(self):
        """Ok button command"""
        try:
            self.client = Client(self.et_api_key.get(), self.et_api_secret.get())
            self.client.get_account()
            self.destroy()
        except BinanceAPIException:
            self.client = None
            tkMessageBox.showwarning('ERROR', 'Invalid api key and/or api sicret. Try again.')

    def mainloop(self, n=0):
        """Mainloop must return the client object"""
        super(Login, self).mainloop(n)
        return self.client


class BinanceDeals(DbUpdate):
    """docstring for BinanceDeals"""
    def __init__(self, folder_db, market_pairs, coins):
        self.client = Login().mainloop()
        if self.client is None: exit()

        self.msg_template = '\n\n{}\n{}\n{} {} {}\nPrice: {} {}\nOrderID: {}\nStatus: {}'
        self.blnce_dict = dict.fromkeys(coins) # Container for balance
        self.calcd_blnce = self.blnce_dict.copy() # Balance, calculated from BUY or SELL responce

        self.get_asset_balance = clientdecor(self.client.get_asset_balance)
        self.order_market_buy = clientdecor(self.client.order_market_buy, attempts=3)
        self.order_market_sell = clientdecor(self.client.order_market_sell, attempts=3)

        # SQL
        self.sqlqueue = Queue()
        self.con = sqlite3.connect('{}BinanceDeals.db'.format(folder_db), detect_types=sqlite3.PARSE_DECLTYPES) # Create or connect database
        self.cur = self.con.cursor()
        for coin in coins: self.cur.execute('CREATE TABLE IF NOT EXISTS balance_{} (id INTEGER PRIMARY KEY, unixtime INT, amount REAL);'.format(coin))
        for cu0, cu1 in market_pairs: self.cur.execute('CREATE TABLE IF NOT EXISTS orders_{} (id INTEGER PRIMARY KEY, unixtime INT, amount REAL, price REAL, side TEXT, status TEXT, order_id INT);'.format(cu0 + cu1))

    def balance(self, *coins):
        amnts = []
        for coin in coins:
            res = self.get_asset_balance(asset=coin)
            co_lo = coin.lower()
            if res is None: amnts.append(self.blnce_dict[coin])
            else:
                amnt = float(res['free'])
                amnts.append(amnt)
                self.blnce_dict[coin] = amnt
                curr_ts = int(time())
                self.sqlqueue.put(('balance_{} (unixtime, amount)'.format(co_lo), (curr_ts, amnt)))
        return amnts

    def buy(self, pair):
        res = self.order_market_buy(symbol=pair.us, quoteOrderQty=floor_str(pair.ams[1],8))
        # res = {u'orderId': 2015555501, 
        #        u'clientOrderId': u'h07IKNSiiNDf6rAIf9BSXf', 
        #        u'origQty': u'0.02496000', 
        #        u'fills': [{u'commission': u'0.00002496', 
        #                    u'price': u'458.85000000', 
        #                    u'commissionAsset': u'ETH', 
        #                    u'tradeId': 205638126, 
        #                    u'qty': u'0.02496000'}], 
        #        u'symbol': u'ETHUSDT', 
        #        u'side': u'BUY', 
        #        u'timeInForce': u'GTC', 
        #        u'status': u'FILLED', 
        #        u'orderListId': -1, 
        #        u'transactTime': 1605181740238L, 
        #        u'type': u'MARKET', 
        #        u'price': u'0.00000000', 
        #        u'executedQty': u'0.02496000', 
        #        u'cummulativeQuoteQty': u'11.45289600'}
        return self.trade_rec(res, pair)

    def sell(self, pair):
        res = self.order_market_sell(symbol=pair.us, quantity=floor_str(pair.ams[0],5))
        # res = {u'orderId': 2015542017, 
        #        u'clientOrderId': u'3Rr6MBijDeZLp4yC98vUGv', 
        #        u'origQty': u'0.02500000', 
        #        u'fills': [{u'commission': u'0.01146450', 
        #                    u'price': u'458.58000000', 
        #                    u'commissionAsset': u'USDT', 
        #                    u'tradeId': 205637461, 
        #                    u'qty': u'0.02500000'}], 
        #        u'symbol': u'ETHUSDT', 
        #        u'side': u'SELL', 
        #        u'timeInForce': u'GTC', 
        #        u'status': u'FILLED', 
        #        u'orderListId': -1, 
        #        u'transactTime': 1605181619482L, 
        #        u'type': u'MARKET', 
        #        u'price': u'0.00000000', 
        #        u'executedQty': u'0.02500000', 
        #        u'cummulativeQuoteQty': u'11.46450000'}
        return self.trade_rec(res, pair)

    def trade_rec(self, res, pair):
        if res is not None:
            try:
                unixtime = res['transactTime'] // 1000
                amount = float(res['origQty'])
                
                orders = res['fills']
                prices = [float(order['price']) for order in orders]
                amounts = [float(order['qty']) for order in orders]
                price = sum(map(lambda x, y: x * y, prices, amounts)) / sum(amounts)
                
                side = res['side']
                executedQty = float(res['executedQty'])
                cummulativeQuoteQty = float(res['cummulativeQuoteQty'])

                status = res['status']
                order_id = res['orderId']
            except:
                print(res)
                return
            
            delta_am0 = executedQty * 0.999 if side == 'BUY' else -executedQty
            delta_am1 = -cummulativeQuoteQty if side == 'BUY' else cummulativeQuoteQty * 0.999
            pair.set_ams(pair.ams[0] + delta_am0, pair.ams[1] + delta_am1)
            pair.autocnt += bool(pair.trading_mode)

            param_tuple = (unixtime, amount, price, side, status, order_id)
            self.sqlqueue.put(('orders_{} (unixtime, amount, price, side, status, order_id)'.format(pair.ls), param_tuple))
            return self.msg_template.format(datetime.fromtimestamp(unixtime), 'AUTO' if pair.trading_mode else 'MANUAL', side.capitalize(), amount, pair.u0, price, pair.u1, order_id, status) # Return message


class CurrencyPair(DbUpdate):
    """Container for lower and uppercase symbols of currency pair"""
    def __init__(self, idx, coins, folder_db, coin0, coin1):
        self.u0 = coin0.upper()
        self.u1 = coin1.upper()
        self.l0 = coin0.lower()
        self.l1 = coin1.lower()
        self.us = self.u0 + self.u1
        self.ls = self.l0 + self.l1
        
        self.idx = idx
        self.trading_mode = 0
        self.autocnt = 0
        self.curr_decision = None
        self.prev_decision = None
        self.last_dec_time = None
        self.cos = (coins.index(coin0), coins.index(coin1))

        # SQL
        self.sqlqueue = Queue()
        self.con = sqlite3.connect('{}Subaccounts.db'.format(folder_db), detect_types=sqlite3.PARSE_DECLTYPES) # Create or connect database
        self.cur = self.con.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, unixtime INT, am0 real, am1 REAL);'.format(self.ls))

        # Fetch data from database
        last_rec = self.cur.execute('SELECT am0, am1 FROM {0} WHERE id = (SELECT MAX(id) FROM {0});'.format(self.ls)).fetchone()
        self.ams = (0, 0) if last_rec is None else last_rec

    def set_ams(self, am0, am1):
        self.ams = (round(am0, 8), round(am1, 8))
        self.sqlqueue.put(('{} (unixtime, am0, am1)'.format(self.ls), (int(time()), am0, am1)))


class SubaccEntry(Entry, object):
    """docstring for SubaccEntry"""
    def __init__(self, parent, coidx, amount, update_asset_func):
        self.sv = StringVar(parent, amount)
        super(SubaccEntry, self).__init__(parent, bg='lemon chiffon', width=8, textvariable=self.sv)
        self.sv.trace("w", self.entry_callback)
        self.update_asset_func = update_asset_func
        self.coidx = coidx
        self.amount = amount
        self.error = False

    def entry_callback(self, *args):
        try: self.amount, self.error = float(self.sv.get()), False
        except: self.amount, self.error = 0.0, True
        self['bg'] = 'red' if self.error else 'lemon chiffon'
        self.update_asset_func(self.coidx)


class FreeAssetLabel(Label, object):
    """docstring for FreeAssetLabel"""
    def __init__(self, parent, curramnt, coslot, entries):
        super(FreeAssetLabel, self).__init__(parent, width=8, borderwidth=1, relief=SUNKEN)
        self.curramnt = curramnt
        self.coslot = coslot
        self.entries = entries
        self.calc_amount()

    def calc_amount(self):
        freeamnt = self.curramnt - sum([self.entries[slot].amount for slot in self.coslot])
        self.negamnt = freeamnt < 0
        self['text'] = sci_round(freeamnt, 5, 5)
        self['bg'] = 'red' if freeamnt < 0 else 'white'


class Dialog(Toplevel, object):
    """Dialog box"""
    def __init__(self, parent):
        super(Dialog, self).__init__()
        self.transient(parent)
        self.title('Subaccounts')
        self.parent = parent
        self.initial_focus = self

        self.grab_set()
        if not self.initial_focus: self.initial_focus = self
        self.geometry('+{:d}+{:d}'.format(parent.winfo_rootx() + 190, parent.winfo_rooty() - 56))
        self.initial_focus.focus_set()

        # Create widgets
            # Subaccounts frame
        subaccs_frame = ttk.LabelFrame(self, labelwidget=Label(self, text='Subaccounts', font=('Times', 10, 'italic')), labelanchor=N)
        self.entries = []
        self.coslots = [[] for _ in range(self.parent.cocnt)] # Indices of the self.entries corresponding to a coin
        for pair in self.parent.pairs:
            if pair.idx: ttk.Separator(subaccs_frame, orient=HORIZONTAL).pack(fill=X)
            subacfr = Frame(subaccs_frame)
            subacfr.pack(pady=2, fill=X)
            Label(subacfr, text='#{}'.format(pair.idx)).grid(row=0, column=1, padx=2, sticky=E)
            for j in range(2):
                coidx = pair.cos[j]
                self.coslots[coidx].append(pair.idx * 2 + j)
                Label(subacfr, text=self.parent.coins[coidx]).grid(row=j+1, column=0, padx=2, sticky=E)
                entry = SubaccEntry(subacfr, coidx, pair.ams[j], self.update_asset)
                entry.grid(row=j+1, column=1, padx=2, sticky=W)
                self.entries.append(entry)

            # Assets frame
        assets_frame = ttk.LabelFrame(self, labelwidget=Label(self, text='Free assets', font=('Times', 10, 'italic')), labelanchor=N)
        self.free_assets = []
        for coidx, (coin, amount) in enumerate(zip(self.parent.coins, self.parent.amnts)):
            Label(assets_frame, text=coin).grid(row=coidx, column=0, padx=2, pady=2, sticky=E)
            free_asset = FreeAssetLabel(assets_frame, amount, self.coslots[coidx], self.entries)
            free_asset.grid(row=coidx, column=1, padx=2, pady=2, sticky=W)
            self.free_assets.append(free_asset)

            # Ok button
        btnbox = Frame(self)
        self.btn_ok = ttk.Button(btnbox, text="OK", width=8, command=self.ok, state=DISABLED if self.check_error() else NORMAL)
        self.btn_ok.pack(side=RIGHT, padx=2, pady=2)
        self.bind("<Return>", self.ok)

        # Pack all widgets
        assets_frame.pack(padx=2, pady=2, fill=X)
        subaccs_frame.pack(padx=2, pady=2, fill=X)
        btnbox.pack(fill=X)

        self.wait_window(self)

    def check_error(self):
        """Returns True if free assets are overdravn"""
        return any([item.negamnt for item in self.free_assets]) or any([item.error for item in self.entries])

    def update_asset(self, coidx):
        self.free_assets[coidx].calc_amount()
        self.btn_ok['state'] = DISABLED if self.check_error() else NORMAL

    def ok(self, event=None):
        """Ok button command"""
        for pair in self.parent.pairs: pair.set_ams(self.entries[pair.idx * 2].amount, self.entries[pair.idx * 2 + 1].amount)
        self.parent.update_subacc()
        self.parent.focus_set() # Put focus back to the parent window
        self.destroy()


class GUI(Tk, Destroyer, SockCli):
    """Main GUI window"""
    def __init__(self, timeshift, market_pairs, tinth, reftime, folder_db):
        self.coins = tuple(set([coin for pair in market_pairs for coin in pair]))
        self.binance_obj = BinanceDeals(folder_db, market_pairs, self.coins) # Instantiate BINANCE API

        super(GUI, self).__init__()

        self.timeshift = timeshift
        self.cocnt = len(self.coins)
        self.pairs = tuple([CurrencyPair(idx, self.coins, folder_db, *pair) for idx, pair in enumerate(market_pairs)])
        self.algotypes = (None, 'AP', 'PT', 'both')

        # Images
        self.imgs = [ImageTk.PhotoImage(Image.open('Images/Coins/{}_s.png'.format(coin))) for coin in self.coins]

        # Window setup
        self.protocol("WM_DELETE_WINDOW", self.destroyer)
        self.title('Binance CONTROL')
        if OP_SYS == 'Linux': self.tk.call('wm', 'iconphoto', self._w, PhotoImage(file='Images/Markets/binance.png'))
        elif OP_SYS == 'Windows': self.iconbitmap('Images/Markets/binance.ico')
        self.resizable(True, True)

        # Create widgets
            # Menu bar
        self.var_w = IntVar(self, 0)
        menubar = Menu(self, relief=FLAT, font=('Helvetica', 10), bg='gray88')
        viewmenu = Menu(menubar, tearoff=0)
        for pair in self.pairs: viewmenu.add_radiobutton(label=pair.us, variable=self.var_w, value=pair.idx, command=self.switch_view)
        menubar.add_cascade(label="View", menu=viewmenu)

        settingsmenu = Menu(menubar, tearoff=0)
        settingsmenu.add_command(label="Subaccounts", command=lambda:Dialog(self))
        menubar.add_cascade(label="Settings", menu=settingsmenu)

        self.config(menu=menubar)

        main_frame = Frame(self)
        main_frame.pack(side=LEFT, fill=BOTH, expand=True)

        tabs_frame = Frame(main_frame)
        tabs_frame.pack(side=LEFT, fill=BOTH, expand=True)
        
            # Subaccount frame
        self.sublw = Label(tabs_frame, font=('Times', 10, 'italic'))
        subacc_frame = ttk.LabelFrame(tabs_frame, labelwidget=self.sublw, labelanchor=N)
        subacc_frame.pack(padx=2, pady=2, fill=X)
        
        self.lbl_cu1 = Label(subacc_frame)
        self.lbl_cu1.grid(row=0, column=0, padx=2, pady=2, sticky=E)
        self.lbl_am1 = Label(subacc_frame, background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_am1.grid(row=0, column=1, padx=2, pady=2, sticky=W)
        
        self.lbl_cu0 = Label(subacc_frame)
        self.lbl_cu0.grid(row=1, column=0, padx=2, pady=2, sticky=E)
        self.lbl_am0 = Label(subacc_frame, background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_am0.grid(row=1, column=1, padx=2, pady=2, sticky=W)
        
        Label(subacc_frame, text='Total, in USD').grid(row=2, column=0, padx=2, pady=2, sticky=E)
        self.lbl_tusd = Label(subacc_frame, background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_tusd.grid(row=2, column=1, padx=2, pady=2, sticky=W)
    
        self.btn_b = ttk.Button(subacc_frame, text='Buy', width=8, command=self.manual_buy)
        self.btn_b.grid(row=3, column=0, padx=2, pady=2, sticky=E)
        self.btn_s = ttk.Button(subacc_frame, text='Sell', width=8, command=self.manual_sell)
        self.btn_s.grid(row=3, column=1, padx=2, pady=2, sticky=W)

        rb_frame = Frame(subacc_frame)
        rb_frame.grid(row=4, column=0, columnspan=2)
        self.var_tm = IntVar(self, 0)
        for idx, mode in enumerate(['Manual mode', 'Algorithm AP', 'Algorithm PT', 'Both algorithms']):
            ttk.Radiobutton(rb_frame, text=mode, variable=self.var_tm, value=idx, command=self.switch_trading_mode).grid(row=idx, column=0, padx=20, pady=2, sticky=W)

        Label(subacc_frame, text='Auto counter').grid(row=8, column=0, padx=2, pady=2, sticky=E)
        self.lbl_cnt = Label(subacc_frame, text=0, background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_cnt.grid(row=8, column=1, padx=2, pady=2, sticky=W)

        subacc_frame.grid_columnconfigure(0, weight=1)
        subacc_frame.grid_columnconfigure(1, weight=1, pad=0)

            # Assets frame
        assets_frame = ttk.LabelFrame(tabs_frame, labelwidget=Label(tabs_frame, text='Assets', font=('Times', 10, 'italic')), labelanchor=N)
        assets_frame.pack(padx=2, pady=2, fill=X)

        self.lbl_amnts = []
        for idx, (coin, img) in enumerate(zip(self.coins, self.imgs)):
            Label(assets_frame, text=coin).grid(row=idx, column=0, padx=2, pady=2, sticky=E)
            lbl_img = Label(assets_frame, image=img)
            lbl_img.grid(row=idx, column=1, padx=2, pady=2, sticky=W)
            lbl_amount = Label(assets_frame, background='white', width=8, borderwidth=1, relief=SUNKEN)
            lbl_amount.grid(row=idx, column=2, padx=2, pady=2, sticky=W)
            self.lbl_amnts.append(lbl_amount)

        Label(assets_frame, text='Total, in USD').grid(row=self.cocnt, column=0, columnspan=2, padx=2, pady=2, sticky=E)
        self.lbl_astot = Label(assets_frame, background='white', width=8, borderwidth=1, relief=SUNKEN)
        self.lbl_astot.grid(row=self.cocnt, column=2, padx=2, pady=2, sticky=W)       

        assets_frame.grid_columnconfigure(0, weight=1)
        assets_frame.grid_columnconfigure(1, weight=0)
        assets_frame.grid_columnconfigure(2, weight=1, pad=10)

            # Message frame
        msg_frame = ttk.LabelFrame(tabs_frame, labelwidget=Label(tabs_frame, text='Messages', font=('Times', 10, 'italic')), labelanchor=N)
        msg_frame.pack(padx=2, pady=2, fill=BOTH, expand=True)

        msg_subframe = Frame(msg_frame)
        msg_subframe.pack(padx=2, pady=2, fill=BOTH, expand=True)

        self.txt = Text(msg_subframe, height=10, width=20, state='disabled')
        yscrollbar = ttk.Scrollbar(msg_subframe, command=self.txt.yview)
        yscrollbar.pack(side=RIGHT, pady=1, fill=Y)
        self.txt.pack(side=RIGHT, fill=BOTH, expand=True)
        self.txt['yscrollcommand'] = yscrollbar.set
        self.text_msg('{}\nApp started'.format(datetime.fromtimestamp(int(time()))))

        # Initialize trading
        self.sockinit()
        self.amnts = self.binance_obj.balance(*self.coins)
        print(self.coins)
        print(self.amnts)
        self.update_balance()
        self.switch_view()
        self.epoch = Epoch(reftime, tinth)
        self.trading_loop()

    # Loop funcs
    def trading_loop(self):
        """Main loop of trading and data updating"""
        print('Epoch {}, {}'.format(self.epoch(), datetime.fromtimestamp(time())))
        self.amnts = self.binance_obj.balance(*self.coins)
        self.update_balance()
        self.update_subacc()
        for pair in self.pairs:
            if pair.trading_mode:
                self.update_decision(pair)
                self.algo_trading(pair) # Execute algo trading
        self.save_sql()
        sleeptime = self.epoch.avedur - (time() - self.epoch.refts - self.timeshift) % self.epoch.avedur
        self.after(int(sleeptime * 1000), self.trading_loop)

    def update_decision(self, pair):
        """Ask the supervisor for advise (buy or sell)"""
        for _ in range(5):
            try: decision, dec_time = self.get_shared_data('get_decision', pair.u0, self.algotypes[pair.trading_mode])
            except: return
            if dec_time != pair.last_dec_time:
                pair.curr_decision, pair.last_dec_time = decision, dec_time
                return
            sleep(2)

    def algo_trading(self, pair):
        if pair.curr_decision != pair.prev_decision and pair.curr_decision is not None: # If the decision has been changed
            pair.prev_decision = pair.curr_decision
            self.trade(self.binance_obj.buy if pair.curr_decision is True else self.binance_obj.sell, pair)

    # Appearance funcs
    def update_subacc(self):
        self.lbl_cnt['text'] = self.pair.autocnt
        amnt0 = self.pair.ams[0]
        amnt1 = self.pair.ams[1]
        self.lbl_am0['text'] = sci_round(amnt0, 5, 5)
        self.lbl_am1['text'] = sci_round(amnt1, 5, 5)
        try: price0, price1 = self.get_shared_data('get_price', self.pair.u0)[0], self.get_shared_data('get_price', self.pair.u1)[0]
        except: return
        self.lbl_tusd['text'] = sci_round(amnt0 * price0 + amnt1 * price1, 5, 5)

    def update_balance(self, changes=()):
        for coidx, amnt in changes: self.amnts[coidx] = amnt # Update amounts
        total_usd = 0
        for lbl, amnt, coin in zip(self.lbl_amnts, self.amnts, self.coins):
            lbl['text'] = sci_round(amnt, 5, 5)
            try: total_usd += amnt * self.get_shared_data('get_price', coin)[0]
            except: total_usd = 'Error'
        self.lbl_astot['text'] = total_usd if type(total_usd) == str else sci_round(total_usd, 5, 5)

    def text_msg(self, msg):
        self.txt.configure(state='normal')
        self.txt.insert(END, msg)
        self.txt.configure(state='disabled')
        self.txt.yview_moveto(1.0)

    def button_control(self):
        if self.pair.trading_mode:
            self.btn_b['state'] = DISABLED
            self.btn_s['state'] = DISABLED
        else:
            self.btn_b['state'] = NORMAL
            self.btn_s['state'] = NORMAL

    # Button calls
    def switch_view(self):
        self.pair = self.pairs[self.var_w.get()]
        self.var_tm.set(self.pair.trading_mode)

        # Ubdate displaying of graphical and numerical data
        self.sublw['text'] = 'Subaccount #{}'.format(self.pair.idx)
        self.button_control()
        self.lbl_cu0['text'] = self.pair.u0
        self.lbl_cu1['text'] = self.pair.u1
        self.update_subacc()

    def switch_trading_mode(self):
        self.pair.trading_mode = self.var_tm.get()
        if self.pair.trading_mode: self.pair.prev_decision = None
        self.button_control()

    def manual_buy(self):
        self.after(10, self.trade, self.binance_obj.buy, self.pair)

    def manual_sell(self):
        self.after(10, self.trade, self.binance_obj.sell, self.pair)

    # Service methods
    def trade(self, trade_func, pair):
        msg = trade_func(pair)
        if msg is not None:
            self.text_msg(msg)
            self.update_balance(zip(pair.cos, self.binance_obj.balance(pair.u0, pair.u1)))
            self.update_subacc()

    def save_sql(self, commit=True):
        self.binance_obj.db_update()
        for pair in self.pairs: pair.db_update()


if __name__ == '__main__':
    # Settings
    tinth = 0.25
    reftime = '2020-7-22 0:14:0'
    timeshift = 5
    folder_db = 'DataBases/'
    market_pairs = (('ETH', 'USDT'),
                    ('XMR', 'USDT'),
                    ('BTC', 'USDT')
    )

    gui = GUI(timeshift, market_pairs, tinth, reftime, folder_db)
    gui.mainloop()
