#!/usr/bin/env python
from Queue import Queue
import sqlite3
from time import sleep, time
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from Common import DbUpdate


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1, zi=None):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    if zi is None: zi = sosfilt_zi(sos)
    return sosfilt(sos, data, zi=zi)

def butter_lowpass_filter(data, highcut, fs, order=1, zi=None):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, btype='lowpass', output='sos')
    if zi is None: zi = sosfilt_zi(sos)
    return sosfilt(sos, data, zi=zi)

def butter_highpass_filter(data, lowcut, fs, order=1, zi=None):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, btype='highpass', output='sos')
    if zi is None: zi = sosfilt_zi(sos)
    return sosfilt(sos, data, zi=zi)


class AmplitudePhaseAlgo(DbUpdate):
    """docstring for AmplitudePhaseAlgo"""
    def __init__(self, paramfile, coin, p_s, folder_db):
        self.coin = coin
        self.histcnt = 0
        self.decision = None

        with np.load(paramfile) as ifile:
            self.band_indices = ifile['band_indices']
            self.means = ifile['means'][self.band_indices]
            self.medians = ifile['medians'][self.band_indices]
            self.threshold = float(ifile['threshold'])
            self.order = int(ifile['order'])
            self.last_cutp = int(ifile['last_cutp'])
            self.steps_in_cut = int(ifile['steps_in_cut'])
            attenuation = float(ifile['attenuation'])
            perlen = int(ifile['perlen'])

        self.dim = len(self.band_indices)
        periods, bands, self.zo_s = self.filternorm(p_s)
        self.hist_zo_s = np.copy(self.zo_s)

        attenuation = float(attenuation)
        self.band_list, self.expcisnorn_s = [], []
        for band_idx in range(self.dim): # Iterate over bands
            period = periods[band_idx]
            viewlen = int(round(period * perlen))
            self.band_list.append(bands[band_idx, -viewlen:])
            normcoef = 2 / np.sum(np.exp(np.arange(1 - viewlen, 1) * attenuation / period))
            expcosisin = np.exp(np.arange(1 - viewlen, 1) * (attenuation - 1j * np.pi * 2) / period)
            self.expcisnorn_s.append(expcosisin * normcoef)

        # SQL
        self.sqlqueue = Queue()
        self.con = sqlite3.connect('{}AlgoAP.db'.format(folder_db), detect_types=sqlite3.PARSE_DECLTYPES) # Create or connect database
        self.cur = self.con.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, unixtime INT, rank REAL, decision INT);'.format(coin))

    def filternorm(self, p_s, zi_s=None):
        if zi_s is None: zi_s = [None] * (self.dim + 1)
        zo_s = []
        bands = np.empty((self.dim, p_s.size))
        periods = np.empty(self.dim)
        j = 0
        fs = 1.0 # Sampling frequency
        highcut = 0.5 * fs
        stepmul = 2 ** (1.0 / self.steps_in_cut) # Band width, times
        halfstepmul = 2 ** (0.5 / self.steps_in_cut) # Distance to the middle of band, times
        for i in range(max(self.band_indices) + 1):
            lowcut = highcut / stepmul
            if i % self.steps_in_cut == self.steps_in_cut - 1: lowcut = round(lowcut, 15)
            if i in self.band_indices:
                periods[j] = halfstepmul / highcut
                if i == 0: bands[j], zo = butter_highpass_filter(p_s, lowcut, fs, order=self.order, zi=zi_s[j])
                else: bands[j], zo = butter_bandpass_filter(p_s, lowcut, highcut, fs, order=self.order, zi=zi_s[j])
                zo_s.append(zo)
                j += 1
            highcut = lowcut
        
        lowfreq, zo = butter_lowpass_filter(p_s, fs / self.last_cutp, fs, order=self.order, zi=zi_s[-1]) # Baseline
        zo_s.append(zo)
        bands /= lowfreq # Normalize bands
        return periods, bands, zo_s

    def append(self, p_s):
        lenp_s = len(p_s)
        if lenp_s == 0: return
        self.histcnt += lenp_s
        _, bands, self.zo_s = self.filternorm(p_s, zi_s=self.zo_s)
        for band_idx in range(self.dim): # Iterate over bands
            if self.band_list[band_idx].size > lenp_s:
                self.band_list[band_idx] = np.roll(self.band_list[band_idx], -lenp_s)
                self.band_list[band_idx][-lenp_s:] = bands[band_idx]
            else: self.band_list[band_idx] = bands[band_idx, -self.band_list[band_idx].size:]

    def replace(self, p_s):
        lenp_s = len(p_s)
        if lenp_s == 0: return
        self.histcnt -= lenp_s
        _, bands, self.zo_s = self.filternorm(p_s, zi_s=self.hist_zo_s)
        self.hist_zo_s = self.zo_s
        for band_idx in range(self.dim): # Iterate over bands
            bandlen = self.band_list[band_idx].size
            dif = bandlen - lenp_s - self.histcnt
            self.band_list[band_idx][max(0,dif):bandlen-self.histcnt] = bands[band_idx][max(0,-dif):]

    def trade_decision(self):
        camps = [np.sum(self.band_list[band_idx] * self.expcisnorn_s[band_idx]) for band_idx in range(self.dim)]
        binmarks = [int(np.floor((np.angle(bca) * 4 / np.pi) % 8)) + np.where(np.abs(bca) >= self.medians[idx], 8, 0) for idx, bca in enumerate(camps)]
        self.rank = sum([self.means[i, binmarks[i]] for i in range(self.dim)])
        self.decision = bool(self.rank >= self.threshold)
        self.sqlqueue.put(('{} (unixtime, rank, decision)'.format(self.coin), (int(time()), self.rank, self.decision)))
        return self.decision


class APAlgoStub(object):
    """A stub if the algorithm is not needed"""
    def __init__(self): self.decision, self.rank = None, None

    def trade_decision(self): return None

    def append(self, _): pass

    def replace(self, _): pass

    def db_update(self, commit): pass


class PerfectTradeBase(object):
    """Base class for perfect trade based algorithm classes"""
    def __init__(self, fee_coef, threshold, buy_first):
        """Set attributes"""
        self.fee_coef = fee_coef
        self.threshold = threshold
        if buy_first:
            self.max_usd = 1
            self.max_eth = 0
            self.temp_is_buy = False
        else:
            self.max_usd = 0
            self.max_eth = 1
            self.temp_is_buy = True


class PerfectTradeBB(PerfectTradeBase):
    """Buy-buy based perfect trade algorithm"""
    def __init__(self, fee_coef, threshold, buy_first=True):
        super(PerfectTradeBB, self).__init__(fee_coef, threshold, buy_first)
        self.lbp = 1e-12 # Last buy price
        self.decision = True
        
    def __call__(self, p_s):
        """Take price(s) and execute trading"""
        if hasattr(p_s, '__iter__') is False: p_s = (p_s,)
        for price in p_s:
            test_usd = self.max_eth * price * self.fee_coef
            test_eth = self.max_usd * self.fee_coef / price
            if test_eth > self.max_eth:
                if self.temp_is_buy: self.temp_is_buy = False
                self.max_eth = test_eth
                self.temp_p = price
            elif test_usd > self.max_usd:
                if not self.temp_is_buy:
                    self.lbp = self.temp_p # Last buy price
                    self.temp_is_buy = True
                self.max_usd = test_usd
            if self.temp_is_buy is True: self.decision = True
            elif price / self.lbp < self.threshold: self.decision = False
        return self.decision


class PerfectTradeSS(PerfectTradeBase):
    """Sell-sell based perfect trade algorithm"""
    def __init__(self, fee_coef, threshold, buy_first=True):
        super(PerfectTradeSS, self).__init__(fee_coef, threshold, buy_first)
        self.lsp = float('inf') # Last sell price
        self.decision = False

    def __call__(self, p_s):
        """Take price(s) and execute trading"""
        if hasattr(p_s, '__iter__') is False: p_s = (p_s,)
        for price in p_s:
            test_usd = self.max_eth * price * self.fee_coef
            test_eth = self.max_usd * self.fee_coef / price
            if test_eth > self.max_eth:
                if self.temp_is_buy:
                    self.lsp = self.temp_p # Last sell price
                    self.temp_is_buy = False
                self.max_eth = test_eth
            elif test_usd > self.max_usd:
                if not self.temp_is_buy: self.temp_is_buy = True
                self.max_usd = test_usd
                self.temp_p = price
            if self.temp_is_buy is False: self.decision = False
            elif price / self.lsp > self.threshold: self.decision = True
        return self.decision


class PerfectTradeAlgo(DbUpdate):
    """docstring for PerfectTradeAlgo"""
    def __init__(self, paramfile, coin, folder_db, p_s=None):
        self.coin = coin

        with np.load(paramfile) as ifile:
            params = ifile['params']
            ptmeans = ifile['ptmeans']
            self.threshold = ifile['threshold']

        algo_tps = (PerfectTradeBB, PerfectTradeSS)
        insts = [algo_tps[int(param[0])](1 / (1.0 + param[1]), param[2]) for param in params]
        self.instmeanzip = zip(insts, ptmeans)

        # SQL
        self.sqlqueue = Queue()
        self.con = sqlite3.connect('{}AlgoPT.db'.format(folder_db), detect_types=sqlite3.PARSE_DECLTYPES) # Create or connect database
        self.cur = self.con.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, unixtime INT, rank REAL, decision INT);'.format(coin))

        if p_s is not None:
            self.rank = np.sum([ptmean[int(inst(p_s))] for inst, ptmean in self.instmeanzip])
            self.decision = bool(self.rank >= self.threshold)
            self.sqlqueue.put(('{} (unixtime, rank, decision)'.format(self.coin), (int(time()), self.rank, self.decision)))

    def __call__(self, price):
        self.rank = np.sum([ptmean[int(inst(price))] for inst, ptmean in self.instmeanzip])
        self.decision = bool(self.rank >= self.threshold)
        self.sqlqueue.put(('{} (unixtime, rank, decision)'.format(self.coin), (int(time()), self.rank, self.decision)))
        return self.decision


class PTAlgoStub(object):
    """A stub if perfect trade algorithm is not needed"""
    def __init__(self): self.decision, self.rank = None, None

    def __call__(self, price): return None

    def db_update(self, commit): pass


if __name__ == '__main__':
    # Load data
    with np.load('../../Web Scraper/BotObormot/HistoricalData/ETH/1610347440.npz') as ifile: p_s = ifile['p_s']

    algoap = AmplitudePhaseAlgo('ParamsAlgoAP/ETH/params21.npz', 'ETH', p_s, 'DataBases/')
    algopt = PerfectTradeAlgo('ParamsAlgoPT/ETH/params0.npz', 'ETH', 'DataBases/', p_s)
    print('It works!')
