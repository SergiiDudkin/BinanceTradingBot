#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import urllib3
import sqlite3
from Queue import Queue
from datetime import datetime, timedelta
from time import time, sleep
import pytz
import os
import functools
import pickle
import numpy as np
from scipy.interpolate import interp1d
from PIL import ImageTk, Image
from Common import DbUpdate, OP_SYS
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from threading import Lock
import json
# from scipy.ndimage import zoom
# import cv2


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class CoinNotFound(Exception):
    pass


class InconsistentRates(Exception):
    pass


class BrowserRequestFailed(Exception):
    pass


def exceptiondecor(func):
    """Decorator to handle bad http responses"""
    name = func.__name__
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try: return func(*args, **kwargs)
        except Exception as e:
            print('{} failed.'.format(name))
            return e
    return wrapper

def coingecko_coin_id(*coins):
    """Scrapes coingecko_id"""
    coin_list = requests.get('http://api.coingecko.com/api/v3/coins/list').json()
    symbols = [coin['symbol'] for coin in coin_list]
    return [coin_list[symbols.index(coin.lower())]['id'] for coin in coins]

def cmc_coin_data(*coins):
    """Scrapes the following coin data: cmc_time_start, cmc_id, cmc_rank"""
    # payload = {'start': 1,
    #            'limit': 20
    # }
    # # data = requests.get('https://web-api.coinmarketcap.com/v1/cryptocurrency/listings/latest', params=payload, headers=dict(cmc_headers, **{'Cache-Control': 'max-age=0'})).json()['data']
    browser = SingleBrowser(executable_path=PATH, options=OPTIONS)
    data = browser.cmc_curr_req(1, 1500)['data']
    symbols = [item['symbol'] for item in data]
    selected = [data[symbols.index(coin)] for coin in coins]
    return [((datetime.strptime(item['date_added'], '%Y-%m-%dT%H:%M:%S.000Z').replace(tzinfo=UTC_TZ) - DATE1970UTC).total_seconds(), item['id'], item['cmc_rank']) for item in selected]

def update_coindict(*coins):
    """Creates a new record or updates the old one in the following dictionary: {coin: (cmc_time_start, cmc_id, cmc_rank, coingecko_id)}"""
    coingecko_list = coingecko_coin_id(*coins)
    cmc_list = cmc_coin_data(*coins)
    vals = [cmc + (coingecko,) for cmc, coingecko in zip(cmc_list, coingecko_list)]
    coindict.update(zip(coins, vals)) # {coin: (cmc_time_start, cmc_id, cmc_rank, coingecko_id)}

    # Save updates
    with open('coindict.pkl', 'wb') as output: pickle.dump(coindict, output, pickle.HIGHEST_PROTOCOL)

# def download_images(*coins):
#     """Downloads graphical coin symbols from Coinmarketcup and resizes them"""
#     for coin in coins:
#         with open('Images/Coins/{}_b.png'.format(coin), 'wb') as ofile: ofile.write(requests.get('https://s2.coinmarketcap.com/static/img/coins/64x64/{}.png'.format(coindict[coin][1])).content)
#         img64 = cv2.imread('Images/Coins/{}_b.png'.format(coin), cv2.IMREAD_UNCHANGED)
#         cv2.imwrite('Images/Coins/{}_m.png'.format(coin), cv2.resize(img64, (32, 32), interpolation=cv2.INTER_AREA))
#         cv2.imwrite('Images/Coins/{}_s.png'.format(coin), cv2.resize(img64, (16, 16), interpolation=cv2.INTER_AREA))

def download_images(*coins):
    for coin in coins:
        with open('Images/Coins/{}_b.png'.format(coin), 'wb') as ofile: ofile.write(requests.get('https://s2.coinmarketcap.com/static/img/coins/64x64/{}.png'.format(coindict[coin][1])).content)
        img64 = np.asarray(Image.open('Images/Coins/{}_b.png'.format(coin)).convert('RGBA'))
        img32 = downscale_half(img64)
        Image.fromarray(img32).save('Images/Coins/{}_m.png'.format(coin))
        img16 = downscale_half(img32)
        Image.fromarray(img16).save('Images/Coins/{}_s.png'.format(coin))

def downscale_half(img):
    return ((img[0::2, 0::2].astype(np.uint16) + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) / 4.0).astype(np.uint8) # img = zoom(img, (0.5, 0.5, 1.0), order=1)

def cmc_download(coin, save_nointerp=False):
    """Downloads entire history from Coinmarketcap"""
    folder = 'HistoricalData/{}'.format(coin)
    if not os.path.exists(folder): os.makedirs(folder)
    t_s, p_s, v_s = cmc_history_scraper(coin, save_nointerp=save_nointerp, delay=330)
    file_name = '{}.npz'.format(int(t_s[-1]))
    np.savez('{}/{}'.format(folder, file_name), t_s=t_s, p_s=p_s, v_s=v_s)
    return file_name

def cmc_history_scraper(coin, time_start=None, time_end=None, save_nointerp=False, delay=0):
    """Scrapes historical data of the given coin from Coinmarketcap"""
    if time_start is None: time_start = coindict[coin][0]
    if time_end is None: time_end = int(time())

    tinth = 0.25
    MAX_PERIOD = 3600 * 24 * 100 # 100 days

    # Initialize start and end timestamps
    t_st = time_start
    t_en = min(t_st + MAX_PERIOD, time_end)

    # Generate historical data time frames (1 frame => 1 request)
    frames = []
    while t_st < t_en:
        frames.append((t_st, t_en))
        t_st = t_en
        t_en = min(t_st + MAX_PERIOD, time_end)

    # Inform user about waiting time
    delay_cnt = len(frames) - 1
    if delay_cnt and delay:
        msg = """
        WARNING: The Coinmarketcap does not likes to be scraped.
        To overcome this problem, artificial delays are used here.
        Please be patient. The download of {} historical data will
        be completed within {}.
        """.format(coin, timedelta(seconds=delay*delay_cnt))
        print(msg)

    rearranged = frames[3::4] + frames[2::4] + frames[1::4] + frames[::4] # Confuse cmc

    # Requests
    browser = SingleBrowser(executable_path=PATH, options=OPTIONS)
    accum = {}
    for idx, (t_st, t_en) in enumerate(rearranged): # Request loop (with artificial delays)
        res = browser.cmc_hist_req(coindict[coin][1], '{}m'.format(int(tinth * 60)), int(t_st), int(t_en))
        accum.update(res['data'])
        if idx < delay_cnt:
            sleep(delay)
            print('{} of {}'.format(idx + 1, delay_cnt))

    # Arrange data into sorted list
    keys = accum.keys()
    keys.sort()
    crude = []
    for key in keys:
        usd = accum[key]['USD']
        btc = accum[key]['BTC']
        crude.append([key] + btc + usd) # time [price_BTC] [price_USD voume_USD cap_USD]

    # Prettify
    utcs, prices, volumes, prbtc, cap = [], [], [], [], []
    for time_, price_BTC, price_USD, voume_USD, cap_USD in crude:
        timestamp = (datetime.strptime(time_, '%Y-%m-%dT%H:%M:%S.000Z').replace(tzinfo=UTC_TZ) - DATE1970UTC).total_seconds()
        utcs.append(timestamp)
        prices.append(price_USD)
        volumes.append(voume_USD)
        prbtc.append(price_BTC)
        cap.append(cap_USD)

    if save_nointerp: np.savez('HistoricalData/{}/nointerp'.format(coin), utcs=utcs, prices=prices, volumes=volumes, prbtc=prbtc, cap=cap)

    # Interpolate and return
    return tpv_interpolate(utcs, prices, volumes, tinth)

def tpv_interpolate(utcs, prices, volumes, tinth):
    """Interpolate times, prices and volumes"""
    if len(utcs) == 0: return np.array([]), np.array([]), np.array([])

    reftime = datetime.strptime('2020-7-22, 0:14:0', '%Y-%m-%d, %H:%M:%S')
    REF_TS = (reftime - datetime(1970, 1, 1, 3)).total_seconds()
    tints = int(tinth * 3600)

    t_st = int(((utcs[0] - REF_TS) // tints) * tints + REF_TS)
    if t_st < utcs[0]: t_st += tints
    t_en = int(((utcs[-1] - REF_TS) // tints) * tints + REF_TS)

    if t_st > t_en: # Corner case
        print('Corner case')
        i_arrs, o_arrs = (utcs, prices, volumes), (np.array([]), np.array([]), np.array([]))
        if (utcs[0] - t_en) < tints * 0.1:
            for i_arr, o_arr in zip(i_arrs, o_arrs): np.append(o_arr, i_arr[0])
        if (t_st - utcs[-1]) < tints * 0.1:
            for i_arr, o_arr in zip(i_arrs, o_arrs): np.append(o_arr, i_arr[-1])
        return o_arrs

    t_s = np.arange(t_st, t_en + 1, tints, dtype=np.int64)
    price_interp_func = interp1d(utcs, prices, kind='nearest')
    volume_interp_func = interp1d(utcs, volumes, kind='nearest')

    return t_s, price_interp_func(t_s), volume_interp_func(t_s)

@exceptiondecor
def coingecko_json(coin):
    """Scrapes the last price of given coin from Coingecko"""
    ids = coindict[coin][3]
    payload = {'ids': ids,
               'vs_currencies': 'usd',
               'include_24hr_vol': 'true',
               'include_last_updated_at': 'true'
    }
    res = requests.get('http://api.coingecko.com/api/v3/simple/price', params=payload).json()[ids]
    return res['last_updated_at'], res['usd'], res['usd_24h_vol']

@exceptiondecor
def investing_html(coin):
    """Scrapes the last price of given coin from Investing.com"""
    # headers = {'Host': 'www.investing.com',
    #            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0',
    #            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    #            'Accept-Language': 'en-US,en;q=0.5',
    #            'Accept-Encoding': 'gzip, deflate, br',
    #            'DNT': '1',
    #            'Connection': 'keep-alive',
    #            'Cookie': 'adBlockerNewUserDomains=1595881920; PHPSESSID=c666cmpirlnn99r7io97rufip0; geoC=UA; prebid_page=0; prebid_session=0; nyxDorf=OT0yZjJsNnRlMmBpNGZifmI0Zj82L2FiYGNnZA%3D%3D; StickySession=id.12349219030.551_www.investing.com; usprivacy=1---; notice_behavior=none; adbBLk=1; G_ENABLED_IDPS=google; gtmFired=OK',
    #            'Upgrade-Insecure-Requests': '1',
    #            'Cache-Control': 'max-age=0'
    # }
    
    # src = requests.get('https://www.investing.com/crypto/currencies', headers=headers).content
    # soup = BeautifulSoup(src, 'lxml')
    
    # tabrow = soup.find('table', class_='genTbl openTbl js-all-crypto-table mostActiveStockTbl crossRatesTbl allCryptoTlb wideTbl elpTbl elp15').find('td', title=coin).parent
    # price = float(tabrow.find('td', class_='price js-currency-price').text.replace(',', ''))
    # volume = float(tabrow.find('td', class_='js-24h-volume').text[1:-1].replace(',', '')) * 1e9
    # return int(time()), price, volume
    browser = SingleBrowser(executable_path=PATH, firefox_profile=PROFILE) #, options=OPTIONS
    res = browser.investing_curr_req(coin)
    return res

@exceptiondecor
def cmc_json(coin):
    """Scrapes the last price of given coin from Coinmarketcap"""
    # payload = {'start': coindict[coin][2],
    #            'limit': 1
    # }
    # r = requests.get('https://web-api.coinmarketcap.com/v1/cryptocurrency/listings/latest', params=payload, headers=dict(cmc_headers, **{'Cache-Control': 'max-age=0'}))
    # res = r.json()
    browser = SingleBrowser(executable_path=PATH, options=OPTIONS)
    res = browser.cmc_curr_req(coindict[coin][2], 1)
    if res['data'][0]['symbol'] == coin: parseobj = res['data'][0]['quote']['USD']
    else:
        # payload = {'start': max(1, coindict[coin][2] - 300),
        #            'limit': coindict[coin][2] + 300
        # }
        # r = requests.get('https://web-api.coinmarketcap.com/v1/cryptocurrency/listings/latest', params=payload, headers=dict(cmc_headers, **{'Cache-Control': 'max-age=0'}))
        # res = r.json()
        res = browser.cmc_curr_req(max(1, coindict[coin][2] - 300), min(600, coindict[coin][2] + 300))
        for idx, item in enumerate(res['data']):
            if item['symbol'] == coin:
                coindict[coin] = coindict[coin][:2] + (res['data'][idx]['cmc_rank'],) + coindict[coin][3:]
                with open('coindict.pkl', 'wb') as output: pickle.dump(coindict, output, pickle.HIGHEST_PROTOCOL) # Save updates
                parseobj = res['data'][idx]['quote']['USD']
                break
        else: raise CoinNotFound
    
    t = (datetime.strptime(parseobj['last_updated'], '%Y-%m-%dT%H:%M:%S.000Z').replace(tzinfo=UTC_TZ) - DATE1970UTC).total_seconds()
    p = float(parseobj['price'])
    v = float(parseobj['volume_24h'])
    return t, p, v

@exceptiondecor
def i_html():
    """Scrapes the last USD-UAH exchange rate from i.ua"""
    curr_time = int(time())
    headers = {'Host': 'finance.i.ua',
               'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:79.0) Gecko/20100101 Firefox/79.0',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Accept-Language': 'en-US,en;q=0.5',
               'Accept-Encoding': 'gzip, deflate, br',
               'DNT': '1',
               'Connection': 'keep-alive',
               'Cookie': 'user_currencies=-1; mcs=1',
               'Upgrade-Insecure-Requests': '1',
               'Cache-Control': 'max-age=0'
    }
    src = requests.get('https://finance.i.ua/', headers=headers, verify=False).content # , verify=False
    soup = BeautifulSoup(src, 'lxml')
    # prices = soup.find('div', class_='widget-currency_bank').find('th', string='USD').parent.find_all('td') + soup.find('div', class_='widget-currency_cash').find('th', string='USD').parent.find_all('td') # Obsolete 0, max(0, 3), min(1, 4)
    # prices = soup.find('a', string='черный рынок').parent.parent.find('b', string='USD').parent.parent.find_all('big') + soup.find('h2', string='Средний курс валют в банках').parent.find('b', string='USD').parent.parent.find_all('big') # Obsolete 1, max(0, 2), min(1, 3)
    prices = soup.find('div', class_='widget-currency_cash').find('th', string='USD').parent.find_all('span', class_=None) + soup.find('div', class_='widget-currency_bank').find('th', string='USD').parent.find_all('span', class_=None)
    usdrate = [float(price.text) for price in prices]
    if np.std(usdrate[:4]) / np.mean(usdrate[:4]) > 0.1: raise InconsistentRates
    true_usdrate = (max(usdrate[0], usdrate[2]) + min(usdrate[1], usdrate[3])) / 2
    return curr_time, true_usdrate

@exceptiondecor
def minfin_html():
    """Scrapes the last USD-UAH exchange rate from minfin.com.ua"""
    curr_time = int(time())
    headers = {'Host': 'minfin.com.ua',
               'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:79.0) Gecko/20100101 Firefox/79.0',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Accept-Language': 'en-US,en;q=0.5',
               'Accept-Encoding': 'gzip, deflate',
               'DNT': '1',
               'Connection': 'keep-alive',
               'Cookie': 'ghost=true; __cfduid=d7b2621501fb35f540e5aa9a517e03d801598522952; minfin_sessions=9999fe4b7754f4550b35480c432ea2da2105f856; minfincomua_region=1000',
               'Upgrade-Insecure-Requests': '1',
               'Cache-Control': 'max-age=0',
               'TE': 'Trailers'
    }
    src = requests.get('https://minfin.com.ua/ua/currency/banks/usd/', headers=headers).content
    soup = BeautifulSoup(src, 'lxml')
    prices = soup.find('table', class_="table-response mfm-table mfcur-table-lg-banks mfcur-table-lg").find_all('td', class_="mfm-text-nowrap")
    usdrate = [float(price.contents[i]) for price in prices for i in [0, 2]]
    if np.std(usdrate[:4]) / np.mean(usdrate[:4]) > 0.1: raise InconsistentRates
    true_usdrate = (max(usdrate[0], usdrate[2]) + min(usdrate[1], usdrate[3])) / 2
    return curr_time, true_usdrate

# cmc_headers = {'Accept': 'application/json, text/plain, */*',
#                'Accept-Encoding': 'gzip, deflate, br',
#                'Accept-Language': 'en-US,en;q=0.5',
#                'Connection': 'keep-alive',
#                'DNT': '1',
#                'Host': 'web-api.coinmarketcap.com',
#                'Origin': 'https://coinmarketcap.com',
#                'Referer': 'https://coinmarketcap.com/',
#                'TE': 'Trailers',
#                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:79.0) Gecko/20100101 Firefox/79.0'
# }


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances: cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingleBrowser(webdriver.Firefox):
    """Singletone of driven Firefox browser"""
    __metaclass__ = Singleton

    def __init__(self, *args, **kwargs):
        super(SingleBrowser, self).__init__(*args, **kwargs)
        self.URL_CMC_HIST = 'https://web-api.coinmarketcap.com/v1.1/cryptocurrency/quotes/historical?convert=USD%2CBTC&time_start={2}&format=chart_crypto_details&interval={1}&time_end={3}&id={0}'
        self.URL_CMC_CURR = 'https://web-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?start={0}&limit={1}'
        self.URL_INV_CURR = 'https://www.investing.com/crypto/currencies'
        self.ORDER_DICT = {'K': 3, 'M': 6, 'B': 9, 'T': 12}
        self._lock = Lock()

    def cmc_hist_req(self, coin_id, interval, t_st, t_en):
        """Download historical data from Coinmarketcap"""
        return self.json_req(self.URL_CMC_HIST.format(coin_id, interval, t_st, t_en))

    def cmc_curr_req(self, rank_st, limit):
        """Download current data from Coinmarketcap"""
        return self.json_req(self.URL_CMC_CURR.format(rank_st, limit))

    def json_req(self, url):
        """Request json data and convert it to python object"""
        with self._lock:
            try:
                self.get(url)
                if OP_SYS == 'Linux':
                    btn = WebDriverWait(self, 1).until(EC.presence_of_element_located((By.ID, 'rawdata-tab')))
                    btn.click()
                    data = WebDriverWait(self, 1).until(EC.presence_of_element_located((By.CLASS_NAME, 'data')))
                elif OP_SYS == 'Windows':
                    data = WebDriverWait(self, 1).until(EC.presence_of_element_located((By.TAG_NAME, 'pre')))
                return json.loads(data.text)
            except:
                print('Coinmarketcap request failed! URL:\n{}'.format(url))
                raise BrowserRequestFailed

    def investing_curr_req(self, coin):
        """Request current HTML data from Investing.com"""
        with self._lock:
            try:
                self.get(self.URL_INV_CURR)
                table = self.find_element_by_class_name('genTbl')
                row = table.find_element_by_css_selector("td[title='{}']".format(coin)).find_element_by_xpath('..')
                sprice = row.find_element_by_class_name('price').text
                svolume = row.find_element_by_class_name('js-24h-volume').text
                price = float(sprice.replace(',', ''))
                svolume = svolume.strip('$')
                if svolume[-1].isalpha(): svolume = '{}e{}'.format(svolume[:-1], self.ORDER_DICT[svolume[-1]])
                volume = float(svolume)
                return int(time()), price, volume
            except:
                print('Investing.com request failed! URL:\n{}'.format(self.URL_INV_CURR))
                raise BrowserRequestFailed


class ScrapeTPV(DbUpdate):
    """Collects time, price and volume data of the given coin"""
    def __init__(self, coin, forder_hist, folder_db):
        self.coin = coin.upper()
        self.currency_l = coin.lower()
        self.forder_hist = forder_hist + coin + '/'
        self.funclist = [coingecko_json, cmc_json, investing_html, self.error_stub]
        if coin == 'USDT': self.funclist[1], self.funclist[2] = self.funclist[2], self.funclist[1]
        self.minsize = 8192

        # Set image
        self.tkimage = ImageTk.PhotoImage(Image.open('Images/Coins/{}_m.png'.format(coin)))

        # Load data
        self.t_s = np.array([])
        self.p_s = np.array([])
        self.v_s = np.array([])
        files = sorted(os.listdir(self.forder_hist))
        if not files:
            print('No historical data found in the local forder. Please wait intil download is finished.')
            files = [cmc_download(coin)] # Download the entire history, if there is no data yet
        for input_f in files:
            with np.load(self.forder_hist + input_f) as data:
                self.t_s = np.append(self.t_s, data['t_s'])
                self.p_s = np.append(self.p_s, data['p_s'])
                self.v_s = np.append(self.v_s, data['v_s'])
        check_time_consistency(self.t_s, msg='Time consistency {} __init__: {}'.format(self.coin, '{}'))
        self.hyst_timing()
        self.append()

        # SQL
        self.sqlqueue = Queue()
        self.con = sqlite3.connect('{}ServerTPV.db'.format(folder_db), detect_types=sqlite3.PARSE_DECLTYPES) # Create or connect database
        self.cur = self.con.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS {}15m (id INTEGER PRIMARY KEY, req_ts INT, source TEXT, time_ INT, price REAL, volume REAL);'.format(self.currency_l))

    def hyst_timing(self, en_in=0):
        """Records the last historical data and sets time of the next history update"""
        self.last_hist = int(self.t_s[en_in-1])
        self.next_update = self.last_hist + 14400 * (1 + np.random.rand())

    def append(self):
        """Adds the latest historical data to previously collected one"""
        new_ts, new_ps, new_vs = cmc_history_scraper(self.coin, time_start=self.t_s[-1]-900)
        if new_ts.size == 0: return
        while new_ts[0] <= self.t_s[-1]: # In case of overlaping
            new_ts, new_ps, new_vs = new_ts[1:], new_ps[1:], new_vs[1:] # remove the first items
            if new_ts.size == 0: return

        self.t_s = np.append(self.t_s, new_ts)
        self.p_s = np.append(self.p_s, new_ps)
        self.v_s = np.append(self.v_s, new_vs)
        self.hyst_timing()

        # Save the updated tpv data
        if not(new_ts.size == new_ps.size == new_vs.size): print('Append. Size inconsistency!', self.last_hist) # Debug
        np.savez('{}{}.npz'.format(self.forder_hist, self.last_hist), t_s=new_ts, p_s=new_ps, v_s=new_vs)

    def replace(self):
        """Replaces live data with the more accurate historical data"""
        try: new_ts, new_ps, new_vs = cmc_history_scraper(self.coin, time_start=self.last_hist-900)
        except: return np.array([])
        if new_ts.size == 0: return np.array([])
        while new_ts[0] <= self.last_hist: # In case of overlaping
            new_ts, new_ps, new_vs = new_ts[1:], new_ps[1:], new_vs[1:] # remove the first item
            if new_ts.size == 0: return np.array([])

        st_in = np.searchsorted(self.t_s, self.last_hist) + 1
        en_in = min(st_in + new_ts.size, self.minsize)
        insrtlen = self.minsize - st_in

        self.t_s[st_in:en_in] = new_ts[:insrtlen]
        self.p_s[st_in:en_in] = new_ps[:insrtlen]
        self.v_s[st_in:en_in] = new_vs[:insrtlen]
        self.hyst_timing(en_in)

        check_time_consistency(new_ts, msg='History update\nTime consistency {} new_ts: {}'.format(self.coin, '{}'))
        check_time_consistency(self.t_s[:en_in], msg='Time consistency {} self.t_s: {}'.format(self.coin, '{}'))

        # Save the updated tpv data
        if not(new_ts.size == new_ps.size == new_vs.size): print('Replace. Size inconsistency!', self.last_hist) # Debug
        np.savez('{}{}.npz'.format(self.forder_hist, self.last_hist), t_s=new_ts, p_s=new_ps, v_s=new_vs)

        return self.p_s[st_in:en_in]

    def error_stub(self, coin):
        """In case if scraping fails, repeats the last price and volume record with present time"""
        print('Error! Cannot scrape TPV data.')
        return int(time()), self.p_s[-1], self.v_s[-1]

    def update(self):
        """Sctapes the latest time, price and volume from one of the alternative web resources"""
        for scrapefunc in self.funclist:
            res = scrapefunc(self.coin)
            if isinstance(res, Exception): print(repr(res))
            elif res[0] <= self.t_s[-1]: print('Time error: {}, res[0] {}, self.t_s[-1] {}'.format(self.coin, res[0], self.t_s[-1]))
            else:
                lastreqtime = int(time())
                t, p, v = res
                self.t_s = np.roll(self.t_s, -1)
                self.t_s[-1] = t
                self.p_s = np.roll(self.p_s, -1)
                self.p_s[-1] = p
                self.v_s = np.roll(self.v_s, -1)
                self.v_s[-1] = v
                param_tuple = (lastreqtime, scrapefunc.__name__, t, p, v)
                self.sqlqueue.put(('{}15m (req_ts, source, time_, price, volume)'.format(self.currency_l), param_tuple))
                break

    def cut_data(self):
        """Shrink data size"""
        self.t_s = self.t_s[-self.minsize:]
        self.p_s = self.p_s[-self.minsize:]
        self.v_s = self.v_s[-self.minsize:]


def check_time_consistency(seq, msg=''):
    is_consistent = np.all(seq[1:] - seq[:-1] == 900)
    if msg: print(msg.format('Ok!' if is_consistent else 'Not Ok!'))
    return is_consistent


class ScrapeFiat(DbUpdate):
    def __init__(self, folder_db):
        self.funclist = [i_html, minfin_html]
        self.currfuncidx = 0

        # SQL
        self.sqlqueue = Queue()
        self.con = sqlite3.connect('{}ServerTPV.db'.format(folder_db), detect_types=sqlite3.PARSE_DECLTYPES) # Create or connect database
        self.cur = self.con.cursor()
        self.cur.execute('CREATE TABLE IF NOT EXISTS usd_rate (id INTEGER PRIMARY KEY, unixtime INT, rate REAL);')

    def get_usduah_rate(self):
        for _ in range(2):
            scrapefunc = self.funclist[self.currfuncidx]
            res = scrapefunc()
            if isinstance(res, Exception):
                print(repr(res))
                self.currfuncidx = (self.currfuncidx + 1) % 2
            else:
                curr_time, true_usdrate = res
                self.true_usdrate = true_usdrate
                self.sqlqueue.put(('usd_rate (unixtime, rate)', (curr_time, true_usdrate)))
                return true_usdrate
        print('Error! Cannot scrape USD-UAH exchange rate.')
        return None


# Define module constants
UTC_TZ = pytz.timezone('UTC')
DATE1970UTC = datetime(1970, 1, 1).replace(tzinfo=UTC_TZ)
REF_TS = (datetime.strptime('2020-7-22 0:14:0', '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1, 3)).total_seconds()
if OP_SYS == 'Linux': 
    PATH = '/home/astatin/Sandbox/venv/lib/python2.7/site-packages/FirefoxWebdriver/geckodriver'
    ADDON_PATHS = () #('/home/astatin/.mozilla/firefox/iiz4al9z.default-release-1/extensions/{73a6fe31-595d-460b-a920-fcc0f8843232}.xpi')
elif OP_SYS == 'Windows':
    PATH = 'C:\\Python27\\lib\\site-packages\\FirefoxWebdriver\\geckodriver.exe'
    ADDON_PATHS = ()
    # ADDON_PATHS = ('C:\\Users\\a\\AppData\\Local\\Temp\\rust_mozprofile.2WBKWDPYF1C6\\extensions\\{660EBB8E-4E21-11E9-9A47-13E64A802C61}.xpi',
    #             'C:\\Users\\a\\AppData\\Local\\Temp\\rust_mozprofile.2WBKWDPYF1C6\\extensions\\{60B7679C-BED9-11E5-998D-8526BB8E7F8B}.xpi')
OPTIONS = webdriver.FirefoxOptions()
OPTIONS.headless = True
PROFILE = webdriver.FirefoxProfile()
# PROFILE.set_preference('browser.download.folderList', 2)
# PROFILE.set_preference('javascript.enabled', False)

for path in ADDON_PATHS: PROFILE.add_extension(path)
# OPTIONS.add_argument('--headless')
# OPTIONS.add_argument('--disable-gpu')
# os.environ['MOZ_HEADLESS'] = '1'


# Load data
with open('coindict.pkl', 'rb') as ifile: coindict = pickle.load(ifile)


if __name__ == "__main__":
    t, p, v = investing_html('USDT')
    print('time {}, price {} USD, 24h volume {}*10^9 USD'.format(datetime.fromtimestamp(t), p, v / 1e9))
