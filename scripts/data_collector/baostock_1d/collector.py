# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import copy
import fire
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Iterable, List

import qlib
from qlib.data import D

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import calc_adjusted_price


class BaostockCollectorHS3001d(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """
        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1d], default 1d
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        bs.login()
        super(BaostockCollectorHS3001d, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )  

    @staticmethod
    def process_interval(interval: str):
        if interval == "1d":
            return {"interval": "d", "fields": "date,code,open,high,low,close,volume,amount,adjustflag"}

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = self.get_data_from_remote(
            symbol=symbol, interval=interval, start_datetime=start_datetime, end_datetime=end_datetime
        )
        if df.empty:
            logger.warning(f"No data retrieved for symbol: {symbol}")
            return df  # Return the empty DataFrame if no data is retrieved

        df.columns = ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "adjustflag"]
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["symbol"] = df["symbol"].map(lambda x: str(x).replace(".", "").upper())
        return df

    @staticmethod
    def get_data_from_remote(
        symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        rs = bs.query_history_k_data_plus(
            symbol,
            BaostockCollectorHS3001d.process_interval(interval=interval)["fields"],
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency=BaostockCollectorHS3001d.process_interval(interval=interval)["interval"],
            adjustflag="3",
        )
        if rs.error_code == "0" and len(rs.data) > 0:
            data_list = rs.data
            columns = rs.fields
            df = pd.DataFrame(data_list, columns=columns)
        return df

    def get_hs300_symbols(self) -> List[str]:
        hs300_stocks = []
        rs = bs.query_hs300_stocks()
        while rs.error_code == "0" and rs.next():
            hs300_stocks.append(rs.get_row_data())
        return sorted({e[1] for e in hs300_stocks})

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = self.get_hs300_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol: str):
        return str(symbol).replace(".", "").upper()


class BaostockNormalizeHS3001d(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]

    def __init__(
        self, qlib_data_1d_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """
        Parameters
        ----------
        qlib_data_1d_dir: str, Path
            the qlib data to be updated for yahoo, usually from: Normalised to 1d using local 1d data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        bs.login()
        qlib.init(provider_uri=qlib_data_1d_dir)
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")
        super(BaostockNormalizeHS3001d, self).__init__(date_field_name, symbol_field_name)

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="1d",
            _1d_data_all=self.all_1d_data,
        )
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # adjusted price
        df = self.adjusted_price(df)
        return df

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """
        Generate a list of trading dates for normalization.
        """
        return D.calendar(freq="day")


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region="HS300"):
        """
        Changed the default value of: scripts.data_collector.base.BaseRun.
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"BaostockCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"BaostockNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """download data from Baostock"""
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):

        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )


if __name__ == "__main__":
    fire.Fire(Run)