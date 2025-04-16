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
from data_collector.utils import calc_adjusted_price, get_calendar_list


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
        if (interval == "1d"):
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
        self, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """
        Parameters
        ----------
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        bs.login()
        qlib.init()
        super(BaostockNormalizeHS3001d, self).__init__(date_field_name, symbol_field_name)

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust the price fields based on adjustment factors.
        """
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjustflag" in df:
            df["factor"] = df["adjustflag"].astype(float)
            df["factor"] = df["factor"].fillna(method="ffill")
        else:
            df["factor"] = 1
        for col in self.COLUMNS:
            if col not in df.columns:
                continue
            if col == "volume":
                df[col] = df[col] / df["factor"]
            else:
                df[col] = df[col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the data by adjusting prices and handling missing values.
        """
        if df.empty:
            return df
        df = self.adjusted_price(df)
        df = self._manual_adj_data(df)
        return df

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Manually adjust data: All fields (except change) are standardized according to the close of the first day.
        """
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        first_close = self._get_first_close(df)
        for col in df.columns:
            if col in [self._symbol_field_name, "change"]:
                continue
            if col == "volume":
                df[col] = df[col] * first_close
            else:
                df[col] = df[col] / first_close
        return df.reset_index()

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """
        Get the first valid close value for normalization.
        """
        df = df.loc[df["close"].first_valid_index():]
        return df["close"].iloc[0]

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """
        Generate a calendar list for 1d data.
        """
        # Use Qlib's calendar API to get the trading calendar
        return get_calendar_list("CSI300")


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
        """Download data from Baostock"""
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
    ):
        """Normalize data for 1d interval"""
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date
        )


if __name__ == "__main__":
    fire.Fire(Run)