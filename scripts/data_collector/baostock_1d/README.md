## Collector Data

### Get Qlib data(`bin file`)

  - get data: `python scripts/get_data.py qlib_data`
  - parameters:
    - `target_dir`: save dir, by default *~/.qlib/qlib_data/cn_data_1d*
    - `version`: dataset version, value from [`v2`], by default `v2`
      - `v2` end date is *2022-12*
    - `interval`: `1d`
    - `region`: `hs300`
    - `delete_old`: delete existing data from `target_dir`(*features, calendars, instruments, dataset_cache, features_cache*), value from [`True`, `False`], by default `True`
    - `exists_skip`: target_dir data already exists, skip `get_data`, value from [`True`, `False`], by default `False`
  - examples:
    ```bash
    # hs300 1d
    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/hs300_data_1d --region hs300 --interval 1d
    ```
    
### Collector *Baostock daily frequency* data to qlib
> collector *Baostock daily frequency* data and *dump* into `qlib` format.
> If the above ready-made data can't meet users' requirements, users can follow this section to crawl the latest data and convert it to qlib-data.
  1. download data to csv: `python scripts/data_collector/baostock_1d/collector.py download_data`
     
     This will download the raw data such as date, symbol, open, high, low, close, volume, amount, adjustflag from baostock to a local directory. One file per symbol.
     - parameters:
          - `source_dir`: save the directory
          - `interval`: `1d`
          - `region`: `HS300`
          - `start`: start datetime, by default *None*
          - `end`: end datetime, by default *None*
     - examples:
          ```bash
          # cn 1d data
          python collector.py download_data --source_dir ~/.qlib/stock_data/source/hs300_1d_original --start 2022-01-01 --end 2022-01-30 --interval 1d --region HS300
          ```
          ```
          python baostock_1d/collector.py download_data --source_dir D:/Codes/Data/stock_data/source/hs300_1d_original --start 2025-01-01 --end 2025-04-15 --interval 1d --region HS300
          python baostock_1d/collector.py download_data --source_dir D:/ProgramCoding/Data/stock_data/source/hs300_1d_original --start 2020-01-01 --end 2025-04-16 --interval 1d --region HS300
          ```
  2. normalize data: `python scripts/data_collector/yahoo/collector.py normalize_data`
     
     This will:
     1. Normalize high, low, close, open price using adjclose.
     2. Normalize the high, low, close, open price so that the first valid trading date's close price is 1. 
     - parameters:
          - `source_dir`: csv directory
          - `normalize_dir`: result directory
          - `interval`: `1d`
          - `region`: `HS300`
          - `date_field_name`: column *name* identifying time in csv files, by default `date`
          - `symbol_field_name`: column *name* identifying symbol in csv files, by default `symbol`
          - `end_date`: if not `None`, normalize the last date saved (*including end_date*); if `None`, it will ignore this parameter; by default `None`
          - `qlib_data_1d_dir`: qlib directory(1d data)
            ```bash
                # qlib_data_1d can be obtained like this:
                python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn --version v3
            ```
      - examples:
        ```bash
        # normalize 1d cn
        ```
        python yahoo/collector.py normalize_data --source_dir D:/Codes/Data/stock_data/source/hs300_1d_original --normalize_dir D:/Codes/Data/stock_data/source/hs300_1d_nor --region CN --interval 1d
        python yahoo/collector.py normalize_data --source_dir D:/ProgramCoding/Data/stock_data/source/hs300_1d_original --normalize_dir D:/ProgramCoding/Data/stock_data/source/hs300_1d_nor --region CN --interval 1d
        ```
  3. dump data: `python scripts/dump_bin.py dump_all`
    
     This will convert the normalized csv in `feature` directory as numpy array and store the normalized data one file per column and one symbol per directory. 
    
     - parameters:
       - `csv_path`: stock data path or directory, **normalize result(normalize_dir)**
       - `qlib_dir`: qlib(dump) data directory
       - `freq`: transaction frequency, by default `day`
         > `freq_map = {1d:day, 5min: 5min}`
       - `max_workers`: number of threads, by default *16*
       - `include_fields`: dump fields, by default `""`
       - `exclude_fields`: fields not dumped, by default `"""
         > dump_fields = `include_fields if include_fields else set(symbol_df.columns) - set(exclude_fields) exclude_fields else symbol_df.columns`
       - `symbol_field_name`: column *name* identifying symbol in csv files, by default `symbol`
       - `date_field_name`: column *name* identifying time in csv files, by default `date`
     - examples:
       ```bash
       # dump 1d cn
       python dump_bin.py dump_all --csv_path ~/.qlib/stock_data/source/hs300_1d_nor --qlib_dir ~/.qlib/qlib_data/hs300_1d_bin --freq 1d --exclude_fields date,symbol
       ```
       ```
       python dump_bin.py dump_all --csv_path D:/Codes/Data/stock_data/source/hs300_1d_nor --qlib_dir D:/Codes/Data/stock_data/qlib_data/hs300_1d_bin --freq day --exclude_fields date,symbol
       python dump_bin.py dump_all --csv_path D:/ProgramCoding/Data/stock_data/source/hs300_1d_nor --qlib_dir D:/ProgramCoding/Data/stock_data/qlib_data/hs300_1d_bin --freq day --exclude_fields date,symbol,adjustflag
       ```