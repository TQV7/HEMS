from pathlib import Path
import datetime as dt
import runpy

import pandas as pd

from config import CONFIG, DATA_DIR


def build_merged_df(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load and merge household load, agile tariff, and solar data into a single
    half-hourly dataframe indexed by UTC timestamp.

    Output columns:
        - load_kwh
        - import_price_p_per_kwh
        - export_price_p_per_kwh
        - solar_kwh
    """
    if data_dir is None:
        data_dir = DATA_DIR


    load_raw = pd.read_csv(data_dir / CONFIG.data.load_filename)
    load_raw[" Start"] = pd.to_datetime(load_raw[" Start"], utc=True)

    load = (
        load_raw.rename(
            columns={
                " Start": CONFIG.data.timestamp_column,
                "Consumption (kWh)": CONFIG.data.load_column,
            }
        )
        .set_index(CONFIG.data.timestamp_column)
        .sort_index()[[CONFIG.data.load_column]]
    )

    prices_raw = pd.read_csv(data_dir / CONFIG.data.price_filename)
    prices_raw["Period from"] = pd.to_datetime(
        prices_raw["Period from"], dayfirst=True
    ).dt.tz_localize("UTC")

    prices = (
        prices_raw.rename(
            columns={
                "Period from": CONFIG.data.timestamp_column,
                "Agile Import price (p/kWh)": CONFIG.tariff.import_price_column,
                "Agile Export price (p/kWh)": CONFIG.tariff.export_price_column,
            }
        )
        .set_index(CONFIG.data.timestamp_column)
        .sort_index()[
            [CONFIG.tariff.import_price_column, CONFIG.tariff.export_price_column]
        ]
    )

    solar_namespace = runpy.run_path(str(data_dir / CONFIG.data.solar_filename))
    actual_solar = solar_namespace["actualSolar"]

    rows = []

    for month in range(1, 13):
        for day in range(1, len(actual_solar[month])):
            for hour in range(24):
                ts = dt.datetime(2024, month, day, hour, tzinfo=dt.timezone.utc)
                hourly_kwh = actual_solar[month][day][hour]
                rows.append((ts, hourly_kwh))

    solar_hourly = (
        pd.DataFrame(rows, columns=[CONFIG.data.timestamp_column, "solar_hourly_kwh"])
        .set_index(CONFIG.data.timestamp_column)
        .sort_index()
    )

    solar_halfhour = solar_hourly.resample("30min").ffill()

    if CONFIG.data.split_hourly_solar_into_half_hours:
        solar_halfhour[CONFIG.data.solar_column] = (
            solar_halfhour["solar_hourly_kwh"] / 2.0
        )
    else:
        solar_halfhour[CONFIG.data.solar_column] = solar_halfhour["solar_hourly_kwh"]

    solar_halfhour = solar_halfhour[[CONFIG.data.solar_column]]

    merged = (
        load.join(prices, how="inner")
        .join(solar_halfhour, how="left")
        .sort_index()
    )

    merged.index.name = CONFIG.data.timestamp_column

    return merged