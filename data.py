"""
获取数据
"""

import os
import pandas as pd
from pathlib import Path
from common import PATH, ADJUST_NONE, ADJUST_PREV, ADJUST_POST


class Data:
    def __init__(self, path: Path = PATH):
        self.path = path

    def basic(self):
        basic = pd.read_csv(self.path / "basic.csv")
        basic["list_date"] = pd.to_datetime(basic["list_date"], format="%Y%m%d").dt.date
        return basic

    def trade_cal(self):
        cal = pd.read_csv(self.path / "trade_cal.csv")
        cal = cal[cal["is_open"] == 1]
        cal["cal_date"] = pd.to_datetime(cal["cal_date"], format="%Y%m%d").dt.date
        cal["pretrade_date"] = pd.to_datetime(cal["pretrade_date"]).dt.date
        cal = cal.sort_values("cal_date", kind="stable")
        return cal.reset_index(drop=True)

    def daily(
        self,
        start_date: str = "20250101",
        end_date: str = "20260101",
        bj: bool = False,
        st: bool = False,
        adjust: int = ADJUST_NONE,
    ):
        daily_path = self.path / "daily"
        files = [
            entry.path
            for entry in os.scandir(daily_path)
            if entry.is_file()
            and entry.name.endswith(".csv")
            and start_date <= entry.name[:8] <= end_date
        ]
        if not files:
            return pd.DataFrame(
                columns=[
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "change",
                    "pct_chg",
                    "vol",
                    "amount",
                    "vwap",
                ]
            )

        files.sort()
        stock_st_path = self.path / "stock_st"
        frames = []
        for file in files:
            frame = pd.read_csv(file)
            if not st:
                st_file = stock_st_path / Path(file).name
                if st_file.is_file():
                    st_codes = pd.read_csv(st_file, usecols=["ts_code"], dtype=str)[
                        "ts_code"
                    ]
                    if not st_codes.empty:
                        frame = frame[~frame["ts_code"].isin(st_codes)]
            frames.append(frame)

        daily = pd.concat(frames, ignore_index=True)
        if not bj:
            daily = daily[~daily["ts_code"].str.endswith("BJ")]
        if daily.empty:
            daily["trade_date"] = pd.to_datetime(
                daily["trade_date"], format="%Y%m%d"
            ).dt.date
            return daily

        if adjust != ADJUST_NONE:
            if adjust not in {ADJUST_PREV, ADJUST_POST}:
                raise ValueError(f"unsupported adjust mode: {adjust}")

            adjusted = daily.sort_values(
                ["ts_code", "trade_date"], kind="stable"
            ).copy()
            ratio = adjusted["close"].div(adjusted["pre_close"])
            cum_ratio = ratio.groupby(adjusted["ts_code"], sort=False).cumprod()
            first_ratio = cum_ratio.groupby(adjusted["ts_code"], sort=False).transform(
                "first"
            )
            first_close = adjusted.groupby("ts_code", sort=False)["close"].transform(
                "first"
            )
            post_close = first_close.mul(cum_ratio.div(first_ratio))
            scale = post_close.div(adjusted["close"])

            if adjust == ADJUST_PREV:
                last_close = adjusted.groupby("ts_code", sort=False)["close"].transform(
                    "last"
                )
                last_post_close = post_close.groupby(
                    adjusted["ts_code"], sort=False
                ).transform("last")
                scale = scale.mul(last_close.div(last_post_close))

            price_cols = ["open", "high", "low", "close", "pre_close", "vwap"]
            adjusted.loc[:, price_cols] = adjusted[price_cols].mul(scale, axis=0)
            if "change" in adjusted.columns:
                adjusted.loc[:, "change"] = adjusted["close"] - adjusted["pre_close"]
            daily.loc[adjusted.index, price_cols] = adjusted[price_cols]
            if "change" in adjusted.columns:
                daily.loc[adjusted.index, "change"] = adjusted["change"]

        daily["trade_date"] = pd.to_datetime(
            daily["trade_date"], format="%Y%m%d"
        ).dt.date
        return daily

    def market(self, name: str = "000001.SH"):
        market = pd.read_csv(self.path / "market" / (name + ".csv"))
        market["trade_date"] = pd.to_datetime(
            market["trade_date"], format="%Y%m%d"
        ).dt.date
        market = market.sort_values("trade_date", kind="stable")
        return market.reset_index(drop=True)


def main():
    path = Path("./data")
    reader = Data(path)
    print(reader.basic().head())
    print(reader.trade_cal().head())
    print(reader.daily(start_date="20260501", end_date="20260508", adjust=ADJUST_PREV))
    print(reader.market().head())


if __name__ == "__main__":
    main()
