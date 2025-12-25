import pandas as pd

from bot import config
from bot import data_client


def _make_df(open_times, interval_ms):
    records = []
    for open_ts in open_times:
        records.append(
            {
                "open_ts": open_ts,
                "close_ts": open_ts + interval_ms,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "trades": 1,
            }
        )
    df = pd.DataFrame(records).set_index("close_ts")
    return df


def test_btc_1d_absolute_earliest_skips_download(monkeypatch, caplog):
    symbol = "BTC"
    interval = "1d"
    caplog.set_level("INFO")
    interval_ms = data_client.interval_to_ms(interval)
    earliest_fact = config.HYPERLIQUID_EARLIEST_KLINES_TS_MS[symbol]
    requested_start = earliest_fact - interval_ms
    requested_end = earliest_fact + interval_ms

    df = _make_df([earliest_fact], interval_ms)

    monkeypatch.setattr(data_client, "get_cache_earliest_ts_ms", lambda s, i: earliest_fact - interval_ms)
    monkeypatch.setattr(data_client, "load_klines_df_from_cache", lambda s, i: df)

    download_calls = []

    def _download(*args, **kwargs):
        download_calls.append((args, kwargs))

    monkeypatch.setattr(data_client, "download_history_to_cache", _download)

    result = data_client.ensure_market_data(symbol, interval, requested_start, requested_end)

    assert download_calls == []
    assert not result.empty
    warning_events = [record.msg for record in caplog.records if record.levelname == "WARNING"]
    info_events = [record.msg for record in caplog.records if record.levelname == "INFO"]
    assert any(msg.get("event") == "REQUEST_BEFORE_ABSOLUTE_EARLIEST" for msg in warning_events)
    assert any(msg.get("action") == "SKIP_DOWNLOAD_ABSOLUTE_EARLIEST_REACHED" for msg in info_events)


def test_btc_1d_absolute_earliest_clamps_and_downloads(monkeypatch, caplog):
    symbol = "BTC"
    interval = "1d"
    caplog.set_level("INFO")
    interval_ms = data_client.interval_to_ms(interval)
    earliest_fact = config.HYPERLIQUID_EARLIEST_KLINES_TS_MS[symbol]
    requested_start = earliest_fact - interval_ms
    requested_end = earliest_fact + interval_ms

    empty_df = pd.DataFrame(columns=["open_ts","close_ts","open","high","low","close","volume","trades"]).set_index("close_ts")
    df = _make_df([earliest_fact], interval_ms)
    load_calls = {"count": 0}

    def _load(*args, **kwargs):
        load_calls["count"] += 1
        return empty_df if load_calls["count"] == 1 else df

    monkeypatch.setattr(data_client, "get_cache_earliest_ts_ms", lambda s, i: None)
    monkeypatch.setattr(data_client, "load_klines_df_from_cache", _load)

    download_calls = []

    def _download(*args, **kwargs):
        download_calls.append((args, kwargs))

    monkeypatch.setattr(data_client, "download_history_to_cache", _download)

    data_client.ensure_market_data(symbol, interval, requested_start, requested_end)

    assert download_calls
    assert download_calls[0][0][2] == earliest_fact
    warning_events = [record.msg for record in caplog.records if record.levelname == "WARNING"]
    info_events = [record.msg for record in caplog.records if record.levelname == "INFO"]
    assert any(msg.get("event") == "REQUEST_BEFORE_ABSOLUTE_EARLIEST" for msg in warning_events)
    assert not any(msg.get("action") == "SKIP_DOWNLOAD_ABSOLUTE_EARLIEST_REACHED" for msg in info_events)


def test_4h_api_window_skip_download(monkeypatch, caplog):
    symbol = "BTC"
    interval = "4h"
    caplog.set_level("INFO")
    interval_ms = data_client.interval_to_ms(interval)
    requested_end = 2_000_000_000_000
    api_window_start = data_client.compute_api_window_start_ts_ms(interval, requested_end)
    requested_start = api_window_start - interval_ms

    df = _make_df([api_window_start], interval_ms)

    monkeypatch.setattr(data_client, "get_cache_earliest_ts_ms", lambda s, i: api_window_start)
    monkeypatch.setattr(data_client, "load_klines_df_from_cache", lambda s, i: df)

    download_calls = []

    def _download(*args, **kwargs):
        download_calls.append((args, kwargs))

    monkeypatch.setattr(data_client, "download_history_to_cache", _download)

    data_client.ensure_market_data(symbol, interval, requested_start, requested_end)

    assert download_calls == []
    warning_events = [record.msg for record in caplog.records if record.levelname == "WARNING"]
    info_events = [record.msg for record in caplog.records if record.levelname == "INFO"]
    assert any(msg.get("event") == "REQUEST_BEFORE_API_WINDOW" for msg in warning_events)
    assert any(msg.get("action") == "SKIP_DOWNLOAD_API_WINDOW_REACHED" for msg in info_events)


def test_4h_api_window_clamps_and_downloads(monkeypatch, caplog):
    symbol = "BTC"
    interval = "4h"
    caplog.set_level("INFO")
    interval_ms = data_client.interval_to_ms(interval)
    requested_end = 2_000_000_000_000
    api_window_start = data_client.compute_api_window_start_ts_ms(interval, requested_end)
    requested_start = api_window_start - interval_ms

    empty_df = pd.DataFrame(columns=["open_ts","close_ts","open","high","low","close","volume","trades"]).set_index("close_ts")
    df = _make_df([api_window_start], interval_ms)
    load_calls = {"count": 0}

    def _load(*args, **kwargs):
        load_calls["count"] += 1
        return empty_df if load_calls["count"] == 1 else df

    monkeypatch.setattr(data_client, "get_cache_earliest_ts_ms", lambda s, i: None)
    monkeypatch.setattr(data_client, "load_klines_df_from_cache", _load)

    download_calls = []

    def _download(*args, **kwargs):
        download_calls.append((args, kwargs))

    monkeypatch.setattr(data_client, "download_history_to_cache", _download)

    data_client.ensure_market_data(symbol, interval, requested_start, requested_end)

    assert download_calls
    assert download_calls[0][0][2] == api_window_start
    warning_events = [record.msg for record in caplog.records if record.levelname == "WARNING"]
    info_events = [record.msg for record in caplog.records if record.levelname == "INFO"]
    assert any(msg.get("event") == "REQUEST_BEFORE_API_WINDOW" for msg in warning_events)
    assert not any(msg.get("action") == "SKIP_DOWNLOAD_API_WINDOW_REACHED" for msg in info_events)


def test_normal_request_no_boundary_warning(monkeypatch, caplog):
    symbol = "BTC"
    interval = "4h"
    caplog.set_level("INFO")
    interval_ms = data_client.interval_to_ms(interval)
    requested_end = 2_000_000_000_000
    api_window_start = data_client.compute_api_window_start_ts_ms(interval, requested_end)
    requested_start = api_window_start + interval_ms

    empty_df = pd.DataFrame(columns=["open_ts","close_ts","open","high","low","close","volume","trades"]).set_index("close_ts")
    df = _make_df([requested_start], interval_ms)
    load_calls = {"count": 0}

    def _load(*args, **kwargs):
        load_calls["count"] += 1
        return empty_df if load_calls["count"] == 1 else df

    monkeypatch.setattr(data_client, "get_cache_earliest_ts_ms", lambda s, i: None)
    monkeypatch.setattr(data_client, "load_klines_df_from_cache", _load)

    download_calls = []

    def _download(*args, **kwargs):
        download_calls.append((args, kwargs))

    monkeypatch.setattr(data_client, "download_history_to_cache", _download)

    data_client.ensure_market_data(symbol, interval, requested_start, requested_end)

    assert download_calls
    assert download_calls[0][0][2] == requested_start
    warning_events = [record.msg for record in caplog.records if record.levelname == "WARNING"]
    assert warning_events == []
