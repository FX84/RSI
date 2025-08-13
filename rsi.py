#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rsi.py — Расчёт и визуализация RSI (Wilder) с генерацией сигналов.
Автор: FX84
Лицензия: MIT
Дисклеймер: Скрипт предназначен для образовательных целей и не является инвестсоветом.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Попробуем лениво импортировать yfinance (нужно по умолчанию)
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # пользователь может работать из CSV

# Для свечного графика используем mplfinance (опционально по флагу --candles)
_mpfin_ok = True
try:
    import mplfinance as mpf
except Exception:
    _mpfin_ok = False


# ----------------------------- Конфигурации/датаклассы -----------------------------

@dataclass
class Args:
    source: str
    csv: Optional[str]
    ticker: Optional[str]
    start: Optional[str]
    end: Optional[str]
    interval: str
    period: int
    overbought: float
    oversold: float
    centerline: bool
    min_gap: int
    ffill: bool
    out: Optional[str]
    jsout: Optional[str]
    tail: Optional[int]
    signals_out: Optional[str]
    show: bool
    save: Optional[str]
    dpi: int
    figsize: Tuple[float, float]
    candles: bool
    tz: Optional[str]
    log_level: str
    seed: Optional[int]


# ----------------------------- Вспомогательные функции -----------------------------

def setup_logging(level: str = "INFO") -> None:
    """Настройка логирования."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> Args:
    """Парсинг аргументов командной строки."""
    p = argparse.ArgumentParser(
        prog="rsi.py",
        description="Расчёт RSI (Wilder), сигналы и графики. Источник: yfinance или CSV.",
    )
    p.add_argument("--source", choices=["yfinance", "csv"], default="yfinance",
                   help="Источник данных (по умолчанию yfinance).")
    p.add_argument("--csv", type=str, default=None, help="Путь к CSV, если --source csv.")
    p.add_argument("--ticker", type=str, default=None, help="Тикер для yfinance (например, AAPL, EURUSD=X).")
    p.add_argument("--start", type=str, default=None, help="Дата начала (YYYY-MM-DD).")
    p.add_argument("--end", type=str, default=None, help="Дата конца (YYYY-MM-DD).")
    p.add_argument("--interval", type=str, default="1d",
                   help="Интервал свечей (1m,5m,15m,1h,1d и т.д., если поддерживается источником).")
    p.add_argument("--period", type=int, default=14, help="Период RSI (по умолчанию 14).")
    p.add_argument("--overbought", type=float, default=70.0, help="Уровень перекупленности (по умолчанию 70).")
    p.add_argument("--oversold", type=float, default=30.0, help="Уровень перепроданности (по умолчанию 30).")
    p.add_argument("--centerline", action="store_true", help="Показывать/учитывать пересечения уровня 50.")
    p.add_argument("--min-gap", type=int, default=0,
                   help="Мин. расстояние в барах между одинаковыми сигналами (шумоподавление).")
    p.add_argument("--ffill", action="store_true",
                   help="Forward-fill пропусков (актуально для внутридневных данных).")
    p.add_argument("--out", type=str, default=None,
                   help="Сохранить таблицу (Datetime, Close, RSI, ...) в CSV.")
    p.add_argument("--json", dest="jsout", type=str, default=None,
                   help="Сохранить последние N точек в JSON (исп. с --tail).")
    p.add_argument("--tail", type=int, default=None, help="Сколько последних точек экспортировать в JSON.")
    p.add_argument("--signals-out", type=str, default=None, help="Сохранить таблицу сигналов в CSV.")
    p.add_argument("--show", action="store_true", help="Показать окно с графиком.")
    p.add_argument("--save", type=str, default=None, help="Сохранить график в файл (PNG/SVG и т.п.).")
    p.add_argument("--dpi", type=int, default=120, help="DPI для сохранения рисунка (по умолчанию 120).")
    p.add_argument("--figsize", nargs=2, type=float, default=[12.0, 7.0],
                   help="Размер фигуры, например: --figsize 12 7")
    p.add_argument("--candles", action="store_true", help="Рисовать верхний график свечами (требует mplfinance).")
    p.add_argument("--tz", type=str, default=None,
                   help="Таймзона для локализации/конвертации дат, например Europe/Madrid. По умолчанию оставляет как есть.")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                   help="Уровень логов (по умолчанию INFO).")
    p.add_argument("--seed", type=int, default=None, help="Фиксировать seed (если нужно).")

    a = p.parse_args()
    return Args(
        source=a.source,
        csv=a.csv,
        ticker=a.ticker,
        start=a.start,
        end=a.end,
        interval=a.interval,
        period=a.period,
        overbought=a.overbought,
        oversold=a.oversold,
        centerline=a.centerline,
        min_gap=a.min_gap,
        ffill=a.ffill,
        out=a.out,
        jsout=a.jsout,
        tail=a.tail,
        signals_out=a.signals_out,
        show=a.show,
        save=a.save,
        dpi=a.dpi,
        figsize=(float(a.figsize[0]), float(a.figsize[1])),
        candles=a.candles,
        tz=a.tz,
        log_level=a.log_level,
        seed=a.seed,
    )


# ----------------------------- Загрузка и подготовка данных -----------------------------

def load_data_from_yf(ticker: str, start: Optional[str], end: Optional[str], interval: str) -> pd.DataFrame:
    """
    Загрузить данные по тикеру через yfinance.
    Ожидаемые колонки: Open, High, Low, Close, Adj Close, Volume
    Индекс: Datetime (tz-aware или naive в зависимости от провайдера).
    """
    if yf is None:
        raise RuntimeError("Пакет yfinance не установлен. Установите: pip install yfinance")

    if not ticker:
        raise ValueError("Параметр --ticker обязателен при --source yfinance")

    logging.info("Загрузка данных через yfinance: ticker=%s, start=%s, end=%s, interval=%s",
                 ticker, start, end, interval)
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Ошибка запроса к yfinance: {e}")

    if df is None or df.empty:
        raise RuntimeError("Не удалось загрузить данные: пустой ответ от yfinance.")

    # yfinance может вернуть колонку 'Adj Close' с пробелом — нормализуем имена
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # Приведём имя 'Adj_Close' для единообразия
    if "Adj_Close" not in df.columns and "Adj_Close" in [c.replace(" ", "_") for c in df.columns]:
        pass  # уже приведено
    return df


def load_data_from_csv(path: str) -> pd.DataFrame:
    """
    Загрузить данные из CSV. Минимум требуется колонка Close.
    Предпочтительно наличие Datetime индекса или колонки Datetime.
    """
    if not path:
        raise ValueError("Параметр --csv обязателен при --source csv")

    logging.info("Загрузка данных из CSV: %s", path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать CSV: {e}")

    # Определим колонку даты/времени
    dt_col = None
    for c in ["Datetime", "Date", "date", "datetime", "Time", "time"]:
        if c in df.columns:
            dt_col = c
            break

    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col], utc=False, errors="coerce")
        df = df.dropna(subset=[dt_col])
        df = df.set_index(dt_col)
    else:
        # Если даты нет, создадим простой RangeIndex -> дальше проверим
        logging.warning("CSV не содержит явной колонки даты/времени — индексом будет порядковый номер.")

    # Нормализуем имена колонок
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    if "Close" not in df.columns:
        raise ValueError("CSV не содержит колонку 'Close'.")

    return df


def preprocess_df(df: pd.DataFrame, tz: Optional[str], ffill: bool) -> pd.DataFrame:
    """
    Базовая очистка данных:
    - Индекс к Datetime (если возможно)
    - Сортировка, удаление дубликатов
    - Обработка пропусков
    - Приведение таймзоны (если указана)
    """
    # Индекс в Datetime, если содержит даты, иначе оставим как есть
    if not isinstance(df.index, pd.DatetimeIndex):
        # Попробуем конвертировать индекс в даты
        try:
            di = pd.to_datetime(df.index, utc=False, errors="coerce")
            if di.notna().all():
                df.index = di
        except Exception:
            pass

    # Если индекс — даты, приведём к монотонности
    if isinstance(df.index, pd.DatetimeIndex):
        # Локализация/конвертация таймзоны
        if tz:
            # Если naive — локализуем; если aware — конвертируем
            if df.index.tz is None:
                try:
                    df.index = df.index.tz_localize(tz)
                except Exception:
                    # fallback: локализуем к UTC, потом конвертируем
                    df.index = df.index.tz_localize("UTC").tz_convert(tz)
            else:
                df.index = df.index.tz_convert(tz)

        # Удаляем дубликаты и сортируем
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

    # Обработка пропусков
    if ffill:
        df = df.ffill()
        logging.warning("Применён forward-fill для пропусков (--ffill).")
    else:
        # Для дневных данных чаще корректно дропнуть пустые строки
        df = df.dropna(subset=["Close"])

    return df


# ----------------------------- RSI (Wilder) и сигналы -----------------------------

def compute_rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """
    Рассчитать RSI по методу Уайлдера.
    Возвращает pd.Series длины close с NaN до инициализации.
    """
    if period < 1:
        raise ValueError("Период RSI должен быть >= 1.")
    if close is None or close.size == 0:
        raise ValueError("Серия цен пуста.")
    if close.size < period + 1:
        raise ValueError(f"Недостаточно данных для периода RSI={period}: требуется минимум {period + 1} баров.")

    # Возьмём разности соседних цен
    delta = close.diff()

    # Положительные и отрицательные изменения
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Инициализация средних значений Уайлдера на первых 'period' точках (со 2-й по period+1-ю дифференцию)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Создадим копии для рекуррентного сглаживания
    avg_gain = avg_gain.copy()
    avg_loss = avg_loss.copy()

    # Рекуррентно вычисляем значения после инициализации
    for i in range(period + 1, len(close)):
        # Wilder smoothing:
        avg_gain.iat[i] = (avg_gain.iat[i - 1] * (period - 1) + gain.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i - 1] * (period - 1) + loss.iat[i]) / period

    # RS и RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Граничные случаи: avg_loss == 0 -> RSI=100; avg_gain == 0 -> RSI=0
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)

    rsi.name = "RSI"
    return rsi


def _cross_up(series: pd.Series, level: float) -> pd.Series:
    """Фиксируем моменты пересечения уровня снизу вверх."""
    prev = series.shift(1)
    return (prev < level) & (series >= level)


def _cross_down(series: pd.Series, level: float) -> pd.Series:
    """Фиксируем моменты пересечения уровня сверху вниз."""
    prev = series.shift(1)
    return (prev > level) & (series <= level)


def detect_signals(rsi: pd.Series,
                   close: pd.Series,
                   overbought: float,
                   oversold: float,
                   centerline: bool,
                   min_gap: int = 0) -> pd.DataFrame:
    """
    По серии RSI генерируем сигналы:
      - cross_up_oversold: пересечение уровня oversold снизу
      - cross_down_overbought: пересечение уровня overbought сверху
      - centerline_up / centerline_down: пересечение уровня 50
    min_gap — минимальное расстояние (в барах) между одинаковыми сигналами.
    """
    if rsi.isna().all():
        return pd.DataFrame(columns=["Datetime", "Type", "RSI", "Close"])

    sigs = []
    idx = rsi.index

    # Бинарные маски пересечений
    m_up_os = _cross_up(rsi, oversold)
    m_dn_ob = _cross_down(rsi, overbought)
    m_up_50 = _cross_up(rsi, 50.0) if centerline else pd.Series(False, index=idx)
    m_dn_50 = _cross_down(rsi, 50.0) if centerline else pd.Series(False, index=idx)

    def append_events(mask: pd.Series, typ: str):
        last_i = None
        for i, flag in enumerate(mask.values):
            if not flag:
                continue
            if min_gap and last_i is not None and (i - last_i) < min_gap:
                continue
            ts = idx[i]
            sigs.append({
                "Datetime": ts,
                "Type": typ,
                "RSI": float(rsi.iat[i]),
                "Close": float(close.iat[i]) if not np.isnan(close.iat[i]) else np.nan
            })
            last_i = i

    append_events(m_up_os, "cross_up_oversold")
    append_events(m_dn_ob, "cross_down_overbought")
    if centerline:
        append_events(m_up_50, "centerline_up")
        append_events(m_dn_50, "centerline_down")

    sdf = pd.DataFrame(sigs)
    if not sdf.empty:
        # Отсортировать по времени на случай перестановок
        sdf = sdf.sort_values(by="Datetime").reset_index(drop=True)
    return sdf


# ----------------------------- Визуализация -----------------------------

def plot_price_and_rsi(df: pd.DataFrame,
                       rsi: pd.Series,
                       args: Args) -> plt.Figure:
    """
    Рисуем две панели: цена и RSI. Для цены по флагу --candles — свечи (mplfinance).
    """
    # Убедимся, что у нас есть базовые колонки
    have_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])

    if args.candles:
        if not _mpfin_ok:
            raise RuntimeError("Для --candles требуется пакет mplfinance. Установите: pip install mplfinance")
        if not have_ohlc:
            raise ValueError("Для свечного графика необходимы колонки Open, High, Low, Close.")

    # Создаём figure/axes
    fig = plt.figure(figsize=args.figsize, dpi=args.dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.05)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_rsi = fig.add_subplot(gs[1, 0], sharex=ax_price)

    # Верхняя панель: цена
    if args.candles:
        # mplfinance удобно работает со своим API: подготовим DataFrame в нужном формате
        mpf_df = df.copy()
        # Убедимся в названии колонок
        mpf_df = mpf_df[["Open", "High", "Low", "Close"]].copy()
        mpf.plot(mpf_df, type="candle", ax=ax_price, xrotation=0, style="yahoo", warn_too_much_data=1000000)
        ax_price.set_ylabel("Price")
    else:
        ax_price.plot(df.index, df["Close"], label="Close")
        ax_price.set_ylabel("Price")
        ax_price.legend(loc="upper left")

    ax_price.grid(True, alpha=0.3)

    # Нижняя панель: RSI
    ax_rsi.plot(rsi.index, rsi.values, label="RSI")
    ax_rsi.axhline(args.overbought, linestyle="--", linewidth=1.0)
    ax_rsi.axhline(args.oversold, linestyle="--", linewidth=1.0)
    if args.centerline:
        ax_rsi.axhline(50.0, linestyle=":", linewidth=1.0)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True, alpha=0.3)
    ax_rsi.legend(loc="upper left")

    # Красивее подписи дат
    fig.autofmt_xdate()
    return fig


# ----------------------------- Экспорт -----------------------------

def export_timeseries(df: pd.DataFrame, rsi: pd.Series, args: Args) -> Optional[str]:
    """
    Сохранить объединённую таблицу (Datetime, Close, RSI, [Open,High,Low,Volume...]) в CSV.
    Возвращает путь к файлу, если сохранено.
    """
    if not args.out:
        return None
    out_df = df.copy()
    out_df["RSI"] = rsi
    try:
        out_df.to_csv(args.out, index=True)
        logging.info("Сохранён таймсериз в CSV: %s", args.out)
        return args.out
    except Exception as e:
        logging.error("Не удалось сохранить CSV: %s", e)
        return None


def export_json_tail(df: pd.DataFrame, rsi: pd.Series, args: Args) -> Optional[str]:
    """
    Экспорт последних N точек в JSON (если указаны --json и --tail).
    Формат: список объектов с Datetime, Close, RSI.
    """
    if not args.jsout or not args.tail:
        return None

    last = pd.DataFrame({
        "Datetime": df.index[-args.tail:],
        "Close": df["Close"].iloc[-args.tail:].values,
        "RSI": rsi.iloc[-args.tail:].values
    })
    # Конвертация индексов/дат в строку ISO
    last["Datetime"] = pd.to_datetime(last["Datetime"]).astype("datetime64[ns]").dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT")
    data = last.to_dict(orient="records")
    try:
        with open(args.jsout, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logging.info("Сохранён JSON с последними %d точками: %s", args.tail, args.jsout)
        return args.jsout
    except Exception as e:
        logging.error("Не удалось сохранить JSON: %s", e)
        return None


def export_signals(signals: pd.DataFrame, path: Optional[str]) -> Optional[str]:
    """
    Сохранить таблицу сигналов в CSV (если указан путь).
    """
    if not path:
        return None
    try:
        # Преобразуем Datetime в строку для совместимости
        sdf = signals.copy()
        if "Datetime" in sdf.columns:
            sdf["Datetime"] = pd.to_datetime(sdf["Datetime"]).astype("datetime64[ns]").astype(str)
        sdf.to_csv(path, index=False)
        logging.info("Сохранены сигналы в CSV: %s", path)
        return path
    except Exception as e:
        logging.error("Не удалось сохранить сигналы: %s", e)
        return None


# ----------------------------- Основной сценарий -----------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Валидация режима источника
    if args.source == "yfinance" and not args.ticker:
        logging.error("При --source yfinance необходимо указать --ticker.")
        sys.exit(2)
    if args.source == "csv" and not args.csv:
        logging.error("При --source csv необходимо указать --csv PATH.")
        sys.exit(2)

    # Загрузка данных
    if args.source == "yfinance":
        try:
            df = load_data_from_yf(args.ticker, args.start, args.end, args.interval)
        except Exception as e:
            logging.error(str(e))
            sys.exit(1)
    else:
        try:
            df = load_data_from_csv(args.csv)
        except Exception as e:
            logging.error(str(e))
            sys.exit(1)

    # Предобработка
    try:
        df = preprocess_df(df, args.tz, args.ffill)
    except Exception as e:
        logging.error("Ошибка предобработки данных: %s", e)
        sys.exit(1)

    # Проверим наличие Close
    if "Close" not in df.columns:
        logging.error("Таблица не содержит колонку Close.")
        sys.exit(1)

    # Расчёт RSI
    try:
        rsi = compute_rsi_wilder(df["Close"].astype(float), args.period)
    except Exception as e:
        logging.error("Ошибка расчёта RSI: %s", e)
        sys.exit(1)

    # Сигналы
    try:
        signals = detect_signals(
            rsi=rsi,
            close=df["Close"],
            overbought=args.overbought,
            oversold=args.oversold,
            centerline=args.centerline,
            min_gap=args.min_gap,
        )
    except Exception as e:
        logging.error("Ошибка генерации сигналов: %s", e)
        sys.exit(1)

    # Краткая сводка в консоль
    try:
        start_dt = df.index.min()
        end_dt = df.index.max()
        last_close = df["Close"].iloc[-1]
        last_rsi = rsi.iloc[-1]
        logging.info("Данные: %s → %s (%d баров)", str(start_dt), str(end_dt), len(df))
        logging.info("Последняя цена: %.6f | Последний RSI(%d): %.2f", last_close, args.period, last_rsi)
        logging.info("Найдено сигналов: %d", len(signals))
    except Exception:
        pass

    # Экспорт таблиц
    export_timeseries(df, rsi, args)
    export_json_tail(df, rsi, args)
    if not signals.empty:
        export_signals(signals, args.signals_out)
    elif args.signals_out:
        # Если пользователь запросил файл, но сигналов нет — создадим пустой шаблон
        export_signals(pd.DataFrame(columns=["Datetime", "Type", "RSI", "Close"]), args.signals_out)

    # Визуализация
    try:
        fig = plot_price_and_rsi(df, rsi, args)
        if args.save:
            fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
            logging.info("График сохранён в файл: %s", args.save)
        if args.show:
            plt.show()
        else:
            plt.close(fig)
    except Exception as e:
        logging.error("Ошибка построения/сохранения графика: %s", e)
        # Не завершаем скрипт аварийно: данные и сигналы могли быть успешно экспортированы


if __name__ == "__main__":
    main()