import asyncio
import aiohttp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import time
import datetime
from pybit.unified_trading import HTTP
from bingx_client import BingxClient
# ================= CONFIG =================

BOT_TOKEN = 6666
CHAT_ID = ""

ZONE_TF = "15m"              # Таймфрейм зон (настраиваемый)
ZONE_LIMIT = 1000             # Количество свечей
ATR_PERIOD = 20               # Период ATR
MIN_MOVE_MULT = 2        # Мин. движение после зоны (в ATR)
LOOKAHEAD = 10                # Сколько свечей смотреть вперёд для подтверждения движения
MAX_ZONES = 10                # Макс. зон на тип (supply/demand)
ZONE_TOLERANCE = 0.003         # Толерантность для входа в зону
INVALIDATION_METHOD = "close" # "close" или "wick" для инвалидации
SCAN_INTERVAL_SEC = 300       # Интервал сканирования в секундах
MAX_CONCURRENT = 10
CHART_CANDLES = 200           # Кол-во свечей на графике

MIN_ZONE_SCORE = 4     # ↓ уменьшаешь — больше зон
SWING_LEFT = 2
SWING_RIGHT = 2
IMPULSE_MIN_ATR = 2    # чувствительность импульса

MAX_ZONE_ATR = 4
# ==========================================

client = HTTP(testnet=False)
sent_signals = {}             # анти-дублирование
used_zones = {}               # для отслеживания использованных зон (одноразовые)

# ================= EXCHANGE =================

API_KEY = ''
API_SECRET = ''
bx = BingxClient(API_KEY, API_SECRET)

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================

def is_zone_broken(df, zone):
    """Проверяет, пробита ли зона после её формирования."""
    start_idx = zone['start_bar'] + 1
    if start_idx >= len(df):
        return False
    slice_df = df.iloc[start_idx:]
    if zone['type'] == 'supply':
        if INVALIDATION_METHOD == "close":
            break_level = slice_df['close'].max()
        else:  # wick
            break_level = slice_df['high'].max()
        return break_level > zone['high']
    else:  # demand
        if INVALIDATION_METHOD == "close":
            break_level = slice_df['close'].min()
        else:
            break_level = slice_df['low'].min()
        return break_level < zone['low']

# ================= ФОРМАЦИИ =================

def is_bullish_engulfing(df, i):
    if i < 1:
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    if prev['close'] >= prev['open']:          # предыдущая должна быть медвежьей
        return False
    if curr['close'] <= curr['open']:          # текущая должна быть бычьей
        return False
    if curr['open'] < prev['close'] and curr['close'] > prev['open']:
        return True
    return False

def is_bearish_engulfing(df, i):
    if i < 1:
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    if prev['close'] <= prev['open']:          # предыдущая должна быть бычьей
        return False
    if curr['close'] >= curr['open']:          # текущая должна быть медвежьей
        return False
    if curr['open'] > prev['close'] and curr['close'] < prev['open']:
        return True
    return False

def is_bullish_pinbar_df(df, i):
    candle = df.iloc[i]
    o = candle['open']
    h = candle['high']
    l = candle['low']
    c = candle['close']
    body = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return False
    if body > 0.3 * total_range:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if c <= o:          # должна быть бычьей
        return False
    if body == 0:
        return False
    if lower_wick >= body * 2 and upper_wick <= body * 0.5:
        return True
    return False

def is_bearish_pinbar_df(df, i):
    candle = df.iloc[i]
    o = candle['open']
    h = candle['high']
    l = candle['low']
    c = candle['close']
    body = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return False
    if body > 0.3 * total_range:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if c >= o:          # должна быть медвежьей
        return False
    if body == 0:
        return False
    if upper_wick >= body * 2 and lower_wick <= body * 0.5:
        return True
    return False

# ================= ПРОВЕРКА ПРЕДШЕСТВУЮЩЕГО ДВИЖЕНИЯ =================

def check_preceding_movement(df, start_bar, zone, atr, zone_type):
    """
    Проверяет, что перед зоной (start_bar) был импульс и откат.
    - zone_type: 'supply' или 'demand'
    - start_bar: индекс первой свечи зоны
    - zone: словарь с ключами 'high', 'low'
    - atr: значение ATR на момент формирования зоны
    """
    lookback = 10          # максимальный поиск назад
    max_gap = 5            # максимальное расстояние между экстремумом и start_bar

    if start_bar < 1:
        return False

    search_start = max(0, start_bar - lookback)
    search_end = start_bar - 1
    if search_end < search_start:
        return False

    # Проверка высоты зоны
    if zone['high'] - zone['low'] < atr:
        return False

    if zone_type == 'supply':
        # Поиск максимума перед зоной
        df_before = df.iloc[search_start:search_end+1]
        if df_before.empty:
            return False
        idx_max = df_before['high'].idxmax()
        if start_bar - idx_max > max_gap:
            return False

        # Поиск минимума перед этим максимумом (в пределах max_gap свечей до idx_max)
        min_search_start = max(search_start, idx_max - max_gap)
        min_search_end = idx_max - 1
        if min_search_end < min_search_start:
            return False
        df_min = df.iloc[min_search_start:min_search_end+1]
        if df_min.empty:
            return False
        idx_min = df_min['low'].idxmin()

        # Проверка роста от минимума до максимума
        rise = df.loc[idx_max, 'high'] - df.loc[idx_min, 'low']
        if rise < 2 * atr:
            return False

        # Проверка падения от максимума до нижней границы зоны
        fall = df.loc[idx_max, 'high'] - zone['low']
        if fall < 1 * atr:
            return False

        return True

    elif zone_type == 'demand':
        # Поиск минимума перед зоной
        df_before = df.iloc[search_start:search_end+1]
        if df_before.empty:
            return False
        idx_min = df_before['low'].idxmin()
        if start_bar - idx_min > max_gap:
            return False

        # Поиск максимума перед этим минимумом
        max_search_start = max(search_start, idx_min - max_gap)
        max_search_end = idx_min - 1
        if max_search_end < max_search_start:
            return False
        df_max = df.iloc[max_search_start:max_search_end+1]
        if df_max.empty:
            return False
        idx_max = df_max['high'].idxmax()

        # Проверка падения от максимума до минимума
        fall = df.loc[idx_max, 'high'] - df.loc[idx_min, 'low']
        if fall < 2 * atr:
            return False

        # Проверка подъёма от минимума до верхней границы зоны
        rise = zone['high'] - df.loc[idx_min, 'low']
        if rise < 1 * atr:
            return False

        return True

    return False

# ================= ПОИСК ЗОН =================

def build_zone(df, i, zone_type):
    o = df['open'].iloc[i]
    c = df['close'].iloc[i]
    h = df['high'].iloc[i]
    l = df['low'].iloc[i]
    atr = df['atr'].iloc[i]

    max_size = atr * MAX_ZONE_ATR

    if zone_type == 'demand':
        low = l
        high = min(o, c)

        if (high - low) > max_size:
            high = low + max_size

        return {
            "low": low,
            "high": high,
            "start_bar": i,
            "type": "demand"
        }

    else:
        low = max(o, c)
        high = h

        if (high - low) > max_size:
            low = high - max_size

        return {
            "low": low,
            "high": high,
            "start_bar": i,
            "type": "supply"
        }
def get_nearest_zones(price, zones, n=2):
    if not zones:
        return []

    def zone_distance(z):
        if price < z["low"]:
            return z["low"] - price
        elif price > z["high"]:
            return price - z["high"]
        else:
            return 0

    zones = sorted(zones, key=zone_distance)
    return zones[:n]        
        
def calculate_zone_score(df, i, zone, atr_val, zone_type):
    score = 0

    # ---- Импульс вперёд ----
    lookahead = LOOKAHEAD
    if i + lookahead >= len(df):
        return 0

    if zone_type == "demand":
        future_high = df['high'].iloc[i+1:i+1+lookahead].max()
        move = future_high - zone['high']
    else:
        future_low = df['low'].iloc[i+1:i+1+lookahead].min()
        move = zone['low'] - future_low

    impulse_strength = move / atr_val
    score += min(3, impulse_strength)

    if impulse_strength < IMPULSE_MIN_ATR:
        return 0

    # ---- Тело свечи ----
    body = abs(df['close'].iloc[i] - df['open'].iloc[i])
    full_range = df['high'].iloc[i] - df['low'].iloc[i]
    if full_range > 0:
        body_ratio = body / full_range
        score += body_ratio * 2

    # ---- Swing подтверждение ----
    if zone_type == "demand" and is_swing_low(df, i):
        score += 2
    if zone_type == "supply" and is_swing_high(df, i):
        score += 2

    return score

def find_supply_demand_zones(df):


    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()

    supply_zones = []
    demand_zones = []

    min_idx = max(ATR_PERIOD + 5, SWING_LEFT)
    max_idx = len(df) - LOOKAHEAD - 1

    for i in range(min_idx, max_idx):

        if pd.isna(df['atr'].iloc[i]):
            continue

        atr_val = df['atr'].iloc[i]

        # ---- DEMAND ----
        if is_bullish_pinbar_df(df, i) or is_bullish_engulfing(df, i):
            zone = build_zone(df, i, "demand")
            score = calculate_zone_score(df, i, zone, atr_val, "demand")

            if score >= MIN_ZONE_SCORE:
                zone["score"] = round(score, 2)
                demand_zones.append(zone)

        # ---- SUPPLY ----
        if is_bearish_pinbar_df(df, i) or is_bearish_engulfing(df, i):
            zone = build_zone(df, i, "supply")
            score = calculate_zone_score(df, i, zone, atr_val, "supply")

            if score >= MIN_ZONE_SCORE:
                zone["score"] = round(score, 2)
                supply_zones.append(zone)

    # Удаляем пробитые
    supply_zones = [z for z in supply_zones if not is_zone_broken(df, z)]
    demand_zones = [z for z in demand_zones if not is_zone_broken(df, z)]

    # Удаляем дубликаты
    supply_zones = list({(z["start_bar"], z["high"], z["low"]): z for z in supply_zones}.values())
    demand_zones = list({(z["start_bar"], z["high"], z["low"]): z for z in demand_zones}.values())

    # Сортировка по score и свежести
    supply_zones = sorted(supply_zones, key=lambda z: (z["score"], z["start_bar"]), reverse=True)[:MAX_ZONES]
    demand_zones = sorted(demand_zones, key=lambda z: (z["score"], z["start_bar"]), reverse=True)[:MAX_ZONES]

    return supply_zones, demand_zones

# ================= ПИНБАР НА 5M =================

def is_bearish_pinbar(candle):
    o = float(candle['open'])
    h = float(candle['high'])
    l = float(candle['low'])
    c = float(candle['close'])
    body = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return False
    if body > 0.3 * total_range:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if c >= o:
        return False
    if body == 0:
        return False
    if upper_wick >= body * 2 and lower_wick <= body * 0.5:
        return True
    return False

def is_bullish_pinbar(candle):
    o = float(candle['open'])
    h = float(candle['high'])
    l = float(candle['low'])
    c = float(candle['close'])
    body = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return False
    if body > 0.3 * total_range:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if c <= o:
        return False
    if body == 0:
        return False
    if lower_wick >= body * 2 and upper_wick <= body * 0.5:
        return True
    return False

def price_in_zone(price, zone):
    return (zone["low"] * (1 - ZONE_TOLERANCE) <= price <= zone["high"] * (1 + ZONE_TOLERANCE))

def check_short_signal(symbol, df_5m, zones):
    last = df_5m.iloc[-2]  # последняя закрытая свеча

    timestamp = int(last["time"].timestamp())
    price = last["close"]

    if not is_bearish_pinbar_df(df_5m, len(df_5m) - 2):
        return None

    for zone in zones:
        if price_in_zone(price, zone):
            key = f"{symbol}_{timestamp}_short"
            if key in sent_signals:
                return None
            sent_signals[key] = True
            return zone

    return None

def is_swing_high(df, i, left=2, right=2):
    if i < left or i + right >= len(df):
        return False
    return df['high'].iloc[i] == max(df['high'].iloc[i-left:i+right+1])

def is_swing_low(df, i, left=2, right=2):
    if i < left or i + right >= len(df):
        return False
    return df['low'].iloc[i] == min(df['low'].iloc[i-left:i+right+1])


def check_long_signal(symbol, df_5m, zones):
    last = df_5m.iloc[-2]

    timestamp = int(last["time"].timestamp())
    price = last["close"]

    if not is_bullish_pinbar_df(df_5m, len(df_5m) - 2):
        return None

    for zone in zones:
        if price_in_zone(price, zone):
            key = f"{symbol}_{timestamp}_long"
            if key in sent_signals:
                return None
            sent_signals[key] = True
            return zone

    return None

# ================= ГЕНЕРАЦИЯ ГРАФИКА =================

def generate_chart(symbol, klines, supply_zones, demand_zones, signal_zone=None):
    df_full = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df_full[["open", "high", "low", "close"]] = df_full[["open", "high", "low", "close"]].astype(float)
    df_full['timestamp'] = df_full['timestamp'].astype(int)
    df_full['datetime'] = pd.to_datetime(df_full['timestamp'], unit='ms')

    offset = max(0, len(df_full) - CHART_CANDLES)
    df = df_full.iloc[offset:].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(20, 10))

    # Рисуем свечи
    for i in range(len(df)):
        o = df['open'][i]
        h = df['high'][i]
        l = df['low'][i]
        c = df['close'][i]
        color = 'green' if c > o else 'red'
        ax.add_patch(patches.Rectangle(
            (i - 0.2, min(o, c)),
            0.4,
            abs(c - o),
            facecolor=color,
            edgecolor=color
        ))
        ax.plot([i, i], [l, min(o, c)], color='black', linewidth=1)
        ax.plot([i, i], [max(o, c), h], color='black', linewidth=1)

    # Рисуем все SUPPLY зоны (красные)
    for zone in supply_zones:
        abs_start = zone['start_bar']
        rel_start = max(0, abs_start - offset)
        if rel_start >= len(df):
            continue
        x_end = len(df)
        color = 'red'
        ax.add_patch(patches.Rectangle(
            (rel_start, zone['low']),
            x_end - rel_start,
            zone['high'] - zone['low'],
            facecolor=color,
            alpha=0.2,
            edgecolor=color,
            linewidth=1
        ))
       

    # Рисуем все DEMAND зоны (синие)
    for zone in demand_zones:
        abs_start = zone['start_bar']
        rel_start = max(0, abs_start - offset)
        if rel_start >= len(df):
            continue
        x_end = len(df)
        color = 'blue'
        ax.add_patch(patches.Rectangle(
            (rel_start, zone['low']),
            x_end - rel_start,
            zone['high'] - zone['low'],
            facecolor=color,
            alpha=0.2,
            edgecolor=color,
            linewidth=1
        ))
        

    # Если есть сигнальная зона, выделяем её жирной обводкой
    if signal_zone:
        abs_start = signal_zone['start_bar']
        rel_start = max(0, abs_start - offset)
        if rel_start < len(df):
            color = 'red' if signal_zone['type'] == 'supply' else 'blue'
            ax.add_patch(patches.Rectangle(
                (rel_start, signal_zone['low']),
                len(df) - rel_start,
                signal_zone['high'] - signal_zone['low'],
                facecolor='none',
                edgecolor=color,
                linewidth=3,
                linestyle='-'
            ))

    ax.set_title(f"{symbol} TF:{ZONE_TF}")
    step = max(1, len(df) // 10)
    ax.set_xticks(range(0, len(df), step))
    ax.set_xticklabels(df['datetime'][::step].dt.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
    ax.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

# ================= TELEGRAM =================

async def send_telegram(image, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    data = aiohttp.FormData()
    data.add_field("chat_id", CHAT_ID)
    data.add_field("caption", caption)
    data.add_field("photo", image, filename="chart.png")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as resp:
            return await resp.text()

# ================= ОБРАБОТКА СИМВОЛА =================

async def process_symbol(symbol, semaphore):
    async with semaphore:
        try:
            # 1. Получаем DataFrame свечей для зон (ZONE_TF)
            df_zone = bx.get_klines(symbol, ZONE_TF, ZONE_LIMIT)
   

            # 2. Вычисляем зоны (функция возвращает списки словарей)
            supply_zones, demand_zones = find_supply_demand_zones(df_zone)
           
            # 3. Текущая цена – последнее закрытие
            current_price = df_zone['close'].iloc[-1]

            # 4. Оставляем только 2 ближайшие зоны каждого типа
            supply_zones = get_nearest_zones(current_price, supply_zones, 2)
            demand_zones = get_nearest_zones(current_price, demand_zones, 2)
            
            # 5. Фильтруем использованные зоны (одноразовые)
            used = used_zones.get(symbol, set())
            supply_zones = [z for z in supply_zones if (z['high'], z['low']) not in used]
            demand_zones = [z for z in demand_zones if (z['high'], z['low']) not in used]

            if not supply_zones and not demand_zones:
                return

            # 6. Получаем последние 2 свечи на 5m (тоже DataFrame)
            df_5m = bx.get_klines(symbol, "5m", 2)
            if df_5m.empty or len(df_5m) < 2:
                return

            # 7. Предпоследняя свеча (последняя закрытая)
            last_5m = df_5m.iloc[-2]
            c = last_5m['close']
            h = last_5m['high']
            l = last_5m['low']

            # 8. Проверяем сигналы (функции должны работать с DataFrame)
            short_zone = check_short_signal(symbol, df_5m, supply_zones)
            if short_zone:
                is_short = True
                zone = short_zone
            else:
                long_zone = check_long_signal(symbol, df_5m, demand_zones)
                if long_zone:
                    is_short = False
                    zone = long_zone
                else:
                    return
            
            # 9. Отмечаем зону как использованную
            used_zones.setdefault(symbol, set()).add((zone['high'], zone['low']))

            # 10. Генерируем график
            chart = generate_chart(symbol, df_zone, supply_zones, demand_zones, signal_zone=zone)

            # 11. Отправляем уведомление
            signal_type = "Short" if is_short else "Long"
            stop = h if is_short else l

            caption = (
                f"🚨 {symbol} {signal_type} Signal\n"
                f"TF: {ZONE_TF} (Pinbar on 5m)\n"
                f"Zone: {zone['low']:.6f} - {zone['high']:.6f}\n"
                f"Current price: {current_price:.6f}\n"
                f"Entry: {current_price:.6f}\n"
                f"Stop: {stop:.6f}"
            )
            await send_telegram(chart, caption)
            print(f"Signal найден: {symbol} {signal_type}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{symbol} error: {e}")

# ================= ГЛАВНЫЙ ЦИКЛ =================

async def main_loop():
    symbols = bx.get_all_tikers()
    print(f"Найдено {len(symbols)} монет")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    while True:
        print("Начинаем сканирование...")
        tasks = [process_symbol(symbol, semaphore) for symbol in symbols]
        await asyncio.gather(*tasks)
        print("Сканирование завершено\n")
        await asyncio.sleep(SCAN_INTERVAL_SEC)

if __name__ == "__main__":
    asyncio.run(main_loop())