#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полностью функциональный прототип ГИС-приложения:
Профиль рельефа Шкловского района (Могилёвская область, РБ) +
вышки сотовой связи + границы населённых пунктов +
проектирование радиосети LoRa (MESH) для системы удалённой идентификации БЛА.

Версия 2.0 — все запрошенные доработки:
1. Убраны атрибуты Leaflet
2. Учёт высоты вышек при размещении LoRa
3. Спутник Esri с подписями
4. Загрузка локального GeoTIFF DEM
5. Расчёт зон Френеля и LOS
6. Интеграция OpenCelliD
7. 3D-визуализация (pydeck)
8. Экспорт в KML / GeoJSON

Запуск: streamlit run app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, MeasureControl, Fullscreen, MiniMap, Draw
import requests
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from shapely.geometry import Point, box, mapping
import json
import time
import io
import zipfile
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import base64

# Опциональные зависимости
try:
    import rasterio
    from rasterio.transform import rowcol
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import pydeck as pdk
    HAS_PYDECK = True
except ImportError:
    HAS_PYDECK = False

# ====================== КОНСТАНТЫ ======================
BBOX = {
    "north": 54.342778,
    "south": 53.963056,
    "east":  30.919444,
    "west":  29.999444
}
CENTER = [(BBOX["north"] + BBOX["south"]) / 2,
          (BBOX["east"]  + BBOX["west"])  / 2]

# Центр города Шклов + радиус для размещения основного/резервного шлюза и городских репитеров
SHKLOV_CENTER = {"lat": 54.2133, "lon": 30.2878}
SHKLOV_RADIUS_KM = 4.5

MESHTASTIC_PRESETS = {
    "LongFast": {
        "sf": 11, "bw": 250, "cr": "4/8",
        "range_km_flat": 8.0, "range_km_elev": 25.0,
        "sensitivity_dbm": -132,   # типичная чувствительность при SF11
        "desc": "Дальний быстрый (рекомендуется для MESH БЛА)"
    },
    "LongSlow": {
        "sf": 12, "bw": 125, "cr": "4/8",
        "range_km_flat": 12.0, "range_km_elev": 40.0,
        "sensitivity_dbm": -137,
        "desc": "Максимальная дальность"
    },
    "MediumFast": {
        "sf": 9, "bw": 250, "cr": "4/8",
        "range_km_flat": 4.0, "range_km_elev": 12.0,
        "sensitivity_dbm": -126,
        "desc": "Средний баланс"
    },
    "MediumSlow": {
        "sf": 10, "bw": 250, "cr": "4/8",
        "range_km_flat": 5.5, "range_km_elev": 18.0,
        "sensitivity_dbm": -129,
        "desc": "Средний медленный"
    },
    "ShortFast": {
        "sf": 7, "bw": 250, "cr": "4/5",
        "range_km_flat": 1.5, "range_km_elev": 5.0,
        "sensitivity_dbm": -120,
        "desc": "Короткий быстрый"
    },
    "ShortSlow": {
        "sf": 8, "bw": 125, "cr": "4/8",
        "range_km_flat": 2.5, "range_km_elev": 8.0,
        "sensitivity_dbm": -123,
        "desc": "Короткий медленный"
    },
    "LongModerate": {
        "sf": 11, "bw": 125, "cr": "4/8",
        "range_km_flat": 10.0, "range_km_elev": 30.0,
        "sensitivity_dbm": -134,
        "desc": "Дальний умеренный"
    }
}

# Физические ограничения сети Remote ID БЛА
LORA_CONSTRAINTS = {
    "beacon_tx_dbm": 15.0,       # мощность маяка БЛА (макс. 15 dBm)
    "node_tx_dbm_max": 33.0,     # 2 Вт = 33 dBm (макс. для репитера/шлюза)
    "antenna_gain_dbi": 2.15,    # типичная четвертьволновая / 2 dBi
    "fade_margin_db": 12.0,      # запас на замирания + тело БЛА + лес
    "cable_loss_db": 0.5,
}

# Модели затухания (path loss) для 868 МГц
# n — показатель степени в log-distance модели; A0 — потери на 1 км
PATH_LOSS_MODELS = {
    "FSPL": {
        "name": "Свободное пространство (FSPL)",
        "desc": "Идеальная LOS без препятствий. Оптимистичная оценка.",
        "n": 2.0, "A0_offset": 0.0, "clutter_db": 0.0
    },
    "TwoRay": {
        "name": "Двухлучевая (земля + прямой)",
        "desc": "Прямой луч + отражение от земли. Для открытой местности.",
        "n": 2.5, "A0_offset": 3.0, "clutter_db": 0.0
    },
    "Rural_Open": {
        "name": "Сельская открытая (ITU-R P.1546 / эмпир.)",
        "desc": "Поля, луга Шкловского района. Рекомендуется по умолчанию.",
        "n": 2.7, "A0_offset": 5.0, "clutter_db": 2.0
    },
    "Suburban": {
        "name": "Пригород / деревня",
        "desc": "Застройка низкой плотности, сады, отдельные здания.",
        "n": 3.0, "A0_offset": 8.0, "clutter_db": 6.0
    },
    "Forest": {
        "name": "Лес / сильная растительность",
        "desc": "Лиственный/хвойный лес. Доп. потери на растительность.",
        "n": 3.3, "A0_offset": 10.0, "clutter_db": 12.0
    },
    "Urban_Light": {
        "name": "Город лёгкий (Шклов)",
        "desc": "Малоэтажная застройка районного центра.",
        "n": 3.5, "A0_offset": 12.0, "clutter_db": 10.0
    },
}

OPERATORS_BY = {
    "A1": "#E30613",
    "МТС": "#FF0000",
    "life:)": "#FFD700",
    "beCloud": "#00A0E3",
    "прочие": "#808080",
    "OpenCelliD": "#9B59B6"
}

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def fresnel_radius_m(d1_m: np.ndarray, d2_m: np.ndarray, freq_mhz: float, n: int = 1) -> np.ndarray:
    """
    Векторный расчёт радиуса n-й зоны Френеля (метры).
    d1_m, d2_m — массивы расстояний до концов трассы в метрах.
    Формула: r_n = sqrt( n * λ * d1 * d2 / (d1 + d2) )
    """
    wavelength = 299.792458 / freq_mhz  # м
    total = d1_m + d2_m
    # защита от деления на ноль
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(n * wavelength * d1_m * d2_m / np.where(total > 0, total, np.nan))
    return np.nan_to_num(r, nan=0.0)


def earth_bulge_m(d1_m: np.ndarray, d2_m: np.ndarray, k_factor: float = 4.0/3.0) -> np.ndarray:
    """
    Выпуклость Земли (метры) с учётом эффективного радиуса (k-фактор).
    Стандарт ITU-R: k = 4/3 для стандартной атмосферы.
    h_bulge = (d1 * d2) / (2 * R_eff), где R_eff = k * 6371 км.
    """
    R_earth_m = 6371000.0
    R_eff = k_factor * R_earth_m
    return (d1_m * d2_m) / (2.0 * R_eff)


def sample_elevation_profile(lat1: float, lon1: float, lat2: float, lon2: float,
                             elev_func, n_samples: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Оптимизированный сэмплинг профиля.
    Возвращает: distances_km, elevations, lats, lons
    Использует больше точек в середине трассы (где зона Френеля максимальна).
    """
    # Равномерная параметризация + сгущение в центре
    t = np.linspace(0, 1, n_samples)
    # небольшое сгущение к середине (синусоидальное)
    t_dense = 0.5 - 0.5 * np.cos(np.pi * t)
    lats = lat1 + (lat2 - lat1) * t_dense
    lons = lon1 + (lon2 - lon1) * t_dense

    # Векторный расчёт расстояний от начальной точки
    dists = np.array([haversine(lat1, lon1, la, lo) for la, lo in zip(lats, lons)])
    elevs = np.array([elev_func(la, lo) for la, lo in zip(lats, lons)], dtype=float)

    # Линейная интерполяция пропущенных (nan) значений
    mask = np.isnan(elevs)
    if np.any(mask) and not np.all(mask):
        elevs[mask] = np.interp(dists[mask], dists[~mask], elevs[~mask])

    return dists, elevs, lats, lons


def check_los(dists_km: np.ndarray, elevs: np.ndarray,
              h1_asl: float, h2_asl: float,
              freq_mhz: float,
              clearance_pct: float = 60.0,
              k_factor: float = 4.0/3.0,
              n_fresnel: int = 1) -> Dict:
    """
    Полный расчёт LOS с зоной Френеля и кривизной Земли (оптимизированный, векторный).

    Параметры:
        dists_km     — массив расстояний от точки А (км)
        elevs        — высоты рельефа (м над ур. моря)
        h1_asl, h2_asl — абсолютные высоты антенн (м ASL)
        freq_mhz     — частота
        clearance_pct — требуемый процент свободной 1-й зоны Френеля (обычно 60 %)
        k_factor     — коэффициент преломления атмосферы (4/3 стандарт)
        n_fresnel    — номер зоны Френеля (1 — основная)

    Возвращает словарь с результатами и массивами для построения профиля.
    """
    total_km = float(dists_km[-1])
    if total_km < 0.05:
        return {
            "los": True, "min_clearance_m": 9999.0, "blocked_at_km": None,
            "fresnel_r_mid_m": 0.0, "worst_idx": 0,
            "los_line": elevs.copy(), "fresnel_upper": elevs.copy(),
            "fresnel_lower": elevs.copy(), "bulge": np.zeros_like(elevs),
            "clearance": np.full_like(elevs, 9999.0)
        }

    dists_m = dists_km * 1000.0
    d1_m = dists_m
    d2_m = dists_m[-1] - dists_m

    # 1. Кривизна Земли
    bulge = earth_bulge_m(d1_m, d2_m, k_factor)

    # 2. Линия прямой видимости (с учётом выпуклости)
    # Антенны уже в ASL, линия LOS в ASL:
    los_line = h1_asl + (h2_asl - h1_asl) * (dists_km / total_km)

    # 3. Радиус зоны Френеля (векторно)
    r_f = fresnel_radius_m(d1_m, d2_m, freq_mhz, n=n_fresnel)
    required = r_f * (clearance_pct / 100.0)

    # 4. Нижняя граница допустимой зоны (LOS минус требуемый зазор минус выпуклость)
    # Рельеф не должен заходить выше этой границы
    fresnel_lower = los_line - required - bulge
    fresnel_upper = los_line + required          # для визуализации

    # 5. Зазор: насколько рельеф ниже нижней границы Френеля
    clearance = fresnel_lower - elevs

    min_clear = float(np.nanmin(clearance[1:-1])) if len(clearance) > 2 else float(np.nanmin(clearance))
    worst_idx = int(np.nanargmin(clearance[1:-1])) + 1 if len(clearance) > 2 else 0
    blocked_at = float(dists_km[worst_idx]) if min_clear < 0 else None

    # Радиус в середине трассы
    mid = len(dists_km) // 2
    r_mid = float(r_f[mid])

    return {
        "los": min_clear >= 0.0,
        "min_clearance_m": min_clear,
        "blocked_at_km": blocked_at,
        "fresnel_r_mid_m": r_mid,
        "worst_idx": worst_idx,
        "los_line": los_line,
        "fresnel_upper": fresnel_upper,
        "fresnel_lower": fresnel_lower,
        "bulge": bulge,
        "clearance": clearance,
        "r_fresnel": r_f
    }


# ====================== ЗАГРУЗКА ДАННЫХ ======================

@st.cache_data(ttl=3600, show_spinner="Загрузка рельефа (SRTM 30м)...")
def get_elevation_grid_opentopo(n_points: int = 22) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats = np.linspace(BBOX["south"], BBOX["north"], n_points)
    lons = np.linspace(BBOX["west"],  BBOX["east"],  n_points)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    locations = [f"{lat_grid[i,j]:.6f},{lon_grid[i,j]:.6f}"
                 for i in range(n_points) for j in range(n_points)]
    elevs = np.full((n_points, n_points), np.nan)
    batch_size = 100
    for start in range(0, len(locations), batch_size):
        batch = locations[start:start+batch_size]
        loc_str = "|".join(batch)
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={loc_str}"
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 200:
                data = r.json()
                for idx, res in enumerate(data.get("results", [])):
                    if res.get("elevation") is not None:
                        flat = start + idx
                        i, j = divmod(flat, n_points)
                        elevs[i, j] = res["elevation"]
            time.sleep(0.25)
        except Exception as e:
            st.warning(f"OpenTopoData ошибка: {e}")
    return lat_grid, lon_grid, elevs


def load_geotiff_dem(uploaded_file) -> Optional[Dict]:
    """Загружает GeoTIFF и возвращает объект для сэмплирования."""
    if not HAS_RASTERIO:
        st.error("rasterio не установлен. Выполните: pip install rasterio")
        return None
    try:
        bytes_data = uploaded_file.read()
        with rasterio.open(io.BytesIO(bytes_data)) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            bounds = src.bounds
        return {
            "data": data,
            "transform": transform,
            "crs": str(crs),
            "nodata": nodata,
            "bounds": bounds,
            "shape": data.shape
        }
    except Exception as e:
        st.error(f"Ошибка чтения GeoTIFF: {e}")
        return None


def load_kmz_overlays(uploaded_file, max_level: int = 3, max_tiles: int = 80) -> List[Dict]:
    """
    Загружает KMZ (в т.ч. SuperOverlay / hierarchical tiles) и извлекает
    GroundOverlay / ImageOverlay тайлы для отображения на карте Folium.

    Поддерживает структуру вашего файла viz.AW3D30_hillshade-color.kmz:
    - doc.kml → NetworkLink → вложенные .kml + .png
    - каждый листовой .kml содержит LatLonBox + путь к PNG

    Возвращает список словарей:
    [{"north":.., "south":.., "east":.., "west":.., "png_bytes": bytes, "name": str}, ...]
    """
    import re
    import xml.etree.ElementTree as ET

    try:
        data = uploaded_file.read()
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as e:
        st.error(f"Не удалось открыть KMZ как ZIP: {e}")
        return []

    # Собираем все KML и PNG
    kml_files = [n for n in zf.namelist() if n.lower().endswith(".kml")]
    png_files = set(n for n in zf.namelist() if n.lower().endswith(".png"))

    overlays = []
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    def parse_box(elem):
        """Извлекает LatLonBox / LatLonAltBox."""
        box = elem.find(".//kml:LatLonBox", ns)
        if box is None:
            box = elem.find(".//kml:LatLonAltBox", ns)
        if box is None:
            # без namespace
            box = elem.find(".//LatLonBox")
            if box is None:
                box = elem.find(".//LatLonAltBox")
        if box is None:
            return None
        def get(tag):
            for t in [f"kml:{tag}", tag]:
                e = box.find(t, ns) if "kml" in t else box.find(t)
                if e is not None and e.text:
                    return float(e.text.strip())
            return None
        n, s, e, w = get("north"), get("south"), get("east"), get("west")
        if None in (n, s, e, w):
            return None
        return {"north": n, "south": s, "east": e, "west": w}

    def find_href(elem):
        """Ищет путь к PNG (Icon/href или просто href)."""
        for path in [".//kml:Icon/kml:href", ".//Icon/href", ".//kml:href", ".//href"]:
            el = elem.find(path, ns) if "kml:" in path else elem.find(path)
            if el is not None and el.text:
                return el.text.strip()
        return None

    for kml_name in kml_files:
        # Определяем уровень по пути (0/, 1/, 2/, 3/, 4/)
        level = -1
        m = re.match(r"^(\d+)/", kml_name)
        if m:
            level = int(m.group(1))
        if level > max_level:
            continue

        try:
            raw = zf.read(kml_name)
            # убираем namespace проблемы
            text = raw.decode("utf-8", errors="ignore")
            # иногда namespace мешает — пробуем оба способа
            try:
                root = ET.fromstring(raw)
            except Exception:
                text2 = re.sub(r'\sxmlns="[^"]+"', "", text, count=1)
                root = ET.fromstring(text2.encode("utf-8"))
        except Exception:
            continue

        # Ищем GroundOverlay или любой элемент с LatLonBox + href на png
        candidates = list(root.iter())
        for elem in candidates:
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag not in ("GroundOverlay", "NetworkLink", "Document", "Folder", "Region"):
                # всё равно проверяем наличие box + href
                pass

            box = parse_box(elem)
            if not box:
                continue
            href = find_href(elem)
            if not href or not href.lower().endswith(".png"):
                continue

            # Нормализуем путь к PNG относительно kml_name
            base_dir = "/".join(kml_name.split("/")[:-1])
            if href.startswith("../"):
                # поднимаемся
                parts = base_dir.split("/")
                href_clean = href
                while href_clean.startswith("../"):
                    href_clean = href_clean[3:]
                    if parts:
                        parts.pop()
                png_path = "/".join(parts + [href_clean]) if parts else href_clean
            elif href.startswith("/"):
                png_path = href[1:]
            else:
                png_path = f"{base_dir}/{href}" if base_dir else href

            png_path = png_path.replace("//", "/").lstrip("/")

            if png_path not in png_files:
                # пробуем просто имя файла
                just_name = href.split("/")[-1]
                matches = [p for p in png_files if p.endswith(just_name)]
                if matches:
                    png_path = matches[0]
                else:
                    continue

            try:
                png_bytes = zf.read(png_path)
            except KeyError:
                continue

            overlays.append({
                "north": box["north"],
                "south": box["south"],
                "east": box["east"],
                "west": box["west"],
                "png_bytes": png_bytes,
                "name": png_path,
                "level": level
            })

            if len(overlays) >= max_tiles:
                break
        if len(overlays) >= max_tiles:
            break

    # Оставляем самые детальные тайлы (высокий level), но не больше max_tiles
    if overlays:
        overlays.sort(key=lambda x: (-x.get("level", 0), x["name"]))
        overlays = overlays[:max_tiles]

    return overlays


def make_elev_func_from_raster(raster_dict):
    """Возвращает функцию elev(lat, lon) из загруженного растра."""
    data = raster_dict["data"]
    transform = raster_dict["transform"]
    nodata = raster_dict["nodata"]

    def elev_func(lat, lon):
        try:
            row, col = rowcol(transform, lon, lat)
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                val = data[row, col]
                if nodata is not None and val == nodata:
                    return np.nan
                return float(val)
        except Exception:
            pass
        return np.nan
    return elev_func


def make_elev_func_from_grid(lat_grid, lon_grid, elevs):
    """Билинейная интерполяция по сетке."""
    from scipy.interpolate import RegularGridInterpolator
    lats = lat_grid[:, 0]
    lons = lon_grid[0, :]
    # Заменяем nan на среднее для интерполятора
    elevs_filled = np.where(np.isnan(elevs), np.nanmean(elevs), elevs)
    interp = RegularGridInterpolator((lats, lons), elevs_filled,
                                     bounds_error=False, fill_value=np.nanmean(elevs))

    def elev_func(lat, lon):
        try:
            return float(interp([[lat, lon]])[0])
        except Exception:
            return np.nan
    return elev_func


@st.cache_data(ttl=7200, show_spinner="Загрузка населённых пунктов (OSM)...")
def get_settlements() -> List[Dict]:
    query = f"""
    [out:json][timeout:60];
    (
      node["place"~"city|town|village|hamlet|suburb|locality"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
      way["place"~"city|town|village|hamlet"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
    );
    out center tags;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=90)
        if r.status_code == 200:
            result = []
            for el in r.json().get("elements", []):
                tags = el.get("tags", {})
                name = tags.get("name") or tags.get("name:ru") or tags.get("name:be") or "без имени"
                place = tags.get("place", "unknown")
                if "center" in el:
                    lat, lon = el["center"]["lat"], el["center"]["lon"]
                elif "lat" in el:
                    lat, lon = el["lat"], el["lon"]
                else:
                    continue
                result.append({"name": name, "place": place, "lat": lat, "lon": lon,
                               "population": tags.get("population")})
            return result
    except Exception as e:
        st.warning(f"Overpass (НП): {e}")
    return []


@st.cache_data(ttl=7200, show_spinner="Загрузка вышек (OSM)...")
def get_cell_towers_osm() -> List[Dict]:
    query = f"""
    [out:json][timeout:90];
    (
      node["man_made"~"tower|mast"]["tower:type"="communication"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
      node["communication"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
      node["man_made"="tower"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
      way["man_made"~"tower|mast"]["tower:type"="communication"]({BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']});
    );
    out center tags;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=120)
        if r.status_code == 200:
            result = []
            for el in r.json().get("elements", []):
                tags = el.get("tags", {})
                if "center" in el:
                    lat, lon = el["center"]["lat"], el["center"]["lon"]
                elif "lat" in el:
                    lat, lon = el["lat"], el["lon"]
                else:
                    continue
                operator = tags.get("operator") or tags.get("owner") or "неизвестно"
                op_lower = operator.lower()
                if "a1" in op_lower or "velcom" in op_lower:
                    op_key = "A1"
                elif "мтс" in op_lower or "mts" in op_lower:
                    op_key = "МТС"
                elif "life" in op_lower or "лайф" in op_lower:
                    op_key = "life:)"
                elif "becloud" in op_lower or "белтелеком" in op_lower:
                    op_key = "beCloud"
                else:
                    op_key = "прочие"
                h = tags.get("height") or tags.get("tower:height")
                try:
                    h = float(str(h).replace("m", "").replace(",", ".").strip()) if h else None
                except Exception:
                    h = None
                result.append({
                    "lat": lat, "lon": lon, "operator": op_key,
                    "raw_operator": operator, "height_m": h,
                    "ref": tags.get("ref") or tags.get("name") or "",
                    "source": "OSM"
                })
            return result
    except Exception as e:
        st.warning(f"Overpass (вышки): {e}")
    return []


def get_opencellid_towers(api_key: str) -> List[Dict]:
    """
    Загружает вышки из OpenCelliD по bbox.
    Согласно официальной документации https://wiki.opencellid.org/wiki/API :
    - BBOX = latmin,lonmin,latmax,lonmax
    - limit максимум 50 за один запрос → используем пагинацию offset
    - format=json
    - radio: GSM / UMTS / LTE / NR (по одному значению или без фильтра)
    """
    if not api_key or len(api_key) < 8:
        return []

    url = "https://opencellid.org/cell/getInArea"
    # Правильный порядок: latmin, lonmin, latmax, lonmax
    bbox_str = f"{BBOX['south']},{BBOX['west']},{BBOX['north']},{BBOX['east']}"

    result = []
    seen = set()
    offset = 0
    limit = 50          # максимум по документации
    max_pages = 20      # защита от слишком большого числа запросов (1000 точек)

    # MCC Беларуси = 257 (можно фильтровать)
    mcc_by = 257

    try:
        while offset < limit * max_pages:
            params = {
                "key": api_key,
                "BBOX": bbox_str,
                "format": "json",
                "limit": limit,
                "offset": offset,
                "mcc": mcc_by,          # только Беларусь
                # radio не указываем — получаем все типы (GSM, UMTS, LTE, NR)
            }
            r = requests.get(url, params=params, timeout=45)

            if r.status_code == 429:
                st.warning("OpenCelliD: превышен дневной лимит запросов (ошибка 7 / HTTP 429).")
                break
            if r.status_code == 503:
                st.warning("OpenCelliD: слишком много запросов, попробуйте позже (HTTP 503).")
                break
            if r.status_code != 200:
                st.warning(f"OpenCelliD HTTP {r.status_code}: {r.text[:250]}")
                break

            data = r.json()
            cells = data.get("cells", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

            if not cells:
                break

            for c in cells:
                lat = c.get("lat")
                lon = c.get("lon")
                if lat is None or lon is None:
                    continue
                key = (round(float(lat), 5), round(float(lon), 5))
                if key in seen:
                    continue
                seen.add(key)

                mcc = c.get("mcc")
                mnc = c.get("mnc")
                # Простая расшифровка операторов РБ по MNC (достоверные данные)
                op_name = "OpenCelliD"
                if mcc == 257:
                    mnc_map = {
                        1: "А1 (velcom)",
                        2: "МТС",
                        4: "life:)",
                        3: "beCloud / BelCel",
                        6: "beCloud",
                    }
                    op_name = mnc_map.get(mnc, f"MCC257/MNC{mnc}")

                result.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "operator": "OpenCelliD",
                    "raw_operator": op_name,
                    "height_m": None,
                    "ref": str(c.get("cellid") or c.get("cid") or ""),
                    "source": "OpenCelliD",
                    "radio": c.get("radio") or "",
                    "range_m": c.get("range"),
                    "samples": c.get("samples"),
                })

            # Если пришло меньше limit — больше страниц нет
            if len(cells) < limit:
                break
            offset += limit
            time.sleep(0.35)  # вежливая пауза между страницами

    except Exception as e:
        st.warning(f"OpenCelliD ошибка: {e}")

    return result


def find_local_extrema(elevs, lat_grid, lon_grid, size=3, n_high=18, n_low=12):
    valid = ~np.isnan(elevs)
    if not np.any(valid):
        return [], []
    max_filt = maximum_filter(np.nan_to_num(elevs, nan=-9999), size=size)
    high_mask = (elevs == max_filt) & valid & (elevs > np.nanpercentile(elevs, 65))
    min_filt = minimum_filter(np.nan_to_num(elevs, nan=9999), size=size)
    low_mask = (elevs == min_filt) & valid & (elevs < np.nanpercentile(elevs, 35))
    highs, lows = [], []
    for i, j in zip(*np.where(high_mask)):
        highs.append({"lat": float(lat_grid[i,j]), "lon": float(lon_grid[i,j]),
                      "elev": float(elevs[i,j]), "type": "high", "source": "relief"})
    for i, j in zip(*np.where(low_mask)):
        lows.append({"lat": float(lat_grid[i,j]), "lon": float(lon_grid[i,j]),
                     "elev": float(elevs[i,j]), "type": "low", "source": "relief"})
    highs = sorted(highs, key=lambda x: -x["elev"])[:n_high]
    lows = sorted(lows, key=lambda x: x["elev"])[:n_low]
    return highs, lows


def path_loss_db(d_km: float, freq_mhz: float, model_key: str = "Rural_Open",
                 extra_clutter_db: float = 0.0) -> float:
    """
    Расчёт затухания на трассе (dB) по выбранной модели.

    Базовая формула log-distance:
      PL(d) = FSPL(1 km) + 10·n·log10(d_km) + A0_offset + clutter + extra

    FSPL(1 km) = 20·log10(f_MHz) + 32.44
    """
    d_km = max(d_km, 0.01)
    model = PATH_LOSS_MODELS.get(model_key, PATH_LOSS_MODELS["Rural_Open"])
    n = model["n"]
    a0 = model["A0_offset"]
    clutter = model["clutter_db"] + extra_clutter_db

    fspl_1km = 20.0 * np.log10(freq_mhz) + 32.44
    pl = fspl_1km + 10.0 * n * np.log10(d_km) + a0 + clutter
    return float(pl)


def calc_max_range_km(tx_dbm: float, rx_sensitivity_dbm: float,
                      tx_gain: float = 2.15, rx_gain: float = 2.15,
                      fade_margin: float = 12.0, freq_mhz: float = 868.0,
                      elev_bonus_km: float = 0.0,
                      model_key: str = "Rural_Open",
                      extra_clutter_db: float = 0.0,
                      body_loss_db: float = 0.0) -> float:
    """
    Максимальная дальность по link budget + уточнённая модель затухания.

    Link budget:
      Pr = Pt + Gt + Gr - PL(d) - fade - cable - body
      Pr >= sensitivity  =>  PL(d) <= Pt+Gt+Gr - sens - fade - cable - body

    PL(d) = FSPL(1km) + 10·n·log10(d) + offsets
    Решаем: d = 10^((PL_allowed - FSPL1km - offsets) / (10·n))
    """
    model = PATH_LOSS_MODELS.get(model_key, PATH_LOSS_MODELS["Rural_Open"])
    n = model["n"]
    a0 = model["A0_offset"]
    clutter = model["clutter_db"] + extra_clutter_db

    allowed_pl = (tx_dbm + tx_gain + rx_gain
                  - rx_sensitivity_dbm
                  - fade_margin
                  - LORA_CONSTRAINTS["cable_loss_db"]
                  - body_loss_db)

    fspl_1km = 20.0 * np.log10(freq_mhz) + 32.44
    # allowed_pl >= fspl_1km + 10*n*log10(d) + a0 + clutter
    residual = allowed_pl - fspl_1km - a0 - clutter
    if residual <= 0 or n <= 0:
        return 0.3

    d_km = 10.0 ** (residual / (10.0 * n))
    d_km = d_km + elev_bonus_km
    return float(min(max(d_km, 0.3), 80.0))


def suggest_lora_mesh(candidates: List[Dict], preset_name: str,
                      freq_mhz: float,
                      max_hops: int = 3,
                      n_gateways: int = 2,
                      n_city_repeaters: int = 3,
                      beacon_tx_dbm: float = 15.0,
                      node_tx_dbm: float = 33.0,
                      antenna_agl_m: float = 10.0,
                      uav_height_m: float = 50.0,
                      path_loss_model: str = "Rural_Open",
                      extra_clutter_db: float = 0.0,
                      fade_margin_db: float = 12.0,
                      body_loss_db: float = 3.0) -> Dict:
    """
    Планирование сети LoRa MESH для Remote ID БЛА.

    Правила:
    - ОСНОВНОЙ шлюз ВСЕГДА в городе Шклов (районный центр).
    - Резервный шлюз — тоже в зоне Шклова (резервирование).
    - Дополнительные городские репитеры в Шклове для разгрузки.
    - Остальные репитеры — по району с учётом высоты и резервирования путей.
    - antenna_agl_m — итоговая высота антенны LoRa над уровнем земли (м AGL).
    """
    empty = {"gateways": [], "repeaters": [], "all_nodes": [],
             "beacon_range_km": 0, "node_range_km": 0, "links": [],
             "coverage_ok": False, "max_hops": max_hops,
             "primary_gateway": None, "backup_gateway": None}

    if not candidates:
        return empty

    preset = MESHTASTIC_PRESETS[preset_name]
    sens = preset.get("sensitivity_dbm", -130)

    # --- Дальности по link-budget + уточнённая модель затухания ---
    # Параметры model_key / extra_clutter / fade приходят из замыкания UI
    # (передаются через аргументы функции — см. вызов)
    uav_bonus = min(uav_height_m / 50.0, 3.0) * (preset["range_km_elev"] * 0.06)
    # Маяк БЛА: антенна ~0 dBi, доп. body loss на корпусе БЛА
    beacon_range = calc_max_range_km(
        tx_dbm=beacon_tx_dbm, rx_sensitivity_dbm=sens,
        tx_gain=0.0, rx_gain=LORA_CONSTRAINTS["antenna_gain_dbi"],
        fade_margin=fade_margin_db + 2.0,
        freq_mhz=freq_mhz,
        elev_bonus_km=preset["range_km_elev"] * 0.08 + uav_bonus,
        model_key=path_loss_model,
        extra_clutter_db=extra_clutter_db,
        body_loss_db=body_loss_db
    )
    # Узлы сети: обе антенны ~2 dBi
    node_range = calc_max_range_km(
        tx_dbm=min(node_tx_dbm, LORA_CONSTRAINTS["node_tx_dbm_max"]),
        rx_sensitivity_dbm=sens,
        tx_gain=LORA_CONSTRAINTS["antenna_gain_dbi"],
        rx_gain=LORA_CONSTRAINTS["antenna_gain_dbi"],
        fade_margin=fade_margin_db,
        freq_mhz=freq_mhz,
        elev_bonus_km=preset["range_km_elev"] * 0.18,
        model_key=path_loss_model,
        extra_clutter_db=extra_clutter_db,
        body_loss_db=0.0
    )
    min_dist = max(node_range * 0.45, 1.5)

    # --- Разделяем кандидатов: в зоне Шклова и остальные ---
    def dist_to_shklov(p):
        return haversine(p["lat"], p["lon"], SHKLOV_CENTER["lat"], SHKLOV_CENTER["lon"])

    city_cands = [p for p in candidates if dist_to_shklov(p) <= SHKLOV_RADIUS_KM]
    rural_cands = [p for p in candidates if dist_to_shklov(p) > SHKLOV_RADIUS_KM]

    # Если в городе мало кандидатов — добавляем виртуальную точку центра Шклова
    if len(city_cands) < 2:
        city_cands.insert(0, {
            "lat": SHKLOV_CENTER["lat"],
            "lon": SHKLOV_CENTER["lon"],
            "elev": 160.0 + antenna_agl_m,   # примерная высота местности + антенна
            "effective_elev": 160.0 + antenna_agl_m,
            "source": "city_center",
            "type": "forced_shklov"
        })

    # Сортировка городских: высота + бонус за вышку
    def city_score(p):
        s = p.get("effective_elev", p.get("elev", 0))
        if p.get("source") in ("OSM", "OpenCelliD", "tower"):
            s += 50
        # чуть ближе к центру — лучше
        s -= dist_to_shklov(p) * 3
        return s

    city_ranked = sorted(city_cands, key=city_score, reverse=True)

    # === ОСНОВНОЙ шлюз — ВСЕГДА в Шклове (первая точка) ===
    primary = dict(city_ranked[0])
    primary["role"] = "gateway"
    primary["gateway_type"] = "primary"
    primary["label"] = "Основной шлюз (Шклов) → Минск"
    primary["elev"] = primary.get("effective_elev", primary.get("elev", 160)) 

    # === РЕЗЕРВНЫЙ шлюз — вторая точка в зоне Шклова ===
    backup = None
    if len(city_ranked) >= 2:
        backup = dict(city_ranked[1])
    else:
        # смещаем немного от основного
        backup = dict(primary)
        backup["lat"] = primary["lat"] + 0.012
        backup["lon"] = primary["lon"] + 0.015
        backup["source"] = "backup_offset"
    backup["role"] = "gateway"
    backup["gateway_type"] = "backup"
    backup["label"] = "Резервный шлюз (Шклов) → Минск"
    backup["elev"] = backup.get("effective_elev", backup.get("elev", 160))

    gateways = [primary, backup]

    # === Городские репитеры в Шклове (разгрузка сети) ===
    city_reps = []
    used = {(round(primary["lat"],5), round(primary["lon"],5)),
            (round(backup["lat"],5), round(backup["lon"],5))}
    for p in city_ranked[2:]:
        key = (round(p["lat"],5), round(p["lon"],5))
        if key in used:
            continue
        if len(city_reps) >= n_city_repeaters:
            break
        # минимальное расстояние внутри города
        if any(haversine(p["lat"], p["lon"], r["lat"], r["lon"]) < 0.8 for r in city_reps):
            continue
        r = dict(p)
        r["role"] = "repeater"
        r["zone"] = "city"
        r["label"] = f"Городской репитер Шклов #{len(city_reps)+1}"
        r["elev"] = r.get("effective_elev", r.get("elev", 0))
        city_reps.append(r)
        used.add(key)

    # === Районные репитеры (жадно по высоте) ---
    rural_selected = []
    remaining = sorted(rural_cands, key=lambda x: -x.get("effective_elev", x.get("elev", 0)))
    while remaining:
        best = remaining.pop(0)
        key = (round(best["lat"],5), round(best["lon"],5))
        if key in used:
            continue
        if any(haversine(best["lat"], best["lon"], s["lat"], s["lon"]) < min_dist
               for s in rural_selected + gateways + city_reps):
            continue
        r = dict(best)
        r["role"] = "repeater"
        r["zone"] = "rural"
        r["label"] = f"Районный репитер #{len(rural_selected)+1}"
        r["elev"] = r.get("effective_elev", r.get("elev", 0))
        rural_selected.append(r)
        used.add(key)

    repeaters = city_reps + rural_selected
    all_nodes = gateways + repeaters

    # --- Граф связей ---
    from collections import deque
    n = len(all_nodes)
    links = []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(all_nodes[i]["lat"], all_nodes[i]["lon"],
                          all_nodes[j]["lat"], all_nodes[j]["lon"])
            if d <= node_range * 0.95:
                links.append({"from": i, "to": j, "dist_km": round(d, 2)})
                adj[i].append((j, d))
                adj[j].append((i, d))

    # --- Кратчайшие HOPы до ближайшего шлюза (оптимизированная маршрутизация) ---
    gw_indices = [i for i, nd in enumerate(all_nodes) if nd.get("role") == "gateway"]
    hops_to_gw = [-1] * n
    next_hop = [-1] * n          # следующий узел на пути к шлюзу
    path_to_gw = [[] for _ in range(n)]

    q = deque()
    for gi in gw_indices:
        hops_to_gw[gi] = 0
        next_hop[gi] = gi
        path_to_gw[gi] = [gi]
        q.append(gi)

    while q:
        u = q.popleft()
        if hops_to_gw[u] >= max_hops:
            continue
        # сортируем соседей по расстоянию — предпочитаем ближайших
        neighbors = sorted(adj[u], key=lambda x: x[1])
        for v, _dist in neighbors:
            if hops_to_gw[v] == -1:
                hops_to_gw[v] = hops_to_gw[u] + 1
                next_hop[v] = u
                path_to_gw[v] = path_to_gw[u] + [v]
                q.append(v)

    # --- Резервирование: есть ли альтернативный путь (второй родитель) ---
    has_backup_path = [False] * n
    for i in range(n):
        if hops_to_gw[i] <= 0:
            has_backup_path[i] = True
            continue
        # ищем соседа, который тоже ближе к шлюзу, но не next_hop
        for v, _d in adj[i]:
            if v != next_hop[i] and hops_to_gw[v] != -1 and hops_to_gw[v] < hops_to_gw[i]:
                has_backup_path[i] = True
                break

    for i, node in enumerate(all_nodes):
        node["hops_to_gateway"] = hops_to_gw[i]
        node["reachable"] = 0 <= hops_to_gw[i] <= max_hops
        node["next_hop"] = next_hop[i]
        node["has_backup_path"] = has_backup_path[i]
        node["path"] = path_to_gw[i]
        # итоговая высота антенны над землёй уже заложена в elev (effective)

    coverage_ok = all(nd.get("reachable", False) for nd in all_nodes)
    redundancy_ok = all(nd.get("has_backup_path", False) for nd in all_nodes if nd.get("role") == "repeater")

    return {
        "gateways": gateways,
        "repeaters": repeaters,
        "all_nodes": all_nodes,
        "beacon_range_km": round(beacon_range, 2),
        "node_range_km": round(node_range, 2),
        "links": links,
        "coverage_ok": coverage_ok,
        "redundancy_ok": redundancy_ok,
        "max_hops": max_hops,
        "primary_gateway": primary,
        "backup_gateway": backup,
        "city_repeaters_count": len(city_reps)
    }


def analyze_coverage(nodes: List[Dict], beacon_range_km: float, node_range_km: float,
                     n_samples: int = 35) -> Dict:
    """
    Анализ зон покрытия репитеров и шлюзов по сетке точек района.
    Возвращает статистику покрытия маяков БЛА и межсетевых зон.
    """
    if not nodes or beacon_range_km <= 0:
        return {
            "beacon_coverage_pct": 0.0,
            "node_coverage_pct": 0.0,
            "avg_beacon_overlap": 0.0,
            "avg_node_overlap": 0.0,
            "gap_points": [],
            "well_covered_pct": 0.0,
            "n_samples": 0
        }

    lats = np.linspace(BBOX["south"], BBOX["north"], n_samples)
    lons = np.linspace(BBOX["west"], BBOX["east"], n_samples)
    total = 0
    beacon_hit = 0
    node_hit = 0
    beacon_counts = []
    node_counts = []
    gap_points = []
    multi_beacon = 0

    for la in lats:
        for lo in lons:
            total += 1
            b_cnt = 0
            n_cnt = 0
            for nd in nodes:
                d = haversine(la, lo, nd["lat"], nd["lon"])
                if d <= beacon_range_km:
                    b_cnt += 1
                if d <= node_range_km * 0.95:
                    n_cnt += 1
            if b_cnt > 0:
                beacon_hit += 1
            else:
                if len(gap_points) < 40:
                    gap_points.append({"lat": float(la), "lon": float(lo)})
            if n_cnt > 0:
                node_hit += 1
            if b_cnt >= 2:
                multi_beacon += 1
            beacon_counts.append(b_cnt)
            node_counts.append(n_cnt)

    return {
        "beacon_coverage_pct": round(100.0 * beacon_hit / total, 1) if total else 0.0,
        "node_coverage_pct": round(100.0 * node_hit / total, 1) if total else 0.0,
        "avg_beacon_overlap": round(float(np.mean(beacon_counts)), 2) if beacon_counts else 0.0,
        "avg_node_overlap": round(float(np.mean(node_counts)), 2) if node_counts else 0.0,
        "well_covered_pct": round(100.0 * multi_beacon / total, 1) if total else 0.0,
        "gap_points": gap_points,
        "n_samples": total,
        "n_nodes": len(nodes)
    }


# ====================== ЭКСПОРТ ======================

def export_geojson(lora_sites, highs, lows, towers) -> str:
    features = []
    for i, s in enumerate(lora_sites):
        role = s.get("role", "repeater")
        features.append({
            "type": "Feature",
            "properties": {
                "name": s.get("label", f"LoRa_{i+1}"),
                "elev_m": s.get("elev"),
                "role": role,
                "hops_to_gateway": s.get("hops_to_gateway"),
                "reachable": s.get("reachable"),
                "type": "lora_" + role,
                "source": s.get("source", "relief")
            },
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]}
        })
    for i, s in enumerate(highs):
        features.append({
            "type": "Feature",
            "properties": {"name": f"Высокая_точка_{i+1}", "elev_m": s["elev"], "type": "high"},
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]}
        })
    for i, s in enumerate(lows):
        features.append({
            "type": "Feature",
            "properties": {"name": f"Низкая_точка_{i+1}", "elev_m": s["elev"], "type": "low"},
            "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]}
        })
    for t in towers:
        features.append({
            "type": "Feature",
            "properties": {"name": t.get("ref") or "вышка", "operator": t["operator"],
                           "height_m": t.get("height_m"), "type": "tower", "source": t.get("source")},
            "geometry": {"type": "Point", "coordinates": [t["lon"], t["lat"]]}
        })
    return json.dumps({"type": "FeatureCollection", "features": features},
                      ensure_ascii=False, indent=2)


def export_kml(lora_sites, highs, lows, towers, freq_mhz, preset_name) -> str:
    """Чистый KML без внешних библиотек."""
    def placemark(name, lat, lon, desc, style="#ylwPushpin"):
        return f"""
    <Placemark>
      <name>{name}</name>
      <description><![CDATA[{desc}]]></description>
      <styleUrl>{style}</styleUrl>
      <Point><coordinates>{lon},{lat},0</coordinates></Point>
    </Placemark>"""

    parts = ['''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Шкловский район — LoRa MESH + рельеф</name>
    <Style id="lora"><IconStyle><color>ff00ff00</color><scale>1.2</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
    </IconStyle></Style>
    <Style id="high"><IconStyle><color>ff0000ff</color><scale>1.0</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>
    </IconStyle></Style>
    <Style id="low"><IconStyle><color>ffff0000</color><scale>1.0</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>
    </IconStyle></Style>
    <Style id="tower"><IconStyle><color>ff00ffff</color><scale>0.9</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href></Icon>
    </IconStyle></Style>
''']
    for i, s in enumerate(lora_sites):
        desc = f"Высота: {s['elev']:.1f} м<br>Частота: {freq_mhz} МГц<br>Пресет: {preset_name}<br>Источник: {s.get('source','relief')}"
        parts.append(placemark(f"LoRa #{i+1}", s["lat"], s["lon"], desc, "#lora"))
    for i, s in enumerate(highs):
        parts.append(placemark(f"▲ {s['elev']:.0f} м", s["lat"], s["lon"],
                               f"Высокая точка {s['elev']:.1f} м", "#high"))
    for i, s in enumerate(lows):
        parts.append(placemark(f"▼ {s['elev']:.0f} м", s["lat"], s["lon"],
                               f"Низкая точка {s['elev']:.1f} м", "#low"))
    for t in towers:
        h = f"{t['height_m']:.0f} м" if t.get("height_m") else "н/д"
        parts.append(placemark(f"📡 {t['operator']}", t["lat"], t["lon"],
                               f"Оператор: {t['raw_operator']}<br>Высота мачты: {h}<br>Источник: {t.get('source')}", "#tower"))
    parts.append("  </Document>\n</kml>")
    return "".join(parts)


# ====================== UI ======================

st.set_page_config(
    page_title="Шклов — Рельеф + LoRa MESH для БЛА v2",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 Шкловский район — Рельеф + LoRa (MESH) для Remote ID БЛА")
st.caption("Версия 2.0 | Все запрошенные доработки реализованы")

# ----- Боковая панель -----
with st.sidebar:
    st.header("⚙️ Параметры")

    st.subheader("1. Источник рельефа")
    dem_source = st.radio(
        "Источник",
        ["OpenTopoData (SRTM 30м)", "Локальный GeoTIFF", "Локальный KMZ (hillshade/тайлы)"],
        index=0,
        help="KMZ — ваш файл viz.AW3D30_hillshade-color.kmz и аналогичные SuperOverlay"
    )
    uploaded_dem = None
    uploaded_kmz = None
    kmz_max_level = 3  # значение по умолчанию
    if dem_source == "Локальный GeoTIFF":
        uploaded_dem = st.file_uploader("Загрузить GeoTIFF DEM", type=["tif", "tiff", "geotiff"])
        if not HAS_RASTERIO:
            st.error("Установите: pip install rasterio")
    elif dem_source == "Локальный KMZ (hillshade/тайлы)":
        uploaded_kmz = st.file_uploader(
            "Загрузить KMZ (тайловый hillshade / SuperOverlay)",
            type=["kmz", "kml"],
            help="Поддерживается структура с NetworkLink + PNG-тайлами (как ваш AW3D30)"
        )
        kmz_max_level = st.slider("Макс. уровень тайлов KMZ", 0, 4, 3,
                                  help="Выше = детальнее, но больше нагрузка (рекомендуется 2–3)")

    n_grid = st.slider("Плотность сетки (только OpenTopo)", 15, 30, 20)

    st.subheader("2. Фильтрация слоёв карты")
    with st.expander("Включить / выключить слои", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            show_relief = st.checkbox("Высокие/низкие точки", value=True)
            show_settlements = st.checkbox("Населённые пункты", value=True)
            show_towers_osm = st.checkbox("Вышки OSM", value=True)
            show_ocid = st.checkbox("Вышки OpenCelliD", value=False)
        with col_b:
            show_lora = st.checkbox("Сеть LoRa MESH", value=True)
            show_beacon_zones = st.checkbox("Зоны приёма маяков БЛА", value=True)
            show_node_zones = st.checkbox("Зоны покрытия узлов", value=True)
            show_fresnel = st.checkbox("LOS + Френель", value=True)
            show_mesh_links = st.checkbox("Связи MESH", value=True)

    use_tower_height = st.checkbox("Учитывать высоту вышек для LoRa", value=True,
                                   help="Добавляет высоту мачты к высоте рельефа при выборе мест размещения")

    st.subheader("3. OpenCelliD")
    ocid_key = st.text_input("API-ключ OpenCelliD", type="password",
                             help="Получить бесплатно на https://opencellid.org")

    st.subheader("4. Сеть LoRa (MESH) — Remote ID БЛА")
    freq_mhz = st.number_input("Частота (МГц)", 863.0, 870.0, 864.43, 0.01)
    preset_name = st.selectbox("Пресет Meshtastic", list(MESHTASTIC_PRESETS.keys()), index=2,
                               format_func=lambda x: f"{x} — {MESHTASTIC_PRESETS[x]['desc']}")

    with st.expander("Параметры сети MESH (мощности, HOPы, шлюзы)", expanded=True):
        beacon_tx_dbm = st.number_input(
            "Мощность маяка БЛА (dBm)", 0.0, 15.0, 15.0, 0.5,
            help="Максимум 15 dBm по требованиям Remote ID"
        )
        uav_height_m = st.number_input(
            "Высота БЛА над уровнем земли (м)", 5.0, 500.0, 50.0, 5.0,
            help="Высота полёта БЛА. Учитывается при расчёте дальности маяк↔репитер и LOS/Френеля."
        )
        node_tx_dbm = st.number_input(
            "Мощность репитера / шлюза (dBm)", 10.0, 33.0, 30.0, 0.5,
            help="Максимум 2 Вт = 33 dBm. Рекомендуется 27–30 dBm"
        )
        max_hops = st.slider(
            "Макс. количество HOPов до шлюза", 1, 6, 3,
            help="Пакет должен дойти до стационарной станции (шлюза) за N прыжков"
        )
        n_city_repeaters = st.slider(
            "Городских репитеров в Шклове (разгрузка)", 1, 6, 3,
            help="Дополнительные репитеры внутри города для снижения нагрузки"
        )
        st.info("📍 **Основной шлюз** всегда размещается в городе **Шклов**. "
                "Резервный шлюз — также в зоне Шклова. "
                "Оба имеют канал 3G/5G → сервер в Минск.")

    with st.expander("Параметры зоны Френеля и высоты антенн", expanded=False):
        antenna_h_m = st.number_input(
            "Высота антенны сети LoRa MESH над уровнем земли (м AGL)",
            min_value=1.0, max_value=80.0, value=12.0, step=0.5,
            help="Итоговая высота антенны над землёй. "
                 "Если установка на существующей мачте — укажите высоту точки крепления антенны от земли. "
                 "Используется в расчётах LOS, Френеля и эффективной высоты узла."
        )
        clearance_pct = st.slider("Требуемый % свободной 1-й зоны Френеля", 40, 100, 60, 5,
                                  help="Рекомендуется 60 % (стандарт радиорелейных линий)")
        k_factor = st.select_slider("k-фактор атмосферы (ITU-R)",
                                    options=[1.0, 1.2, 4/3, 1.5, 2.0/3, 1.0/3],
                                    value=4/3,
                                    format_func=lambda x: f"{x:.3f}" + (" (стандарт)" if abs(x-4/3)<0.01 else
                                                                        " (субрефракция)" if x < 1 else
                                                                        " (суперрефракция)"))
        n_profile_samples = st.slider("Точек профиля трассы", 40, 150, 80, 10,
                                      help="Больше точек = точнее, но чуть медленнее")

    with st.expander("Параметры расчёта затухания (path loss)", expanded=True):
        path_loss_model = st.selectbox(
            "Модель распространения",
            options=list(PATH_LOSS_MODELS.keys()),
            index=2,  # Rural_Open
            format_func=lambda k: PATH_LOSS_MODELS[k]["name"]
        )
        st.caption(PATH_LOSS_MODELS[path_loss_model]["desc"])
        m = PATH_LOSS_MODELS[path_loss_model]
        st.markdown(f"Показатель **n** = {m['n']}, A0 = {m['A0_offset']} дБ, clutter = {m['clutter_db']} дБ")

        fade_margin_db = st.slider(
            "Запас на замирания (fade margin), дБ", 6.0, 25.0, 12.0, 0.5,
            help="Учитывает многолучевость, движение БЛА, погоду. Рекомендуется 10–15 дБ."
        )
        extra_clutter_db = st.slider(
            "Доп. потери среды (clutter), дБ", 0.0, 20.0, 0.0, 0.5,
            help="Дополнительные потери: лесополосы, рельеф, локальные препятствия сверх модели."
        )
        body_loss_db = st.slider(
            "Потери на корпусе БЛА (body loss), дБ", 0.0, 10.0, 3.0, 0.5,
            help="Экранирование антенны маяка корпусом/аккумулятором БЛА. Типично 2–5 дБ."
        )
        st.markdown("""
**Формула затухания:**
`PL(d) = 20·log₁₀(f_МГц) + 32.44 + 10·n·log₁₀(d_км) + A0 + clutter + extra`

**Link budget:**  
`PL_допустимое = Pt + Gt + Gr − Smin − fade − cable − body`  
Дальность находится из равенства PL(d) = PL_допустимое.
""")

    st.subheader("5. 3D и экспорт")
    show_3d = st.checkbox("3D-визуализация (pydeck)", value=False)
    do_export = st.checkbox("Показать кнопки экспорта", value=True)

    st.markdown("---")
    st.info(f"**{preset_name}**\nSF={MESHTASTIC_PRESETS[preset_name]['sf']}, "
            f"BW={MESHTASTIC_PRESETS[preset_name]['bw']} кГц\n"
            f"Дальность с высотой ≈ {MESHTASTIC_PRESETS[preset_name]['range_km_elev']} км")

# ----- Подготовка данных -----
elev_func = None
lat_grid = lon_grid = elevs = None
highs = lows = []
kmz_overlays = []  # список тайлов из KMZ для отрисовки

if dem_source == "Локальный GeoTIFF" and uploaded_dem is not None:
    raster = load_geotiff_dem(uploaded_dem)
    if raster:
        elev_func = make_elev_func_from_raster(raster)
        st.success(f"GeoTIFF загружен: {raster['shape']}, CRS={raster['crs']}")
        n = n_grid
        lats = np.linspace(BBOX["south"], BBOX["north"], n)
        lons = np.linspace(BBOX["west"], BBOX["east"], n)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        elevs = np.array([[elev_func(la, lo) for lo in lons] for la in lats])
    else:
        # fallback
        lat_grid, lon_grid, elevs = get_elevation_grid_opentopo(n_grid)
        elev_func = make_elev_func_from_grid(lat_grid, lon_grid, elevs)

elif dem_source == "Локальный KMZ (hillshade/тайлы)" and uploaded_kmz is not None:
    with st.spinner("Разбор KMZ и извлечение тайлов..."):
        kmz_overlays = load_kmz_overlays(uploaded_kmz, max_level=kmz_max_level, max_tiles=100)
    if kmz_overlays:
        st.success(f"KMZ загружен: извлечено **{len(kmz_overlays)}** тайлов (уровень ≤ {kmz_max_level})")
        # Числовых высот в hillshade-KMZ нет → используем OpenTopoData для расчётов
        lat_grid, lon_grid, elevs = get_elevation_grid_opentopo(n_grid)
        elev_func = make_elev_func_from_grid(lat_grid, lon_grid, elevs)
        st.info("KMZ содержит визуализацию (hillshade). Для высот, LOS и Френеля используется SRTM 30 м.")
    else:
        st.warning("В KMZ не найдено подходящих PNG-тайлов с координатами. Проверьте структуру файла.")
        lat_grid, lon_grid, elevs = get_elevation_grid_opentopo(n_grid)
        elev_func = make_elev_func_from_grid(lat_grid, lon_grid, elevs)
else:
    # OpenTopoData по умолчанию
    lat_grid, lon_grid, elevs = get_elevation_grid_opentopo(n_grid)
    elev_func = make_elev_func_from_grid(lat_grid, lon_grid, elevs)

if elevs is not None and not np.all(np.isnan(elevs)):
    highs, lows = find_local_extrema(elevs, lat_grid, lon_grid)

settlements = get_settlements() if show_settlements else []
towers_osm = get_cell_towers_osm() if show_towers_osm else []
towers_ocid = get_opencellid_towers(ocid_key) if show_ocid and ocid_key else []
all_towers = towers_osm + towers_ocid

# Кандидаты для LoRa.
# effective_elev = высота антенны над ур. моря = высота земли + высота антенны AGL
# antenna_h_m — итоговая высота антенны сети LoRa MESH над уровнем земли.
candidates = []
for h in highs:
    # На открытой точке: антенна на мачте/опоре высотой antenna_h_m
    eff = h["elev"] + antenna_h_m
    candidates.append({
        **h,
        "elev": eff,
        "effective_elev": eff,
        "ground_elev": h["elev"],
        "antenna_agl": antenna_h_m,
        "source": "relief"
    })

if use_tower_height and all_towers:
    for t in all_towers:
        ground = elev_func(t["lat"], t["lon"]) if elev_func else 0
        if np.isnan(ground):
            ground = 0
        # Если известна высота мачты — антенна на её вершине (или на antenna_h_m, что больше)
        tower_h = t.get("height_m")
        if tower_h and tower_h > 0:
            agl = max(float(tower_h), antenna_h_m)
        else:
            agl = max(30.0, antenna_h_m)  # типичная мачта ≥ 30 м
        eff = ground + agl
        candidates.append({
            "lat": t["lat"], "lon": t["lon"],
            "elev": eff, "effective_elev": eff,
            "type": "tower", "source": t.get("source", "OSM"),
            "tower_height": tower_h, "ground_elev": ground,
            "antenna_agl": agl,
            "operator": t.get("operator")
        })

candidates = sorted(candidates, key=lambda x: -x.get("effective_elev", x.get("elev", 0)))

# Планирование MESH-сети с учётом маяков БЛА, мощности ≤2Вт, HOPов и шлюзов
mesh_plan = {"gateways": [], "repeaters": [], "all_nodes": [],
             "beacon_range_km": 0, "node_range_km": 0, "links": [], "coverage_ok": False}
if show_lora and candidates:
    mesh_plan = suggest_lora_mesh(
        candidates, preset_name, freq_mhz,
        max_hops=max_hops,
        n_gateways=2,                    # всегда основной + резервный
        n_city_repeaters=n_city_repeaters,
        beacon_tx_dbm=beacon_tx_dbm,
        node_tx_dbm=node_tx_dbm,
        antenna_agl_m=antenna_h_m,
        uav_height_m=uav_height_m,
        path_loss_model=path_loss_model,
        extra_clutter_db=extra_clutter_db,
        fade_margin_db=fade_margin_db,
        body_loss_db=body_loss_db
    )

lora_sites = mesh_plan.get("all_nodes", [])  # для совместимости с остальным кодом
gateways = mesh_plan.get("gateways", [])
repeaters = mesh_plan.get("repeaters", [])

# Анализ зон покрытия
coverage = {"beacon_coverage_pct": 0, "node_coverage_pct": 0,
            "avg_beacon_overlap": 0, "avg_node_overlap": 0,
            "well_covered_pct": 0, "gap_points": [], "n_samples": 0}
if show_lora and lora_sites:
    coverage = analyze_coverage(
        lora_sites,
        beacon_range_km=mesh_plan.get("beacon_range_km", 0),
        node_range_km=mesh_plan.get("node_range_km", 0),
        n_samples=32
    )

# LOS / Френель между узлами сети
los_results = []
if show_fresnel and len(lora_sites) >= 2 and elev_func:
    max_r = mesh_plan.get("node_range_km", 15) * 0.98
    for i in range(len(lora_sites)):
        for j in range(i + 1, len(lora_sites)):
            a, b = lora_sites[i], lora_sites[j]
            dist = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if dist > max_r:
                continue
            # elev уже содержит высоту антенны ASL (земля + AGL)
            h1 = a.get("elev", 0)
            h2 = b.get("elev", 0)
            dists, elevs_prof, _, _ = sample_elevation_profile(
                a["lat"], a["lon"], b["lat"], b["lon"], elev_func,
                n_samples=n_profile_samples)
            res = check_los(
                dists, elevs_prof, h1, h2, freq_mhz,
                clearance_pct=clearance_pct,
                k_factor=k_factor,
                n_fresnel=1
            )
            los_results.append({
                "from": i, "to": j, "dist_km": dist,
                "los": res["los"],
                "clearance": res["min_clearance_m"],
                "fresnel_r": res["fresnel_r_mid_m"],
                "blocked_at_km": res["blocked_at_km"],
                "a": a, "b": b,
                "profile": {
                    "dists": dists,
                    "elevs": elevs_prof,
                    "los_line": res["los_line"],
                    "fresnel_lower": res["fresnel_lower"],
                    "fresnel_upper": res["fresnel_upper"],
                    "bulge": res["bulge"],
                    "clearance": res["clearance"]
                }
            })

# Метрики
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Макс. высота", f"{np.nanmax(elevs):.0f} м" if elevs is not None else "—")
with c2:
    st.metric("Нас. пункты", len(settlements))
with c3:
    st.metric("Вышки", len(all_towers))
with c4:
    st.metric("Шлюзы (→Минск)", len(gateways))
with c5:
    st.metric("Репитеры", len(repeaters))
with c6:
    br = mesh_plan.get("beacon_range_km", 0)
    st.metric("Дальность маяка БЛА", f"{br} км")

if show_lora and mesh_plan.get("all_nodes"):
    ok = mesh_plan.get("coverage_ok", False)
    red_ok = mesh_plan.get("redundancy_ok", False)
    city_n = mesh_plan.get("city_repeaters_count", 0)
    msg = (f"Основной шлюз: **Шклов**. Резервный шлюз: **Шклов**. "
           f"Городских репитеров: {city_n}. "
           f"Дальность маяка БЛА ({beacon_tx_dbm} dBm, H={uav_height_m:.0f} м): **{mesh_plan['beacon_range_km']} км**. "
           f"Дальность узлов ({node_tx_dbm} dBm ≤ 2 Вт): **{mesh_plan['node_range_km']} км**. "
           f"Высота антенн AGL: **{antenna_h_m} м**.")
    if ok and red_ok:
        st.success(f"✅ Все узлы достигают шлюза за ≤ {max_hops} HOPов + есть резервные пути. " + msg)
    elif ok:
        st.warning(f"⚠ Все узлы достигают шлюза, но у части нет резервного пути. " + msg)
    else:
        unreachable = sum(1 for n in mesh_plan["all_nodes"] if not n.get("reachable", False))
        st.error(f"❌ {unreachable} узел(ов) НЕ достигают шлюза за {max_hops} HOPов. " + msg)

    # --- Анализ зон покрытия ---
    with st.expander("📊 Анализ зон покрытия репитеров и стационарных станций", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Покрытие маяков БЛА", f"{coverage.get('beacon_coverage_pct', 0)} %")
        with c2:
            st.metric("Покрытие сети узлов", f"{coverage.get('node_coverage_pct', 0)} %")
        with c3:
            st.metric("Зоны с 2+ узлами (маяк)", f"{coverage.get('well_covered_pct', 0)} %")
        with c4:
            st.metric("Среднее перекрытие (маяк)", f"{coverage.get('avg_beacon_overlap', 0)}")

        st.markdown(f"""
**Пояснения:**
- **Покрытие маяков БЛА** — доля территории района, где маяк БЛА (до {beacon_tx_dbm} dBm, высота {uav_height_m:.0f} м)  
  уверенно принимается хотя бы одним репитером/шлюзом (дальность ≈ {mesh_plan.get('beacon_range_km', 0)} км).
- **Покрытие сети узлов** — доля территории в зоне связи между репитерами/шлюзами  
  (дальность ≈ {mesh_plan.get('node_range_km', 0)} км при мощности ≤ 2 Вт).
- **Зоны с 2+ узлами** — резервирование приёма маяка (желательно ≥ 50 %).
- **Красные точки** на карте (слой зон маяков) — пробелы покрытия маяков БЛА.
- Точек анализа: **{coverage.get('n_samples', 0)}**, узлов сети: **{coverage.get('n_nodes', 0)}**.
""")
        bc = coverage.get("beacon_coverage_pct", 0)
        if bc >= 90:
            st.success("Покрытие маяков БЛА отличное (≥ 90 %).")
        elif bc >= 70:
            st.warning("Покрытие маяков БЛА приемлемое (70–90 %). Рекомендуется добавить репитеры в пробелах.")
        else:
            st.error("Покрытие маяков БЛА недостаточное (< 70 %). Увеличьте число репитеров или измените пресет/высоту.")

# ====================== КАРТА ======================
m = folium.Map(location=CENTER, zoom_start=10, tiles=None, control_scale=True,
               attr="")  # убираем дефолтный attribution

# Слои без атрибуции (или минимальной)
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr=" ",  # пустой
    name="OpenStreetMap",
    overlay=False, control=True
).add_to(m)

folium.TileLayer(
    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attr=" ",
    name="OpenTopoMap (рельеф)",
    overlay=False, control=True
).add_to(m)

# Спутник Esri + подписи (два слоя)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr=" ",
    name="Спутник (Esri)",
    overlay=False, control=True
).add_to(m)

# Подписи для спутника (Reference layer)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr=" ",
    name="Подписи (Esri)",
    overlay=True, control=True, show=False
).add_to(m)

# Дополнительно — Hybrid-подобный слой с подписями сразу
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr=" ",
    name="Карта с подписями (Esri)",
    overlay=False, control=True
).add_to(m)

# Скрываем attribution через CSS
m.get_root().html.add_child(folium.Element("""
<style>
.leaflet-control-attribution { display: none !important; }
.leaflet-bottom.leaflet-right { display: none !important; }
</style>
"""))

# Рамка района
folium.Rectangle(
    bounds=[[BBOX["south"], BBOX["west"]], [BBOX["north"], BBOX["east"]]],
    color="#FF4500", weight=2, fill=False,
    tooltip="Шкловский район (границы из KMZ)"
).add_to(m)

# ----- KMZ hillshade тайлы (ImageOverlay) -----
if kmz_overlays:
    kmz_fg = folium.FeatureGroup(name="KMZ рельеф (hillshade)", show=True)
    for i, ov in enumerate(kmz_overlays):
        # base64 PNG для Folium ImageOverlay
        b64 = base64.b64encode(ov["png_bytes"]).decode("ascii")
        img_url = f"data:image/png;base64,{b64}"
        folium.raster_layers.ImageOverlay(
            image=img_url,
            bounds=[[ov["south"], ov["west"]], [ov["north"], ov["east"]]],
            opacity=0.75,
            name=ov.get("name", f"tile_{i}"),
            interactive=False,
            cross_origin=False,
            zindex=1
        ).add_to(kmz_fg)
    kmz_fg.add_to(m)
    st.caption(f"Отображено тайлов KMZ: {len(kmz_overlays)}")

# Высокие / низкие точки
if show_relief and highs:
    fg = folium.FeatureGroup(name="Высокие точки", show=True)
    for i, p in enumerate(highs):
        folium.CircleMarker([p["lat"], p["lon"]], radius=7, color="#8B0000",
                            fill=True, fill_color="#FF4500", fill_opacity=0.9,
                            popup=f"<b>Высокая #{i+1}</b><br>{p['elev']:.1f} м",
                            tooltip=f"▲ {p['elev']:.0f} м").add_to(fg)
        off = [p["lat"]+0.007, p["lon"]+0.010]
        folium.PolyLine([[p["lat"], p["lon"]], off], color="#8B0000", weight=1.5,
                        opacity=0.7, dash_array="4").add_to(fg)
        folium.Marker(off, icon=folium.DivIcon(
            html=f'<div style="font-size:11px;color:#8B0000;font-weight:bold;">▲ {p["elev"]:.0f} м</div>'
        )).add_to(fg)
    fg.add_to(m)

if show_relief and lows:
    fg = folium.FeatureGroup(name="Низкие точки", show=True)
    for i, p in enumerate(lows):
        folium.CircleMarker([p["lat"], p["lon"]], radius=6, color="#00008B",
                            fill=True, fill_color="#1E90FF", fill_opacity=0.85,
                            popup=f"<b>Низкая #{i+1}</b><br>{p['elev']:.1f} м",
                            tooltip=f"▼ {p['elev']:.0f} м").add_to(fg)
        off = [p["lat"]-0.007, p["lon"]-0.009]
        folium.PolyLine([[p["lat"], p["lon"]], off], color="#00008B", weight=1.5,
                        opacity=0.7, dash_array="4").add_to(fg)
        folium.Marker(off, icon=folium.DivIcon(
            html=f'<div style="font-size:11px;color:#00008B;font-weight:bold;">▼ {p["elev"]:.0f} м</div>'
        )).add_to(fg)
    fg.add_to(m)

# Населённые пункты
if show_settlements and settlements:
    try:
        fg = folium.FeatureGroup(name="Населённые пункты", show=True)
        colors = {"city": "#FF0000", "town": "#FF8C00", "village": "#32CD32",
                  "hamlet": "#90EE90", "suburb": "#FFD700"}
        for s in settlements:
            c = colors.get(s["place"], "#808080")
            r = {"city": 10, "town": 8, "village": 6}.get(s["place"], 4)
            folium.CircleMarker(
                [s["lat"], s["lon"]], radius=r, color=c, fill=True,
                fill_opacity=0.75,
                tooltip=str(s.get("name", "")),
                popup=f"<b>{s.get('name','')}</b><br>{s.get('place','')}"
            ).add_to(fg)
        fg.add_to(m)
    except Exception as e:
        st.warning(f"Слой населённых пунктов: {e}")

# Вышки
if all_towers:
    try:
        fg = folium.FeatureGroup(name="Вышки связи", show=True)
        cluster = MarkerCluster().add_to(fg)
        for t in all_towers:
            h_val = t.get("height_m")
            h_str = f"{h_val:.0f} м" if h_val else "н/д"
            color = "purple" if t.get("source") == "OpenCelliD" else "gray"
            folium.Marker(
                [t["lat"], t["lon"]],
                icon=folium.Icon(color=color, icon="info-sign"),
                popup=(f"<b>Вышка связи</b><br>"
                       f"Оператор: {t.get('raw_operator')}<br>"
                       f"Высота мачты: <b>{h_str}</b><br>"
                       f"Источник: {t.get('source')}<br>"
                       f"Ref: {t.get('ref') or '—'}"),
                tooltip=f"{t.get('operator')} ({h_str})"
            ).add_to(cluster)
        fg.add_to(m)
    except Exception as e:
        st.warning(f"Слой вышек: {e}")

# ===== Сеть LoRa MESH (шлюзы + репитеры + зоны приёма маяков) =====
if show_lora and lora_sites:
    beacon_rng = mesh_plan.get("beacon_range_km", 3)
    node_rng = mesh_plan.get("node_range_km", 10)

    # Зоны приёма маяков БЛА
    if show_beacon_zones:
        fg_beacon = folium.FeatureGroup(name="Зоны приёма маяков БЛА", show=True)
        for site in lora_sites:
            folium.Circle(
                [site["lat"], site["lon"]],
                radius=max(beacon_rng, 0.3) * 1000,
                color="#FFD700", weight=1,
                fill=True, fill_color="#FFD700", fill_opacity=0.06,
                tooltip=f"Приём маяка БЛА ≤ {beacon_rng} км"
            ).add_to(fg_beacon)
        for gp in coverage.get("gap_points", [])[:25]:
            folium.CircleMarker(
                [gp["lat"], gp["lon"]], radius=3,
                color="#FF0000", fill=True, fill_color="#FF0000", fill_opacity=0.5,
                tooltip="Пробел покрытия маяков БЛА"
            ).add_to(fg_beacon)
        fg_beacon.add_to(m)

    # Зоны покрытия узлов
    if show_node_zones:
        fg_nodez = folium.FeatureGroup(name="Зоны покрытия узлов", show=True)
        for site in lora_sites:
            folium.Circle(
                [site["lat"], site["lon"]],
                radius=max(node_rng, 0.5) * 1000 * 0.9,
                color="#00BFFF", weight=1,
                fill=True, fill_color="#00BFFF", fill_opacity=0.04,
                tooltip=f"Связь узлов ≤ {node_rng} км"
            ).add_to(fg_nodez)
        fg_nodez.add_to(m)

    # Шлюзы
    fg_gw = folium.FeatureGroup(name="Шлюзы Шклов (3G/5G)", show=True)
    for site in gateways:
        is_primary = site.get("gateway_type") == "primary"
        color = "darkblue" if is_primary else "blue"
        icon_name = "home" if is_primary else "upload"
        try:
            agl = site.get("antenna_agl", antenna_h_m)
            folium.Marker(
                [site["lat"], site["lon"]],
                icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
                popup=(f"<b>{site.get('label', 'Шлюз')}</b><br>"
                       f"{'Основной' if is_primary else 'Резервный'} шлюз в Шклове<br>"
                       f"Высота антенны ASL: <b>{site.get('elev', 0):.1f} м</b><br>"
                       f"Высота антенны AGL: <b>{agl:.1f} м</b><br>"
                       f"Мощность: ≤ {node_tx_dbm} dBm (≤ 2 Вт)<br>"
                       f"Канал: 3G/5G модем → сервер в Минск"),
                tooltip=site.get("label", "Шлюз Шклов")
            ).add_to(fg_gw)
        except Exception as e:
            st.warning(f"Ошибка маркера шлюза: {e}")
    fg_gw.add_to(m)

    # Репитеры
    fg_rep = folium.FeatureGroup(name="Репитеры MESH", show=True)
    for i, site in enumerate(repeaters):
        reachable = site.get("reachable", False)
        has_bak = site.get("has_backup_path", False)
        color = "green" if (reachable and has_bak) else ("orange" if reachable else "red")
        hops = site.get("hops_to_gateway", -1)
        zone = site.get("zone", "rural")
        folium.Marker(
            [site["lat"], site["lon"]],
            icon=folium.Icon(color=color, icon="wifi", prefix="fa"),
            popup=(f"<b>{site.get('label', 'Репитер')}</b><br>"
                   f"Зона: {'город Шклов' if zone=='city' else 'район'}<br>"
                   f"Высота антенны ASL: <b>{site.get('elev', 0):.1f} м</b><br>"
                   f"Высота AGL: <b>{site.get('antenna_agl', antenna_h_m):.1f} м</b><br>"
                   f"HOPов до шлюза: <b>{hops if hops >= 0 else 'нет'}</b><br>"
                   f"{'Достижим' if reachable else 'Нет пути к шлюзу'} | "
                   f"{'есть резервный путь' if has_bak else 'нет резерва'}"),
            tooltip=f"{site.get('label', 'Репитер')} (HOP={hops})"
        ).add_to(fg_rep)
    fg_rep.add_to(m)

    # Линии связи
    if show_mesh_links or show_fresnel:
        fg_links = folium.FeatureGroup(name="Связи MESH + LOS", show=True)
        for res in los_results:
            color = "#00CED1" if res["los"] else "#FF4500"
            weight = 3 if res["los"] else 2
            dash = "4" if res["los"] else "8"
            status = "LOS есть" if res["los"] else f"LOS нет (зазор {res['clearance']:.1f} м)"
            blocked = f"<br>Блокировка на: {res['blocked_at_km']:.1f} км" if res.get("blocked_at_km") else ""
            folium.PolyLine(
                [[res["a"]["lat"], res["a"]["lon"]], [res["b"]["lat"], res["b"]["lon"]]],
                color=color, weight=weight, opacity=0.75, dash_array=dash,
                popup=(f"<b>{status}</b><br>Расстояние: {res['dist_km']:.1f} км<br>"
                       f"R зоны Френеля: {res['fresnel_r']:.1f} м{blocked}")
            ).add_to(fg_links)
        fg_links.add_to(m)

Fullscreen().add_to(m)
MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)
MiniMap(toggle_display=True).add_to(m)
Draw(export=True).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=None, height=680, returned_objects=[])

# ====================== LOS + ПРОФИЛИ ФРЕНЕЛЯ ======================
if show_fresnel and los_results:
    with st.expander("📡 Результаты проверки LOS и зон Френеля (оптимизированный расчёт)", expanded=True):
        st.markdown(f"""
**Параметры расчёта:** частота = **{freq_mhz} МГц**,  
требуемый зазор = **{clearance_pct} %** 1-й зоны Френеля,  
k-фактор = **{k_factor:.3f}**, высота антенны = **{antenna_h_m} м**,  
точек профиля = **{n_profile_samples}**.
""")
        rows = []
        for r in los_results:
            rows.append({
                "От": f"#{r['from']+1}",
                "До": f"#{r['to']+1}",
                "Расстояние, км": round(r["dist_km"], 2),
                "LOS": "✅ Да" if r["los"] else "❌ Нет",
                "Мин. зазор, м": round(r["clearance"], 1),
                "Блокировка на, км": round(r["blocked_at_km"], 2) if r["blocked_at_km"] else "—",
                "R Френеля (сер.), м": round(r["fresnel_r"], 1)
            })
        st.dataframe(rows, use_container_width=True)

        # Выбор трассы для детального профиля
        st.markdown("#### Профиль трассы (рельеф + LOS + зона Френеля)")
        link_labels = [f"#{r['from']+1} → #{r['to']+1}  ({r['dist_km']:.1f} км)  "
                       f"{'✅ LOS' if r['los'] else '❌ НЕТ LOS'}" for r in los_results]
        sel_idx = st.selectbox("Выберите радиолинию", range(len(los_results)),
                               format_func=lambda i: link_labels[i])

        if sel_idx is not None:
            r = los_results[sel_idx]
            p = r["profile"]
            try:
                import matplotlib
                matplotlib.use("Agg")  # безопасный backend для Streamlit
                import matplotlib.pyplot as plt
            except ImportError:
                st.error("Пакет matplotlib не установлен. Выполните в терминале:\n\n```\npip install matplotlib\n```")
                st.stop()

            fig, ax = plt.subplots(figsize=(11, 4.5))
            d = p["dists"]
            # Рельеф
            ax.fill_between(d, p["elevs"], alpha=0.45, color="#8B4513", label="Рельеф")
            ax.plot(d, p["elevs"], color="#5D3A1A", linewidth=1.2)
            # Линия LOS
            ax.plot(d, p["los_line"], "b-", linewidth=1.8, label="Линия LOS (прямая видимость)")
            # Зона Френеля
            ax.fill_between(d, p["fresnel_lower"], p["fresnel_upper"],
                            alpha=0.25, color="#00BFFF", label=f"1-я зона Френеля ({clearance_pct} %)")
            ax.plot(d, p["fresnel_lower"], color="#0088CC", linewidth=1.0, linestyle="--",
                    label="Нижняя граница зоны (критическая)")
            # Антенны
            ax.plot(d[0], p["los_line"][0], "g^", markersize=12, label="Антенна A")
            ax.plot(d[-1], p["los_line"][-1], "r^", markersize=12, label="Антенна B")

            # Отметить точку минимального зазора
            if r["clearance"] < 0 and r["blocked_at_km"] is not None:
                wi = int(np.argmin(np.abs(d - r["blocked_at_km"])))
                ax.axvline(r["blocked_at_km"], color="red", linestyle=":", alpha=0.7)
                ax.plot(d[wi], p["elevs"][wi], "rx", markersize=10, mew=2,
                        label=f"Блокировка ({r['clearance']:.1f} м)")

            ax.set_xlabel("Расстояние от точки A, км")
            ax.set_ylabel("Высота, м (над ур. моря)")
            ax.set_title(f"Профиль трассы #{r['from']+1} → #{r['to']+1}  |  "
                         f"{'LOS есть' if r['los'] else 'LOS отсутствует'}  |  "
                         f"Rф = {r['fresnel_r']:.1f} м")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, d[-1])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Краткая физика
            st.caption(
                f"Формула радиуса зоны Френеля:  "
                f"rₙ = √(n · λ · d₁ · d₂ / (d₁ + d₂)),  λ = 299.79 / f(МГц) м.  "
                f"Выпуклость Земли: h = d₁·d₂ / (2·k·Rₑ), k = {k_factor:.3f}."
            )

# ====================== 3D ======================
if show_3d:
    st.subheader("🌐 3D-визуализация (pydeck)")
    if not HAS_PYDECK:
        st.warning("Установите: pip install pydeck")
    else:
        # Точки для 3D
        data_3d = []
        for s in lora_sites:
            data_3d.append({"lat": s["lat"], "lon": s["lon"], "elev": s["elev"],
                            "name": "LoRa", "color": [0, 255, 100]})
        for h in highs[:10]:
            data_3d.append({"lat": h["lat"], "lon": h["lon"], "elev": h["elev"],
                            "name": "High", "color": [255, 80, 0]})
        for t in all_towers[:50]:
            data_3d.append({"lat": t["lat"], "lon": t["lon"],
                            "elev": (t.get("height_m") or 30) + 50,
                            "name": "Tower", "color": [150, 100, 255]})

        layer = pdk.Layer(
            "ColumnLayer",
            data=data_3d,
            get_position=["lon", "lat"],
            get_elevation="elev",
            elevation_scale=3,
            radius=400,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True
        )
        view = pdk.ViewState(latitude=CENTER[0], longitude=CENTER[1],
                             zoom=9, pitch=45, bearing=20)
        deck = pdk.Deck(layers=[layer], initial_view_state=view,
                        tooltip={"text": "{name}\nВысота: {elev} м"},
                        map_style="mapbox://styles/mapbox/satellite-v9")
        st.pydeck_chart(deck)

# ====================== ЭКСПОРТ ======================
if do_export:
    st.subheader("📤 Экспорт")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        geojson_str = export_geojson(lora_sites, highs, lows, all_towers)
        st.download_button("Скачать GeoJSON", geojson_str,
                           file_name="shklov_lora_mesh.geojson",
                           mime="application/geo+json")
    with col_b:
        kml_str = export_kml(lora_sites, highs, lows, all_towers, freq_mhz, preset_name)
        st.download_button("Скачать KML", kml_str,
                           file_name="shklov_lora_mesh.kml",
                           mime="application/vnd.google-earth.kml+xml")
    with col_c:
        if lora_sites:
            import pandas as pd
            df = pd.DataFrame([{
                "№": i+1,
                "role": s.get("role", "repeater"),
                "label": s.get("label", ""),
                "lat": s["lat"],
                "lon": s["lon"],
                "elev_m": round(s.get("elev", 0), 1),
                "hops_to_gateway": s.get("hops_to_gateway"),
                "reachable": s.get("reachable"),
                "source": s.get("source"),
                "freq_mhz": freq_mhz,
                "preset": preset_name,
                "beacon_tx_dbm": beacon_tx_dbm,
                "node_tx_dbm": node_tx_dbm,
                "max_hops": max_hops
            } for i, s in enumerate(lora_sites)])
            st.download_button("Скачать CSV (узлы LoRa MESH)",
                               df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"lora_mesh_{preset_name}_hops{max_hops}.csv",
                               mime="text/csv")

with st.expander("ℹ️ Что нового в версии 2.0"):
    st.markdown("""
1. **Атрибуты карты скрыты** (CSS + пустой attr).
2. **Высота вышек учитывается** — при выборе мест для LoRa эффективная высота = высота рельефа + высота мачты (если известна, иначе 30 м).
3. **Спутник Esri** — добавлен отдельный слой «Подписи (Esri)» и слой «Карта с подписями». Включайте «Подписи (Esri)» вместе со «Спутник (Esri)».
4. **Загрузка локального GeoTIFF DEM** — переключатель в боковой панели.
5. **LOS + зоны Френеля** — между всеми парами репитеров в пределах дальности пресета.
6. **OpenCelliD** — поле для API-ключа + отображение вышек.
7. **3D (pydeck)** — ColumnLayer по высотам.
8. **Экспорт** — GeoJSON, KML, CSV.
""")

st.caption("Прототип v2.0 | Шкловский район | Remote ID БЛА | LoRa 868 МГц")
