"""
module1.py

模块一：RINEX NAV 导航文件解析 + 模拟伪距生成模块。

功能：
1. 解析 RINEX NAV 导航文件，提取北斗三号卫星广播星历参数；
2. 基于星历计算卫星 ECEF 坐标；
3. 计算几何距离 rho 和卫星高度角；
4. 按误差模型生成模拟伪距观测值；
5. 数据预处理：健康筛选、高度角筛选；
6. 输出解析调试 CSV 和模拟伪距 CSV。

约束：
- 不读取 OBS 观测文件；
- 不使用真实观测伪距；
- 所有距离单位为米，时间单位为秒；
- 支持随机种子 seed，保证结果可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import csv
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 常量定义（从 module2 迁移）
# ============================================================================
MU = 3.986004418e14  # 地球引力常数，单位 m^3/s^2
OMEGA_E = 7.2921150e-5  # 地球自转角速度，单位 rad/s
C = 299_792_458.0  # 光速，单位 m/s
BDS_CNAV_A_REF = 27_906_100.0  # BDS-3 CNAV DeltaA reference semi-major axis, m

BDT_EPOCH = datetime(2006, 1, 1, 0, 0, 0)  # 北斗时起点近似
SECONDS_IN_WEEK = 604_800.0
HALF_WEEK = 302_400.0
RELATIVITY_F = -2.0 * math.sqrt(MU) / (C * C)

PROJECT_TIME_SYSTEM = "BDT"
# 本项目不读取 OBS 文件，所有仿真历元 epoch_time 与 RINEX CNAV 中的 toc/toe
# 统一按 BDT 时间系统理解。
# 当前不做 UTC/GPST/BDT 转换。
# 如果后续接入真实 OBS 文件，需要单独实现时间系统转换模块。

# WGS84 椭球常数（从 module3 迁移）
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# ============================================================================
# RINEX 解析辅助
# ============================================================================
_FLOAT_PATTERN = re.compile(
    r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[DEde][+-]?\d+)?"
)


@dataclass
class BroadcastEphemeris:
    """北斗广播星历数据结构。

    单位约定：
    - 时间字段使用 datetime 或秒；
    - 角度字段使用弧度；
    - 距离字段使用米；
    - sqrt_a 为轨道长半轴平方根，单位为 sqrt(m)。
    """

    sat_id: str
    toc: datetime
    af0: float  # 卫星钟差多项式常数项，单位 s
    af1: float  # 卫星钟差多项式一次项，单位 s/s
    af2: float  # 卫星钟差多项式二次项，单位 s/s^2

    aode: float  # 星历数据龄期
    crs: float  # 轨道半径正弦调和改正项，单位 m
    delta_n: float  # 平均角速度改正量，单位 rad/s
    m0: float  # 参考时刻平近点角，单位 rad

    cuc: float  # 纬度幅角余弦调和改正项，单位 rad
    eccentricity: float  # 轨道偏心率
    cus: float  # 纬度幅角正弦调和改正项，单位 rad
    sqrt_a: float  # 轨道长半轴平方根，单位 sqrt(m)

    toe: float  # 星历参考时刻，BDS 周内秒
    cic: float  # 轨道倾角余弦调和改正项，单位 rad
    omega0: float  # 参考时刻升交点赤经，单位 rad
    cis: float  # 轨道倾角正弦调和改正项，单位 rad

    i0: float  # 参考时刻轨道倾角，单位 rad
    crc: float  # 轨道半径余弦调和改正项，单位 m
    omega: float  # 近地点幅角，单位 rad
    omega_dot: float  # 升交点赤经变化率，单位 rad/s

    idot: float  # 轨道倾角变化率，单位 rad/s
    data_source: float
    week: float  # BDS 周数
    accuracy: float

    health: float  # 卫星健康状态，通常 0 表示健康
    tgd1: float  # BDS TGD1 群延迟，单位 s
    tgd2: float  # BDS TGD2 群延迟，单位 s
    transmission_time: float  # 电文发射时刻，BDS 周内秒
    a_dot: float = 0.0  # CNAV semi-major axis rate, m/s
    fit_interval: float = 0.0
    parse_status: str = "ok"  # 解析状态：ok / partial / failed


@dataclass
class NavParseInfo:
    """导航文件解析过程中的统计信息。"""

    nav_file_path: str = ""
    rinex_version: str = "未知"
    skipped_non_bds_records: int = 0
    skipped_bds2_records: int = 0
    failed_records: int = 0
    incomplete_records: int = 0
    error_messages: List[str] = field(default_factory=list)


def _rinex_float(text: str) -> float:
    """将 RINEX 数字字符串转为 float，兼容 D/E 指数。"""
    return float(text.replace("D", "E").replace("d", "E"))


def _extract_floats(line: str) -> List[float]:
    """从一行 RINEX 文本中提取所有浮点数。"""
    return [_rinex_float(token) for token in _FLOAT_PATTERN.findall(line)]


def parse_nav_float(value: str) -> float:
    """解析 RINEX 导航文件中的浮点数，兼容 D/E 科学计数法。"""
    cleaned = value.strip().replace("D", "E").replace("d", "e")
    if not cleaned:
        return 0.0
    return float(cleaned)


def parse_nav_4_fields(line: str, start: int = 4) -> List[float]:
    """按固定宽度解析一行中的 4 个浮点数字段，每字段 19 字符。

    RINEX 3.x 导航文件数据行格式（默认从索引 4 开始）：
    - 第 1 个字段：start + 0  ~ start + 18
    - 第 2 个字段：start + 19 ~ start + 37
    - 第 3 个字段：start + 38 ~ start + 56
    - 第 4 个字段：start + 57 ~ start + 75

    对于记录首行（钟差参数），可从 start=23 开始解析 af0/af1/af2。
    """
    fields = []
    for offset in (0, 19, 38, 57):
        text = line[start + offset : start + offset + 19].strip()
        if text:
            fields.append(parse_nav_float(text))
        else:
            fields.append(0.0)
    return fields


def _parse_epoch_from_first_line(line: str) -> datetime:
    """解析 RINEX 3.x 导航记录首行中的星历钟参考时间 toc。

    该函数解析 RINEX CNAV 记录首行中的 toc。
    本项目将 toc 作为 BDT 时间系统下的无时区 datetime 使用。
    不在此处执行 UTC/GPST/BDT 转换。
    """
    year = int(line[4:8])
    month = int(line[9:11])
    day = int(line[12:14])
    hour = int(line[15:17])
    minute = int(line[18:20])
    second_text = line[21:23].strip() or "0"
    second = int(float(second_text))
    return datetime(year, month, day, hour, minute, second)


def ensure_bdt_naive_datetime(value: datetime) -> datetime:
    """确保时间为无时区 datetime，并在本项目中统一按 BDT 理解。"""
    if value.tzinfo is not None:
        raise ValueError("本项目当前要求使用无时区 datetime，并统一按 BDT 时间系统理解")
    return value


def _build_ephemeris(record_lines: List[str]) -> Optional[BroadcastEphemeris]:
    """将 8 行 RINEX 3.x BDS 导航记录转换为 BroadcastEphemeris。

    兼容普通 RINEX NAV 文件解析，使用正则提取浮点数。
    """
    if len(record_lines) < 8:
        return None

    first_line = record_lines[0]
    sat_id = first_line[:3].strip()
    if not sat_id.startswith("C"):
        return None
    # 只保留北斗三号（PRN >= 19）
    try:
        prn = int(sat_id[1:])
        if prn < 19:
            return None
    except ValueError:
        return None

    toc = _parse_epoch_from_first_line(first_line)
    clock_values = _extract_floats(first_line[23:])
    if len(clock_values) < 3:
        raise ValueError(f"{sat_id} {toc} 的卫星钟差字段不完整")

    values: List[float] = []
    for line in record_lines[1:8]:
        values.extend(_extract_floats(line[3:]))
    if len(values) < 26:
        raise ValueError(f"{sat_id} {toc} 的轨道参数字段不完整，仅解析到 {len(values)} 个")
    values = (values + [0.0] * 26)[:26]

    return BroadcastEphemeris(
        sat_id=sat_id,
        toc=toc,
        af0=clock_values[0],
        af1=clock_values[1],
        af2=clock_values[2],
        aode=values[0],
        crs=values[1],
        delta_n=values[2],
        m0=values[3],
        cuc=values[4],
        eccentricity=values[5],
        cus=values[6],
        sqrt_a=values[7],
        toe=values[8],
        cic=values[9],
        omega0=values[10],
        cis=values[11],
        i0=values[12],
        crc=values[13],
        omega=values[14],
        omega_dot=values[15],
        idot=values[16],
        data_source=values[17],
        week=values[18],
        accuracy=values[20],
        health=values[21],
        tgd1=values[22],
        tgd2=values[23],
        transmission_time=values[24],
        fit_interval=values[25],
        parse_status="ok",
    )


def _build_ephemeris_cnav(record_lines: List[str]) -> Optional[BroadcastEphemeris]:
    """将 8 行 BDS-3 CNAV 导航记录转换为 BroadcastEphemeris，使用固定宽度解析。"""
    if len(record_lines) < 8:
        return None

    first_line = record_lines[0]
    sat_id = first_line[:3].strip()
    if not sat_id.startswith("C"):
        return None

    try:
        prn = int(sat_id[1:])
        if prn < 19:
            return None
    except ValueError:
        return None

    toc = _parse_epoch_from_first_line(first_line)

    # 第 1 行：af0, af1, af2（从索引 23 开始按 19 字符宽度解析）
    clock_fields = parse_nav_4_fields(first_line, start=23)
    af0, af1, af2 = clock_fields[0], clock_fields[1], clock_fields[2]

    # 后 7 行固定宽度解析
    # 第 2 行：aode, crs, delta_n, m0
    line2 = parse_nav_4_fields(record_lines[1])
    aode, crs, delta_n, m0 = line2[0], line2[1], line2[2], line2[3]

    # 第 3 行：cuc, e, cus, DeltaA
    line3 = parse_nav_4_fields(record_lines[2])
    cuc, e, cus, delta_a = line3[0], line3[1], line3[2], line3[3]
    sqrtA = math.sqrt(BDS_CNAV_A_REF + delta_a)

    # 第 4 行：toe, cic, omega0, cis
    line4 = parse_nav_4_fields(record_lines[3])
    toe, cic, omega0, cis = line4[0], line4[1], line4[2], line4[3]

    # 第 5 行：i0, crc, omega, omega_dot
    line5 = parse_nav_4_fields(record_lines[4])
    i0, crc, omega, omega_dot = line5[0], line5[1], line5[2], line5[3]

    # 第 6 行：idot, spare, bdt_week, A_dot
    line6 = parse_nav_4_fields(record_lines[5])
    idot, data_source, bdt_week, a_dot = line6[0], line6[1], line6[2], line6[3]

    # 第 7 行：第 2 个字段作为 health
    line7 = parse_nav_4_fields(record_lines[6])
    accuracy = line7[0] if len(line7) > 0 else 0.0
    health = line7[1] if len(line7) > 1 else 0.0
    tgd1 = line7[2] if len(line7) > 2 else 0.0
    tgd2 = line7[3] if len(line7) > 3 else 0.0

    # 第 8 行：扩展字段
    line8 = parse_nav_4_fields(record_lines[7])
    transmission_time = line8[0] if len(line8) > 0 else 0.0
    fit_interval = line8[3] if len(line8) > 3 else 0.0

    # 判断解析状态
    parse_status = "ok"
    try:
        if sqrtA <= 0:
            parse_status = "partial_sqrtA_invalid"
        elif toe <= 0:
            parse_status = "partial_toe_invalid"
        elif e < 0:
            parse_status = "partial_e_invalid"
    except Exception:
        parse_status = "partial"

    return BroadcastEphemeris(
        sat_id=sat_id,
        toc=toc,
        af0=af0,
        af1=af1,
        af2=af2,
        aode=aode,
        crs=crs,
        delta_n=delta_n,
        m0=m0,
        cuc=cuc,
        eccentricity=e,
        cus=cus,
        sqrt_a=sqrtA,
        toe=toe,
        cic=cic,
        omega0=omega0,
        cis=cis,
        i0=i0,
        crc=crc,
        omega=omega,
        omega_dot=omega_dot,
        idot=idot,
        data_source=data_source,
        week=bdt_week,
        accuracy=accuracy,
        health=health,
        tgd1=tgd1,
        tgd2=tgd2,
        transmission_time=transmission_time,
        a_dot=a_dot,
        fit_interval=fit_interval,
        parse_status=parse_status,
    )


def parse_rinex_nav_with_info(
    nav_file: str | Path,
) -> Tuple[Dict[str, List[BroadcastEphemeris]], NavParseInfo]:
    """解析 RINEX NAV 文件，并返回北斗三号星历和解析统计信息。

    兼容旧函数名，内部直接调用 BDS-3 CNAV 解析逻辑。
    """
    return parse_bds_cnav_file(nav_file)


def parse_rinex_nav(nav_file: str | Path) -> Dict[str, List[BroadcastEphemeris]]:
    """解析 RINEX NAV 文件，只返回北斗三号广播星历字典。"""
    nav_data, _ = parse_rinex_nav_with_info(nav_file)
    return nav_data


def group_ephemeris_by_sat(
    records: List[BroadcastEphemeris],
) -> Dict[str, List[BroadcastEphemeris]]:
    """将星历记录按卫星编号分组。"""
    nav_data: Dict[str, List[BroadcastEphemeris]] = {}
    for eph in records:
        nav_data.setdefault(eph.sat_id, []).append(eph)
    return nav_data


def parse_bds_cnav_file(
    nav_path: str | Path,
) -> Tuple[Dict[str, List[BroadcastEphemeris]], NavParseInfo]:
    """解析 BDS-3 CNAV1/CNAV2 广播星历文件（如 *.26b_cnav）。

    使用固定宽度解析 8 行一组的北斗三号星历记录。
    """
    path = Path(nav_path)
    info = NavParseInfo(nav_file_path=str(path))
    if not path.exists():
        raise FileNotFoundError(f"NAV 文件不存在: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if lines:
        version_text = lines[0][:20].strip()
        info.rinex_version = version_text or "未知"

    start_index = None
    for index, line in enumerate(lines):
        if "END OF HEADER" in line:
            start_index = index + 1
            break
    if start_index is None:
        raise ValueError("未在 RINEX NAV 文件中找到 END OF HEADER")

    records: List[BroadcastEphemeris] = []
    i = start_index
    while i < len(lines):
        line = lines[i]
        sat_id = line[:3].strip()
        if not sat_id:
            i += 1
            continue

        if len(lines[i : i + 8]) < 8:
            info.incomplete_records += 1
            break

        if not sat_id.startswith("C"):
            info.skipped_non_bds_records += 1
            i += 8
            continue

        try:
            prn = int(sat_id[1:])
            if prn < 19:
                info.skipped_bds2_records += 1
                i += 8
                continue
        except ValueError:
            info.skipped_non_bds_records += 1
            i += 8
            continue

        record_lines = lines[i : i + 8]
        try:
            eph = _build_ephemeris_cnav(record_lines)
            if eph is not None:
                records.append(eph)
        except Exception as exc:
            info.failed_records += 1
            info.error_messages.append(f"{sat_id} 第 {i + 1} 行附近解析失败: {exc}")
        i += 8

    nav_data = group_ephemeris_by_sat(records)
    for eph_list in nav_data.values():
        eph_list.sort(key=lambda item: item.toc)
    return nav_data, info


def parse_nav_file(
    nav_path: str | Path,
) -> Tuple[Dict[str, List[BroadcastEphemeris]], NavParseInfo]:
    """解析导航文件。

    默认所有输入导航文件都按 BDS-3 CNAV 格式解析，直接调用 parse_bds_cnav_file。
    不再根据文件后缀或文件头判断普通 NAV 与 CNAV。
    """
    return parse_bds_cnav_file(nav_path)


def select_ephemeris_for_epoch(
    records: List[BroadcastEphemeris],
    epoch: datetime,
) -> Optional[BroadcastEphemeris]:
    """从同一颗卫星的多条星历中选择与 epoch 时间最接近的记录。

    如果同一时刻出现多条记录，优先选择字段完整（parse_status == 'ok'）
    且 health == 0 的记录。
    """
    epoch = ensure_bdt_naive_datetime(epoch)
    if not records:
        return None

    from collections import defaultdict

    # 按 toc 分组，处理同一时刻多条 CNAV 记录的情况
    by_toc: Dict[datetime, List[BroadcastEphemeris]] = defaultdict(list)
    for eph in records:
        by_toc[eph.toc].append(eph)

    # 找到与 epoch 时间差最小的 toc
    best_toc = min(by_toc.keys(), key=lambda toc: abs((epoch - toc).total_seconds()))
    candidates = by_toc[best_toc]

    if len(candidates) == 1:
        return candidates[0]

    # 多记录时优先：health==0 > parse_status=='ok' > 其他
    def _priority(eph: BroadcastEphemeris) -> tuple:
        is_healthy = int(round(eph.health)) == 0
        is_complete = getattr(eph, "parse_status", "ok") == "ok"
        return (is_healthy, is_complete)

    candidates.sort(key=_priority, reverse=True)
    return candidates[0]


def select_ephemeris(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    sat_id: str,
    epoch_time: datetime,
    healthy_only: bool = False,
) -> Optional[BroadcastEphemeris]:
    """为指定卫星和历元选择 toc 时间最接近的星历。"""
    epoch_time = ensure_bdt_naive_datetime(epoch_time)
    eph_list = nav_data.get(sat_id)
    if not eph_list:
        return None
    if healthy_only:
        eph_list = [eph for eph in eph_list if int(round(eph.health)) == 0]
        if not eph_list:
            return None
    return select_ephemeris_for_epoch(eph_list, epoch_time)


# ============================================================================
# 卫星位置计算（从 module2 迁移）
# ============================================================================
def _bds_seconds_of_week(epoch_time: datetime) -> float:
    """将 datetime 转换为 BDS 周内秒。

    输入 epoch_time 必须是按 BDT 理解的无时区 datetime。
    """
    epoch_time = ensure_bdt_naive_datetime(epoch_time)
    total_seconds = (epoch_time - BDT_EPOCH).total_seconds()
    return total_seconds % SECONDS_IN_WEEK


def _normalize_time(seconds: float) -> float:
    """将时间差归化到半周范围，避免跨周时出现过大的时间差。"""
    if seconds > HALF_WEEK:
        seconds -= SECONDS_IN_WEEK
    elif seconds < -HALF_WEEK:
        seconds += SECONDS_IN_WEEK
    return seconds


def _solve_kepler(mean_anomaly: float, eccentricity: float, max_iter: int, tol: float) -> float:
    """迭代求解 Kepler 方程 E = M + e * sin(E)。"""
    eccentric_anomaly = mean_anomaly
    for _ in range(max_iter):
        next_value = mean_anomaly + eccentricity * math.sin(eccentric_anomaly)
        if abs(next_value - eccentric_anomaly) < tol:
            return next_value
        eccentric_anomaly = next_value
    return eccentric_anomaly


def _is_bds_geo(sat_id: str) -> bool:
    """粗略判断北斗 GEO 卫星。"""
    try:
        prn = int(sat_id[1:])
    except ValueError:
        return False
    return 1 <= prn <= 5 or 59 <= prn <= 63


def _semi_major_axis(eph: BroadcastEphemeris, tk: float = 0.0) -> float:
    """Return semi-major axis in meters, including CNAV A_dot when present."""

    return eph.sqrt_a * eph.sqrt_a + getattr(eph, "a_dot", 0.0) * tk


def compute_satellite_position(
    eph: BroadcastEphemeris,
    epoch_time: datetime,
) -> Tuple[float, float, float]:
    """基于广播星历计算卫星 ECEF 坐标。

    返回 (sat_x, sat_y, sat_z)，单位为米。
    算法依据：北斗 ICD 文件广播星历标准公式。
    """
    epoch_time = ensure_bdt_naive_datetime(epoch_time)
    if eph.sqrt_a <= 0.0:
        raise ValueError(f"{eph.sat_id} 的 sqrtA 非法：{eph.sqrt_a}")

    t = _bds_seconds_of_week(epoch_time)
    tk = _normalize_time(t - eph.toe)

    semi_major_axis = _semi_major_axis(eph, tk)
    mean_motion_0 = math.sqrt(MU / (semi_major_axis ** 3))
    mean_motion = mean_motion_0 + eph.delta_n

    mean_anomaly = math.fmod(eph.m0 + mean_motion * tk, 2.0 * math.pi)
    eccentric_anomaly = _solve_kepler(mean_anomaly, eph.eccentricity, 30, 1e-13)

    sin_e = math.sin(eccentric_anomaly)
    cos_e = math.cos(eccentric_anomaly)

    true_anomaly = math.atan2(
        math.sqrt(1.0 - eph.eccentricity * eph.eccentricity) * sin_e,
        cos_e - eph.eccentricity,
    )
    phi = true_anomaly + eph.omega

    sin_2phi = math.sin(2.0 * phi)
    cos_2phi = math.cos(2.0 * phi)
    delta_u = eph.cus * sin_2phi + eph.cuc * cos_2phi
    delta_r = eph.crs * sin_2phi + eph.crc * cos_2phi
    delta_i = eph.cis * sin_2phi + eph.cic * cos_2phi

    u = phi + delta_u
    r = semi_major_axis * (1.0 - eph.eccentricity * cos_e) + delta_r
    i = eph.i0 + eph.idot * tk + delta_i

    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)

    if _is_bds_geo(eph.sat_id):
        omega_k = eph.omega0 + eph.omega_dot * tk - OMEGA_E * eph.toe
        x_g = x_orb * math.cos(omega_k) - y_orb * math.cos(i) * math.sin(omega_k)
        y_g = x_orb * math.sin(omega_k) + y_orb * math.cos(i) * math.cos(omega_k)
        z_g = y_orb * math.sin(i)

        geo_tilt = math.radians(-5.0)
        x_t = x_g
        y_t = y_g * math.cos(geo_tilt) + z_g * math.sin(geo_tilt)
        z_t = -y_g * math.sin(geo_tilt) + z_g * math.cos(geo_tilt)

        cos_rot = math.cos(OMEGA_E * tk)
        sin_rot = math.sin(OMEGA_E * tk)
        x = x_t * cos_rot + y_t * sin_rot
        y = -x_t * sin_rot + y_t * cos_rot
        z = z_t
    else:
        omega_k = eph.omega0 + (eph.omega_dot - OMEGA_E) * tk - OMEGA_E * eph.toe
        x = x_orb * math.cos(omega_k) - y_orb * math.cos(i) * math.sin(omega_k)
        y = x_orb * math.sin(omega_k) + y_orb * math.cos(i) * math.cos(omega_k)
        z = y_orb * math.sin(i)

    return x, y, z


def compute_satellite_clock_bias(
    eph: BroadcastEphemeris,
    epoch_time: datetime,
) -> Tuple[float, float]:
    """基于广播星历计算卫星钟差（单位：秒）和相对论效应修正（单位：秒）。

    返回 (clock_bias, relativistic_correction)。

    钟差计算模型：
        clock_bias = af0 + af1 * dt + af2 * dt^2 + relativistic_correction
    其中：
        dt = 观测历元相对星历钟参考时刻 toc 的时间差（单位：s）
        af0：卫星钟差多项式常数项（单位：s）
        af1：卫星钟差多项式一次项（单位：s/s）
        af2：卫星钟差多项式二次项（单位：s/s^2）

    相对论效应修正：
        relativistic_correction = F * e * sqrt(a) * sin(E)
    其中：
        F = -2 * sqrt(mu) / c^2（常数，单位 s/m^(1/2)）
        e：轨道偏心率
        sqrt(a)：轨道长半轴平方根（单位：sqrt(m)）
        E：偏近点角（单位：rad）

    dt_clock = epoch_time - eph.toc
    其中 epoch_time 和 eph.toc 在本项目中均统一按 BDT 时间系统理解。
    """
    epoch_time = ensure_bdt_naive_datetime(epoch_time)
    # 计算观测历元相对星历钟参考时刻 toc 的时间差 dt_clock（单位：s）
    dt_clock = _normalize_time((epoch_time - eph.toc).total_seconds())

    # 重新计算偏近点角 E（单位：rad），用于相对论效应修正
    # 计算过程与 compute_satellite_position 中完全一致，确保 E 相同
    t = _bds_seconds_of_week(epoch_time)
    tk = _normalize_time(t - eph.toe)
    semi_major_axis = _semi_major_axis(eph, tk)
    mean_motion_0 = math.sqrt(MU / (semi_major_axis ** 3))
    mean_motion = mean_motion_0 + eph.delta_n
    mean_anomaly = math.fmod(eph.m0 + mean_motion * tk, 2.0 * math.pi)
    eccentric_anomaly = _solve_kepler(mean_anomaly, eph.eccentricity, 30, 1e-13)
    sin_e = math.sin(eccentric_anomaly)

    # 相对论效应修正（单位：s）
    # 公式：F * e * sqrt(a) * sin(E)
    # RELATIVITY_F = -2 * sqrt(MU) / (C * C)，已在模块顶部定义
    relativity = RELATIVITY_F * eph.eccentricity * math.sqrt(semi_major_axis) * sin_e

    # 卫星钟差多项式修正（单位：s）
    # 公式：af0 + af1 * dt + af2 * dt^2
    clock_bias = eph.af0 + eph.af1 * dt_clock + eph.af2 * dt_clock * dt_clock + relativity
    return clock_bias, relativity


def compute_satellite_position_with_debug(
    eph: BroadcastEphemeris,
    epoch_time: datetime,
) -> dict:
    """基于广播星历计算卫星 ECEF 坐标，并返回中间调试变量。

    返回字典包含以下字段（所有角度单位为 rad，距离单位为 m，时间单位为 s）：
    - x, y, z: ECEF 坐标（m）
    - tk: 相对 toe 的时间差（s）
    - semi_major_axis: 轨道长半轴（m）
    - mean_motion_0: 参考平均角速度（rad/s）
    - mean_motion: 改正后平均角速度（rad/s）
    - mean_anomaly: 平近点角（rad）
    - eccentric_anomaly: 偏近点角（rad）
    - true_anomaly: 真近点角（rad）
    - phi: 纬度幅角（rad）
    - delta_u: 纬度幅角摄动修正（rad）
    - delta_r: 轨道半径摄动修正（m）
    - delta_i: 轨道倾角摄动修正（rad）
    - u: 改正后纬度幅角（rad）
    - r: 改正后轨道半径（m）
    - i: 改正后轨道倾角（rad）
    - omega_k: 升交点赤经（rad）
    - x_orb, y_orb: 轨道平面坐标（m）
    - is_geo: 是否为 GEO 卫星
    """
    epoch_time = ensure_bdt_naive_datetime(epoch_time)
    if eph.sqrt_a <= 0.0:
        raise ValueError(f"{eph.sat_id} 的 sqrtA 非法：{eph.sqrt_a}")

    # 时间差（单位：s）
    t = _bds_seconds_of_week(epoch_time)
    tk = _normalize_time(t - eph.toe)

    # 轨道参数
    semi_major_axis = _semi_major_axis(eph, tk)
    mean_motion_0 = math.sqrt(MU / (semi_major_axis ** 3))
    mean_motion = mean_motion_0 + eph.delta_n
    mean_anomaly = math.fmod(eph.m0 + mean_motion * tk, 2.0 * math.pi)
    eccentric_anomaly = _solve_kepler(mean_anomaly, eph.eccentricity, 30, 1e-13)

    sin_e = math.sin(eccentric_anomaly)
    cos_e = math.cos(eccentric_anomaly)

    true_anomaly = math.atan2(
        math.sqrt(1.0 - eph.eccentricity * eph.eccentricity) * sin_e,
        cos_e - eph.eccentricity,
    )
    phi = true_anomaly + eph.omega

    sin_2phi = math.sin(2.0 * phi)
    cos_2phi = math.cos(2.0 * phi)
    delta_u = eph.cus * sin_2phi + eph.cuc * cos_2phi
    delta_r = eph.crs * sin_2phi + eph.crc * cos_2phi
    delta_i = eph.cis * sin_2phi + eph.cic * cos_2phi

    u = phi + delta_u
    r = semi_major_axis * (1.0 - eph.eccentricity * cos_e) + delta_r
    i = eph.i0 + eph.idot * tk + delta_i

    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)

    is_geo = _is_bds_geo(eph.sat_id)
    if is_geo:
        omega_k = eph.omega0 + eph.omega_dot * tk - OMEGA_E * eph.toe
        x_g = x_orb * math.cos(omega_k) - y_orb * math.cos(i) * math.sin(omega_k)
        y_g = x_orb * math.sin(omega_k) + y_orb * math.cos(i) * math.cos(omega_k)
        z_g = y_orb * math.sin(i)

        geo_tilt = math.radians(-5.0)
        x_t = x_g
        y_t = y_g * math.cos(geo_tilt) + z_g * math.sin(geo_tilt)
        z_t = -y_g * math.sin(geo_tilt) + z_g * math.cos(geo_tilt)

        cos_rot = math.cos(OMEGA_E * tk)
        sin_rot = math.sin(OMEGA_E * tk)
        x = x_t * cos_rot + y_t * sin_rot
        y = -x_t * sin_rot + y_t * cos_rot
        z = z_t
    else:
        omega_k = eph.omega0 + (eph.omega_dot - OMEGA_E) * tk - OMEGA_E * eph.toe
        x = x_orb * math.cos(omega_k) - y_orb * math.cos(i) * math.sin(omega_k)
        y = x_orb * math.sin(omega_k) + y_orb * math.cos(i) * math.cos(omega_k)
        z = y_orb * math.sin(i)

    return {
        "x": x, "y": y, "z": z,
        "tk": tk,
        "semi_major_axis": semi_major_axis,
        "mean_motion_0": mean_motion_0,
        "mean_motion": mean_motion,
        "mean_anomaly": mean_anomaly,
        "eccentric_anomaly": eccentric_anomaly,
        "true_anomaly": true_anomaly,
        "phi": phi,
        "delta_u": delta_u,
        "delta_r": delta_r,
        "delta_i": delta_i,
        "u": u,
        "r": r,
        "i": i,
        "omega_k": omega_k,
        "x_orb": x_orb,
        "y_orb": y_orb,
        "is_geo": is_geo,
    }


# ============================================================================
# 几何距离与高度角计算（从 module3 迁移）
# ============================================================================
def ecef_to_blh(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """将 WGS84 ECEF 坐标转换为经纬高。

    返回值为 (lat, lon, height)，纬度和经度单位为度，高程单位为米。
    """
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))

    for _ in range(20):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = p / max(math.cos(lat), 1e-15) - n
        next_lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + height)))
        if abs(next_lat - lat) < 1e-12:
            lat = next_lat
            break
        lat = next_lat

    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    height = p / max(math.cos(lat), 1e-15) - n
    return math.degrees(lat), math.degrees(lon), height


def compute_geometric_range(
    receiver_xyz: Tuple[float, float, float],
    satellite_xyz: Tuple[float, float, float],
) -> float:
    """计算卫星到接收机的几何距离 rho，单位 m。"""
    dx = satellite_xyz[0] - receiver_xyz[0]
    dy = satellite_xyz[1] - receiver_xyz[1]
    dz = satellite_xyz[2] - receiver_xyz[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_elevation(
    receiver_xyz: Tuple[float, float, float],
    satellite_xyz: Tuple[float, float, float],
) -> float:
    """计算卫星相对接收机的高度角，单位为度。"""
    lat_deg, lon_deg, _ = ecef_to_blh(*receiver_xyz)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    dx = satellite_xyz[0] - receiver_xyz[0]
    dy = satellite_xyz[1] - receiver_xyz[1]
    dz = satellite_xyz[2] - receiver_xyz[2]

    east = -math.sin(lon) * dx + math.cos(lon) * dy
    north = (
        -math.sin(lat) * math.cos(lon) * dx
        - math.sin(lat) * math.sin(lon) * dy
        + math.cos(lat) * dz
    )
    up = (
        math.cos(lat) * math.cos(lon) * dx
        + math.cos(lat) * math.sin(lon) * dy
        + math.sin(lat) * dz
    )
    horizontal = math.sqrt(east * east + north * north)
    return math.degrees(math.atan2(up, horizontal))


# ============================================================================
# 伪距模拟（从 module3 迁移改造）
# ============================================================================
def simulate_pseudorange(rho: float, rng: random.Random) -> dict:
    """按误差模型生成模拟伪距，并保存每一项误差。

    误差模型：
        P = rho
            + random.gauss(0, 0.5)       # SISRE
            + random.gauss(10, 3)        # 电离层误差
            + random.gauss(4, 1.5)       # 对流层误差
            + (60 + random.gauss(0, 12)) # 接收机钟差
            + random.gauss(0, 0.3)       # 观测噪声

    参数:
        rho: 几何距离，单位 m
        rng: random.Random 实例，保证可复现性

    返回:
        包含各误差项和最终伪距的字典
    """
    sisre_error = rng.gauss(0.0, 0.5)
    iono_error = rng.gauss(10.0, 3.0)
    tropo_error = rng.gauss(4.0, 1.5)
    receiver_clock_error = 60.0 + rng.gauss(0.0, 12.0)
    noise_error = rng.gauss(0.0, 0.3)
    pseudorange = (
        rho
        + sisre_error
        + iono_error
        + tropo_error
        + receiver_clock_error
        + noise_error
    )

    return {
        "rho": rho,
        "sisre_error": sisre_error,
        "iono_error": iono_error,
        "tropo_error": tropo_error,
        "receiver_clock_error": receiver_clock_error,
        "noise_error": noise_error,
        "pseudorange": pseudorange,
    }


# ============================================================================
# 预处理与过滤
# ============================================================================
def preprocess_pseudorange_records(
    records: List[dict],
    elevation_mask_deg: float = 0.0,
    enable_outlier_filter: bool = False,
    outlier_threshold_m: Optional[float] = None,
) -> List[dict]:
    """对模拟伪距记录进行预处理过滤。

    过滤规则（按优先级）：
    1. 不健康卫星（health != 0）→ reject_reason = "unhealthy"
    2. 高度角低于阈值 → reject_reason = "low_elevation"
    3. 伪距超出合理范围 [15M, 50M] m → reject_reason = "range_outlier"
       （仅在 enable_outlier_filter=True 时执行）
    4. 相对中位数偏差过大 → reject_reason = "statistical_outlier"
       （仅在 enable_outlier_filter=True 且 outlier_threshold_m 不为 None 时执行）

    被剔除的数据保留在返回列表中，is_used 标记为 False。
    """
    if not records:
        return []

    # 收集所有伪距用于统计（仅在启用粗差筛选时计算）
    median_p = 0.0
    if enable_outlier_filter and outlier_threshold_m is not None and outlier_threshold_m > 0:
        pseudorange_values = [r["pseudorange"] for r in records]
        median_p = float(sorted(pseudorange_values)[len(pseudorange_values) // 2]) if pseudorange_values else 0.0

    processed: List[dict] = []
    for rec in records:
        rec = dict(rec)  # 复制，避免修改原数据
        rec["is_used"] = True
        rec["reject_reason"] = ""

        # 1. 健康筛选（始终执行）
        health = rec.get("health", 0.0)
        if health not in ("", None) and int(round(float(health))) != 0:
            rec["is_used"] = False
            rec["reject_reason"] = "unhealthy"
            processed.append(rec)
            continue

        # 2. 高度角筛选（始终执行）
        elevation = rec.get("elevation_deg", 90.0)
        if elevation < elevation_mask_deg:
            rec["is_used"] = False
            rec["reject_reason"] = "low_elevation"
            processed.append(rec)
            continue

        # 3. 伪距范围检查（仅在启用粗差筛选时执行）
        if enable_outlier_filter:
            p = rec["pseudorange"]
            if not (15_000_000.0 <= p <= 50_000_000.0):
                rec["is_used"] = False
                rec["reject_reason"] = "range_outlier"
                processed.append(rec)
                continue

            # 4. 统计粗差检查（仅在启用粗差筛选且阈值有效时执行）
            if outlier_threshold_m is not None and outlier_threshold_m > 0:
                if abs(p - median_p) > outlier_threshold_m:
                    rec["is_used"] = False
                    rec["reject_reason"] = "statistical_outlier"
                    processed.append(rec)
                    continue

        processed.append(rec)

    return processed


# ============================================================================
# 统一入口函数
# ============================================================================
def run_module1(
    nav_path: str | Path,
    receiver_approx: Tuple[float, float, float],
    epochs: List[datetime],
    seed: int,
    output_dir: str | Path = "outputs/basic/module",
    elevation_mask_deg: float = 0.0,
    enable_pseudorange_outlier_filter: bool = False,
) -> Dict[str, Path]:
    """模块一统一入口：解析 NAV、计算卫星位置、生成模拟伪距、输出 CSV。

    参数:
        nav_path: RINEX NAV 文件路径
        receiver_approx: 接收机概略 ECEF 坐标 (x, y, z)，单位 m
        epochs: 仿真历元列表
        seed: 随机数种子，保证可复现
        output_dir: 输出目录
        elevation_mask_deg: 高度角截止阈值，默认 0°

    返回:
        {"nav_debug": path1, "simulated_pseudorange": path2}
    """
    nav_path = Path(nav_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 解析 NAV 文件（自动判断是否为 BDS-3 CNAV 格式）
    nav_data, parse_info = parse_nav_file(nav_path)

    # 2. 初始化随机数生成器
    rng = random.Random(seed)

    # 3. 逐历元处理
    all_records: List[dict] = []
    for epoch in epochs:
        # 同一历元共用接收机钟差
        receiver_clock_error = 60.0 + rng.gauss(0.0, 12.0)

        for sat_id in sorted(nav_data):
            eph = select_ephemeris(nav_data, sat_id, epoch, healthy_only=False)
            if eph is None:
                continue

            try:
                sat_x, sat_y, sat_z = compute_satellite_position(eph, epoch)
            except Exception:
                continue

            rho = compute_geometric_range(receiver_approx, (sat_x, sat_y, sat_z))
            elevation_deg = compute_elevation(receiver_approx, (sat_x, sat_y, sat_z))

            # 生成伪距（使用同一历元的接收机钟差）
            sim = simulate_pseudorange(rho, rng)
            # 用同一历元的统一接收机钟差覆盖 individual 随机值
            sim["receiver_clock_error"] = receiver_clock_error
            sim["pseudorange"] = (
                rho
                + sim["sisre_error"]
                + sim["iono_error"]
                + sim["tropo_error"]
                + receiver_clock_error
                + sim["noise_error"]
            )

            record = {
                "epoch": epoch.isoformat(sep=" "),
                "sat_id": sat_id,
                "sat_x": sat_x,
                "sat_y": sat_y,
                "sat_z": sat_z,
                "receiver_x": receiver_approx[0],
                "receiver_y": receiver_approx[1],
                "receiver_z": receiver_approx[2],
                "elevation_deg": elevation_deg,
                "health": eph.health,
                **sim,
            }
            all_records.append(record)

    # 4. 预处理过滤
    processed_records = preprocess_pseudorange_records(
        all_records,
        elevation_mask_deg=elevation_mask_deg,
        enable_outlier_filter=enable_pseudorange_outlier_filter,
    )

    # 5. 保存输出
    nav_debug_path = _save_nav_debug_csv(nav_data, output_path)
    pseudo_path = _save_simulated_pseudorange_csv(processed_records, output_path)
    summary_path = _save_nav_summary(nav_data, output_path, parse_info)

    return {
        "nav_debug": nav_debug_path,
        "simulated_pseudorange": pseudo_path,
        "summary": summary_path,
        "records": processed_records,
    }


# ============================================================================
# 输出文件保存
# ============================================================================
def _save_nav_debug_csv(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    output_path: Path,
) -> Path:
    """保存 module1_导航电文解析调试.csv。"""
    csv_path = output_path / "module1_导航电文解析调试.csv"
    fieldnames = [
        "sat_id", "toc", "toe", "af0", "af1", "af2",
        "sqrtA", "e", "i0", "Omega0", "omega", "M0",
        "DeltaN", "IDOT", "Cuc", "Cus", "Crc", "Crs", "Cic", "Cis",
        "health", "is_healthy", "parse_status",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sat_id in sorted(nav_data):
            for eph in nav_data[sat_id]:
                is_healthy = int(round(eph.health)) == 0
                writer.writerow(
                    {
                        "sat_id": eph.sat_id,
                        "toc": eph.toc.isoformat(sep=" "),
                        "toe": eph.toe,
                        "af0": eph.af0,
                        "af1": eph.af1,
                        "af2": eph.af2,
                        "sqrtA": eph.sqrt_a,
                        "e": eph.eccentricity,
                        "i0": eph.i0,
                        "Omega0": eph.omega0,
                        "omega": eph.omega,
                        "M0": eph.m0,
                        "DeltaN": eph.delta_n,
                        "IDOT": eph.idot,
                        "Cuc": eph.cuc,
                        "Cus": eph.cus,
                        "Crc": eph.crc,
                        "Crs": eph.crs,
                        "Cic": eph.cic,
                        "Cis": eph.cis,
                        "health": eph.health,
                        "is_healthy": "yes" if is_healthy else "no",
                        "parse_status": getattr(eph, "parse_status", "ok"),
                    }
                )

    return csv_path


def _save_nav_summary(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    output_path: Path,
    parse_info: NavParseInfo,
) -> Path:
    """保存 module1_导航电文解析汇总.txt。"""
    summary_path = output_path / "module1_导航电文解析汇总.txt"
    total_records = sum(len(items) for items in nav_data.values())
    with summary_path.open("w", encoding="utf-8-sig") as file:
        file.write("模块一：RINEX NAV 导航文件解析结果\n")
        file.write("=" * 50 + "\n")
        file.write(f"NAV 文件路径：{parse_info.nav_file_path}\n")
        file.write(f"RINEX 文件版本：{parse_info.rinex_version}\n")
        file.write(f"成功解析的北斗三号卫星数量：{len(nav_data)}\n")
        file.write(f"成功解析的广播星历记录总数：{total_records}\n")
        file.write(f"跳过的非北斗记录数：{parse_info.skipped_non_bds_records}\n")
        file.write(f"跳过的北斗二号记录数：{parse_info.skipped_bds2_records}\n")
        file.write(f"不完整记录数：{parse_info.incomplete_records}\n")
        file.write(f"解析失败的北斗记录数：{parse_info.failed_records}\n")
        file.write("\n模块运行状态：导航文件解析完成。\n")

    return summary_path


def _save_ephemeris_list_csv(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    output_path: Path,
) -> Path:
    """保存 module1_星历列表.csv（兼容旧输出格式）。"""
    csv_path = output_path / "module1_星历列表.csv"
    fieldnames = [
        "sat_id", "toc_time", "toe", "af0", "af1", "af2",
        "sqrtA", "e", "i0", "Omega0", "omega", "M0", "delta_n", "health",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sat_id in sorted(nav_data):
            for eph in nav_data[sat_id]:
                writer.writerow(
                    {
                        "sat_id": eph.sat_id,
                        "toc_time": eph.toc.isoformat(sep=" "),
                        "toe": eph.toe,
                        "af0": eph.af0,
                        "af1": eph.af1,
                        "af2": eph.af2,
                        "sqrtA": eph.sqrt_a,
                        "e": eph.eccentricity,
                        "i0": eph.i0,
                        "Omega0": eph.omega0,
                        "omega": eph.omega,
                        "M0": eph.m0,
                        "delta_n": eph.delta_n,
                        "health": eph.health,
                    }
                )

    return csv_path


def _save_simulated_pseudorange_csv(records: List[dict], output_path: Path) -> Path:
    """保存 module1_模拟伪距.csv。"""
    csv_path = output_path / "module1_模拟伪距.csv"
    fieldnames = [
        "epoch", "sat_id", "sat_x", "sat_y", "sat_z",
        "receiver_x", "receiver_y", "receiver_z",
        "elevation_deg", "rho", "sisre_error", "iono_error",
        "tropo_error", "receiver_clock_error", "noise_error",
        "pseudorange", "is_used", "reject_reason",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "epoch": rec.get("epoch", ""),
                    "sat_id": rec.get("sat_id", ""),
                    "sat_x": rec.get("sat_x", ""),
                    "sat_y": rec.get("sat_y", ""),
                    "sat_z": rec.get("sat_z", ""),
                    "receiver_x": rec.get("receiver_x", ""),
                    "receiver_y": rec.get("receiver_y", ""),
                    "receiver_z": rec.get("receiver_z", ""),
                    "elevation_deg": rec.get("elevation_deg", ""),
                    "rho": rec.get("rho", ""),
                    "sisre_error": rec.get("sisre_error", ""),
                    "iono_error": rec.get("iono_error", ""),
                    "tropo_error": rec.get("tropo_error", ""),
                    "receiver_clock_error": rec.get("receiver_clock_error", ""),
                    "noise_error": rec.get("noise_error", ""),
                    "pseudorange": rec.get("pseudorange", ""),
                    "is_used": "yes" if rec.get("is_used", True) else "no",
                    "reject_reason": rec.get("reject_reason", ""),
                }
            )

    return csv_path


# ============================================================================
# 兼容旧接口
# ============================================================================
def save_nav_parse_outputs(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    output_dir: str | Path,
    parse_info: Optional[NavParseInfo] = None,
    nav_file_path: str | Path | None = None,
) -> Dict[str, Path]:
    """兼容旧接口：保存解析摘要 TXT 和星历明细 CSV。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    info = parse_info or NavParseInfo(nav_file_path=str(nav_file_path or "未知"))
    summary_path = _save_nav_summary(nav_data, output_path, info)
    ephemeris_path = _save_ephemeris_list_csv(nav_data, output_path)

    return {"summary": summary_path, "ephemeris_csv": ephemeris_path}


def read_rinex_nav(nav_file: str | Path) -> Dict[str, List[BroadcastEphemeris]]:
    """parse_rinex_nav 的别名，便于外部调用。"""
    return parse_rinex_nav(nav_file)


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    RECEIVER_APPROX = (-2267800.0, 5009340.0, 3221000.0)
    SEED = 2026
    EPOCHS = [datetime(2026, 4, 1, 0, 0, 0)]

    paths = run_module1(
        nav_path="nav/tarc0910.26b_cnav",
        receiver_approx=RECEIVER_APPROX,
        epochs=EPOCHS,
        seed=SEED,
        output_dir="outputs/basic/module",
        elevation_mask_deg=0.0,
    )

    print(f"模块一输出文件：")
    print(f"  导航调试：{paths['nav_debug']}")
    print(f"  模拟伪距：{paths['simulated_pseudorange']}")
    print(f"  解析摘要：{paths['summary']}")
