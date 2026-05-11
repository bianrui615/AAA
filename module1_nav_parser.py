"""
module1_nav_parser.py

模块一：RINEX NAV 导航文件解析模块。

本模块只读取 RINEX NAV 导航文件，不读取 OBS 观测文件。解析时只保留
北斗卫星，卫星编号以 C 开头，例如 C01、C02。模块支持 RINEX 3.x 常见
导航文件格式，并兼容科学计数法中的 D/E 指数写法，例如 1.23D-04。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# RINEX 文件中的数字经常写成 1.234D-04 或 1.234E-04。
# 这个正则表达式用于从一行文本中提取整数、小数和 D/E 科学计数法数字。
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

    aode: float  # 星历数据龄期，BDS 中常称 AODE
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
    data_source: float  # 数据源或码标志，不同 RINEX 生成器含义可能略有差异
    week: float  # BDS 周数
    accuracy: float  # 用户测距精度或 SISAI 指示值

    health: float  # 卫星健康状态，通常 0 表示健康
    tgd1: float  # BDS TGD1 群延迟，单位 s
    tgd2: float  # BDS TGD2 群延迟，单位 s
    transmission_time: float  # 电文发射时刻，BDS 周内秒
    fit_interval: float = 0.0  # 拟合区间，部分文件中可能为空


@dataclass
class NavParseInfo:
    """导航文件解析过程中的统计信息。"""

    nav_file_path: str = ""
    rinex_version: str = "未知"
    skipped_non_bds_records: int = 0
    failed_records: int = 0
    incomplete_records: int = 0
    error_messages: List[str] = field(default_factory=list)


def _rinex_float(text: str) -> float:
    """将 RINEX 数字字符串转为 float，兼容 D/E 指数。"""

    return float(text.replace("D", "E").replace("d", "E"))


def _extract_floats(line: str) -> List[float]:
    """从一行 RINEX 文本中提取所有浮点数。"""

    return [_rinex_float(token) for token in _FLOAT_PATTERN.findall(line)]


def _parse_epoch_from_first_line(line: str) -> datetime:
    """解析 RINEX 3.x 导航记录首行中的星历钟参考时间 toc。

    常见首行格式为：
    sat yyyy mm dd hh mm ss af0 af1 af2

    有些文件在秒和 af0 之间没有额外空格，因此这里对时间字段使用固定列
    读取，比直接 split 更稳健。
    """

    year = int(line[4:8])
    month = int(line[9:11])
    day = int(line[12:14])
    hour = int(line[15:17])
    minute = int(line[18:20])
    second_text = line[21:23].strip() or "0"
    second = int(float(second_text))
    return datetime(year, month, day, hour, minute, second)


def _build_ephemeris(record_lines: List[str]) -> Optional[BroadcastEphemeris]:
    """将 8 行 RINEX 3.x BDS 导航记录转换为 BroadcastEphemeris。"""

    if len(record_lines) < 8:
        return None

    first_line = record_lines[0]
    sat_id = first_line[:3].strip()
    if not sat_id.startswith("C"):
        return None

    toc = _parse_epoch_from_first_line(first_line)
    clock_values = _extract_floats(first_line[23:])
    if len(clock_values) < 3:
        raise ValueError(f"{sat_id} {toc} 的卫星钟差字段不完整")

    # RINEX 3 BDS/GPS 类导航记录首行之后通常有 26 个数值字段，排列为：
    # 4+4+4+4+4+4+2。不同软件生成的文件可能会在末尾追加空字段，本程序只
    # 使用前 26 个必要字段，缺失的末尾字段用 0.0 补齐。
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
    )


def parse_rinex_nav_with_info(
    nav_file: str | Path,
) -> Tuple[Dict[str, List[BroadcastEphemeris]], NavParseInfo]:
    """解析 RINEX NAV 文件，并返回北斗星历和解析统计信息。"""

    path = Path(nav_file)
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

    nav_data: Dict[str, List[BroadcastEphemeris]] = {}
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

        record = lines[i : i + 8]
        try:
            eph = _build_ephemeris(record)
            if eph is not None:
                nav_data.setdefault(eph.sat_id, []).append(eph)
        except Exception as exc:
            info.failed_records += 1
            info.error_messages.append(f"{sat_id} 第 {i + 1} 行附近解析失败: {exc}")
        i += 8

    for eph_list in nav_data.values():
        eph_list.sort(key=lambda item: item.toc)
    return nav_data, info


def parse_rinex_nav(nav_file: str | Path) -> Dict[str, List[BroadcastEphemeris]]:
    """解析 RINEX NAV 文件，只返回北斗广播星历字典。"""

    nav_data, _ = parse_rinex_nav_with_info(nav_file)
    return nav_data


def select_ephemeris(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    sat_id: str,
    epoch_time: datetime,
    healthy_only: bool = False,
) -> Optional[BroadcastEphemeris]:
    """为指定卫星和历元选择 toc 时间最接近的星历。"""

    eph_list = nav_data.get(sat_id)
    if not eph_list:
        return None
    if healthy_only:
        eph_list = [eph for eph in eph_list if int(round(eph.health)) == 0]
        if not eph_list:
            return None
    return min(eph_list, key=lambda eph: abs((epoch_time - eph.toc).total_seconds()))


def save_nav_parse_outputs(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    output_dir: str | Path,
    parse_info: Optional[NavParseInfo] = None,
    nav_file_path: str | Path | None = None,
) -> Dict[str, Path]:
    """保存模块一输出：解析摘要 TXT 和星历明细 CSV。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "module1_nav_parse_summary.txt"
    csv_path = output_path / "module1_ephemeris_list.csv"

    info = parse_info or NavParseInfo(nav_file_path=str(nav_file_path or "未知"))
    total_records = sum(len(items) for items in nav_data.values())
    unhealthy = [
        eph for eph_list in nav_data.values() for eph in eph_list if int(round(eph.health)) != 0
    ]

    with summary_path.open("w", encoding="utf-8-sig") as file:
        file.write("模块一：RINEX NAV 导航文件解析结果\n")
        file.write("=" * 50 + "\n")
        file.write(f"NAV 文件路径：{info.nav_file_path or nav_file_path or '未知'}\n")
        file.write(f"RINEX 文件版本：{info.rinex_version}\n")
        file.write(f"成功解析的北斗卫星数量：{len(nav_data)}\n")
        file.write(f"成功解析的广播星历记录总数：{total_records}\n")
        file.write(f"跳过的非北斗记录数：{info.skipped_non_bds_records}\n")
        file.write(f"不完整记录数：{info.incomplete_records}\n")
        file.write(f"解析失败的北斗记录数：{info.failed_records}\n")
        file.write("\n每颗北斗卫星对应的星历记录数：\n")
        for sat_id in sorted(nav_data):
            file.write(f"  {sat_id}：{len(nav_data[sat_id])}\n")
        file.write("\n健康状态检查：\n")
        if unhealthy:
            unhealthy_ids = sorted({eph.sat_id for eph in unhealthy})
            file.write("  是否存在健康状态异常的卫星：是\n")
            file.write(f"  健康状态异常卫星：{', '.join(unhealthy_ids)}\n")
            file.write(f"  健康状态异常星历记录数：{len(unhealthy)}\n")
        else:
            file.write("  是否存在健康状态异常的卫星：否\n")
        if info.error_messages:
            file.write("\n解析失败详情：\n")
            for message in info.error_messages[:20]:
                file.write(f"  {message}\n")
            if len(info.error_messages) > 20:
                file.write(f"  其余 {len(info.error_messages) - 20} 条错误已省略\n")
        file.write("\n模块运行状态：导航文件解析完成。\n")

    fieldnames = [
        "sat_id",
        "toc_time",
        "toe",
        "af0",
        "af1",
        "af2",
        "sqrtA",
        "e",
        "i0",
        "Omega0",
        "omega",
        "M0",
        "delta_n",
        "health",
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

    return {"summary": summary_path, "ephemeris_csv": csv_path}


def read_rinex_nav(nav_file: str | Path) -> Dict[str, List[BroadcastEphemeris]]:
    """parse_rinex_nav 的别名，便于外部调用。"""

    return parse_rinex_nav(nav_file)


if __name__ == "__main__":
    data, parse_info = parse_rinex_nav_with_info("tarc0910.26b")
    paths = save_nav_parse_outputs(data, "output", parse_info)
    print(f"解析到北斗卫星数量：{len(data)}")
    print(f"解析到星历记录总数：{sum(len(items) for items in data.values())}")
    print(f"模块一输出文件：{paths['summary']}，{paths['ephemeris_csv']}")
