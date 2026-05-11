from pathlib import Path

from beidou_spp.rinex.nav_parser import parse_nav_file


def test_parse_sample_nav_has_beidou_satellites():
    nav_path = Path("data/sample.nav")
    nav_data = parse_nav_file(nav_path)
    assert len(nav_data) >= 4
    first_sat = sorted(nav_data)[0]
    eph = nav_data[first_sat][0]
    assert eph.sat_id.startswith("C")
    assert eph.sqrt_a > 0

