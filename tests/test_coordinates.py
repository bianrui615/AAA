from beidou_spp.positioning.coordinates import blh_to_ecef, ecef_to_blh


def test_blh_ecef_roundtrip():
    lat, lon, h = 30.5285, 114.3569, 42.0
    x, y, z = blh_to_ecef(lat, lon, h)
    lat2, lon2, h2 = ecef_to_blh(x, y, z)
    assert abs(lat - lat2) < 1e-7
    assert abs(lon - lon2) < 1e-7
    assert abs(h - h2) < 1e-3


def test_ecef_to_blh_range():
    lat, lon, h = ecef_to_blh(-2267800.0, 5009340.0, 3221000.0)
    assert -90.0 <= lat <= 90.0
    assert -180.0 <= lon <= 180.0
    assert h > -1000.0

