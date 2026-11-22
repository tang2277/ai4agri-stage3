import pyproj, os
print('pyproj:', pyproj.__version__)
print('PROJ dir:', pyproj.datadir.get_data_dir())

from pyproj import CRS

# 各种构造方式
tests = [
    ("from_epsg(4326)", lambda: CRS.from_epsg(4326)),
    ("from_proj4", lambda: CRS.from_proj4("+proj=longlat +datum=WGS84")),
    ("from_wkt", lambda: CRS.from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')),
]

for name, fn in tests:
    try:
        c = fn()
        print(f"✅ {name}: {c}")
    except Exception as e:
        print(f"❌ {name}: {e}")

# 试试设置 PROJ env
print("\n--- 设置 env 后再试 ---")
os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
os.environ["PROJ_NETWORK"] = "OFF"

try:
    c = CRS.from_epsg(4326)
    print(f"✅ from_epsg after env: {c}")
except Exception as e:
    print(f"❌ from_epsg after env: {e}")

# 直接用 PROJ 字符串
try:
    c = CRS("+init=epsg:4326")
    print(f"✅ +init: {c}")
except Exception as e:
    print(f"❌ +init: {e}")

try:
    c = CRS("epsg:4326")
    print(f"✅ epsg:4326: {c}")
except Exception as e:
    print(f"❌ epsg:4326: {e}")
