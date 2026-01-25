from awpy.data import map_data

print("Downloading/refreshing awpy map data...")
# При импорте awpy обычно пытается загрузить map-data.json.
# В некоторых версиях есть helper. Попробуем безопасно:
try:
    from awpy.cli.get import get_maps
    get_maps()
    print("Done: get_maps()")
except Exception as e:
    print("get_maps() failed:", e)
    print("Fallback: just import map_data and check local files.")

print("Local map data path:", map_data.MAP_DATA_PATH if hasattr(map_data, "MAP_DATA_PATH") else "unknown")
