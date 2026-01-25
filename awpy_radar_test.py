import matplotlib
matplotlib.use("Agg")

from awpy.plot import heatmap

fig, ax = heatmap(
    map_name="de_mirage",
    points=[(-608.0, -1371.0, 0.0)],  # любая точка
    method="hex",
    size=12,
    alpha=0.7,
)
fig.savefig("awpy_radar_test.png", dpi=180, bbox_inches="tight", pad_inches=0)
print("saved awpy_radar_test.png")
