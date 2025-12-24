import pandas as pd
from os import makedirs

from utils.utils import (
        load_base_stations_positions,
        load_config)

cfg = load_config("config/config.yaml")

file_path = "mobility/map"
base_stations = load_base_stations_positions(cfg["simulation"]["base_station"]["positions"])[0]

writer_buffer = "<additional>\n"

for base_station in base_stations:
    
    writer_buffer += f"\t<poi id=\"bs\" x={base_station[0]} y=\"{base_station[1]}\" color=\"255,0,0\" layer=3/>\n"

writer_buffer += "</additional>"

makedirs(file_path,
         exist_ok=True)

with open(f"{file_path}/poi.xml", "w") as writer:

    writer.writelines(writer_buffer)
    
