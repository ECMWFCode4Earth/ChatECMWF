#!/usr/bin/python3
"""
Requires gdal 
"""
import argparse
import os
import subprocess as sp
import traceback
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")

if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.filename
    path = "/".join(filename.split("/")[:-1])
    try:
        os.remove(f"{path}/download.grib.aux.xml")
        os.remove(f"{path}/download.grib.png")
    except:
        pass
    if filename is None:
        print("No filename supplied!")
        sys.exit(1)
    gribinfo = sp.run(["gdalinfo", "-stats", "-json", filename], stdout=sp.PIPE)
    grib_dict = None
    for n in range(5):
        try:
            grib_dict = eval(b"\n".join(gribinfo.stdout.split(b"\n")[n:]))
            break
        except:
            traceback.print_exc()
            print(
                f"Could not parse grib. Skipping now {n+1} lines. After 5 trials, the script exits."
            )
            continue

    if not gribinfo is None:
        coordinates = None
        n_bands = None
        v_min, v_max = None, None
        if "cornerCoordinates" in grib_dict.keys():
            coordinates = grib_dict["cornerCoordinates"]
        if "bands" in grib_dict.keys():
            mins, maxs = [], []
            valid_time = []
            bands = []
            counter = 0
            for band in grib_dict["bands"]:
                print(band)
                mins.append(band["minimum"])
                maxs.append(band["maximum"])
                meta_key = list(band["metadata"].keys())[0]
                bands.append(band["band"])
                if counter == 0:
                    ref_time = band["metadata"][meta_key]["GRIB_REF_TIME"]
                    name = band["metadata"][meta_key]["GRIB_COMMENT"]

                valid_time.append(int(band["metadata"][meta_key]["GRIB_VALID_TIME"]))
                counter += 1
            v_min = min(mins)
            v_max = max(maxs)
            start_time = datetime.fromtimestamp(min(valid_time)).strftime("%Y-%m-%d")
            stop_time = datetime.fromtimestamp(max(valid_time)).strftime("%Y-%m-%d")
            n_bands = counter
            for i, band in enumerate(bands):
                date = datetime.fromtimestamp(valid_time[i]).strftime("%Y-%m-%d")
                hours = str(datetime.fromtimestamp(valid_time[i]).hour)
                output = sp.run(
                    [
                        "./build_colored_tif.sh",
                        "-i",
                        f"{path}/download.grib",
                        "-o",
                        f"{path}/output",
                        "-b",
                        f"{band}",
                        "-min",
                        f"{v_min}",
                        "-max",
                        f"{v_max}",
                        "-date",
                        f"{date}",
                        "-hours",
                        f"{hours}",
                        "-name",
                        f"{name}",
                    ]
                )

    with open("map.html.template", "r") as fin:
        with open(f"{path}/map.html", "w") as fout:
            for line in fin:
                if "REPLACE_TITLE" in line:
                    line = line.replace("REPLACE_TITLE", name)
                if "REPLACE_CENTER_X" in line:
                    line = line.replace(
                        "REPLACE_CENTER_X", f"{coordinates['center'][0]}"
                    )
                if "REPLACE_CENTER_Y" in line:
                    line = line.replace(
                        "REPLACE_CENTER_Y", f"{coordinates['center'][1]}"
                    )
                if "REPLACE_TIMES" in line:
                    to_replace = ",".join([f"{v*1000}" for v in valid_time])
                    line = line.replace("REPLACE_TIMES", f"[{to_replace}]")
                if "REPLACE_DOMAIN_MIN" in line:
                    line = line.replace("REPLACE_DOMAIN_MIN", f"{v_min}")
                if "REPLACE_DOMAIN_MAX" in line:
                    line = line.replace("REPLACE_DOMAIN_MAX", f"{v_max}")
                if "REPLACE_GRIB_COMMENT" in line:
                    line = line.replace("REPLACE_GRIB_COMMENT", name)
                print(line)
                fout.write(line)
