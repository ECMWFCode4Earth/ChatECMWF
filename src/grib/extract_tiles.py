import argparse
import logging
import logging.config
import os
import sqlite3
import sys
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename")
parser.add_argument("-d", "--date")
parser.add_argument("-t", "--time")
parser.add_argument("-n", "--name")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.filename
    path = "/".join(filename.split("/")[:-1])
    if filename is None:
        logger.error("Please supply the variable name!")
        sys.exit(1)
    logger.info("Connecting to mbtiles...")
    conn = sqlite3.connect("{}_colored.mbtiles".format(filename))
    logger.info("Selecting tiles from mbtiles...")
    results = conn.execute("select * from tiles").fetchall()
    logger.info("Saving the tiles...")
    today = args.date
    hour = args.time
    var_name = args.name
    for result in results:
        zoom, column, row, png = result
        try:
            os.makedirs(
                "{}/data/{}/{}/output/{}/{}/".format(path, today, hour, zoom, row)
            )
        except:
            logger.warning("Directory already existing!")
            pass
        tile_out = open(
            "{}/data/{}/{}/output/{}/{}/{}.png".format(
                path, today, hour, zoom, row, column
            ),
            "wb",
        )
        tile_out.write(png)
        tile_out.close()

        logger.info("Tiles successully created!")
    sys.exit(0)
