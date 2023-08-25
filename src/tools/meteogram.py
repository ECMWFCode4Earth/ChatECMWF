import datetime

import requests


def meteogram(lat: float, lon: float, station_name: str):
    """
    This tool can be used to retrieve a meteogram for a given time and location.
    A meteogram is a graphical representation of the following data: total cloud cover, total precipation, 10m wind speed, 2m temperature.
    For each meteogram one needs to specify a location, and the time.
    """
    now = datetime.datetime.utcnow()
    now = datetime.datetime(now.year, now.month, now.day, 12)

    for _ in range(4):
        base_time = now.isoformat()

        r = requests.get(
            "https://charts.ecmwf.int/opencharts-api/v1/products/opencharts_meteogram/"
            f"?lat={lat}&lon={lon}&station_name={station_name}&base_time={base_time}Z"
        )
        if r.status_code != 404:
            break

        now = now - datetime.timedelta(hours=12)

    r.raise_for_status()
    url = r.json()["data"]["link"]["href"]
    return f"{url}"
