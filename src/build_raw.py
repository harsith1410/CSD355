import yaml, pandas as pd
from pathlib import Path

base = Path("../f1db-main/f1db-main/src/data/seasons")   # change to wherever you unzipped it
rows = []
for year in range(2014, 2027):
    for race_dir in sorted((base / str(year) / "races").iterdir()):
        quali_file = race_dir / "qualifying-results.yml"
        if not quali_file.exists():
            continue
        race = yaml.safe_load(open(race_dir / "race.yml", encoding="utf-8"))
        for r in yaml.safe_load(open(quali_file, encoding="utf-8")):
            rows.append({"year": year, "round": race["round"],
                         "circuit": race["circuitId"],
                         "circuitType": race["circuitType"],
                         "date": race["date"],
                         **r})



df = pd.DataFrame(rows)

drivers_dir = Path("../f1db-main/f1db-main/src/data/drivers")
dob = {}
for d in df["driverId"].unique():
    info = yaml.safe_load(open(drivers_dir / f"{d}.yml", encoding="utf-8"))
    dob[d] = info["dateOfBirth"]
df["dateOfBirth"] = pd.to_datetime(df["driverId"].map(dob))


print(len(df))          # 5,465 for 2014 to 2026 Singapore
df.to_csv("data/qualifying_raw.csv", index=False)