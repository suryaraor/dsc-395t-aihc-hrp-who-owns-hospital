"""Run the whole pipeline in order. Usage: python run_all.py [--from 02]   (stage ids: 00 01 01b 02 ... 08)"""
import subprocess
import sys
from pathlib import Path

STAGES = [("00", "00_download.py"), ("01", "01_clean.py"), ("01b", "01b_nppes.py"),
          ("02", "02_entity_resolution.py"), ("03", "03_graph.py"), ("04", "04_link_prediction.py"),
          ("05", "05_quality_cost.py"), ("06", "06_error_analysis.py"), ("07", "07_figures.py"),
          ("08", "08_robustness.py")]

ids = [s for s, _ in STAGES]
start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "00"
if start not in ids:
    sys.exit(f"unknown stage {start!r}; choose one of {ids}")
here = Path(__file__).parent
RAW = here.parent / "04-Data" / "raw"
AHRQ = ["ahrq_hospital_linkage_2018.csv", "ahrq_hospital_linkage_2022.csv", "ahrq_systems_2018.csv", "ahrq_systems_2022.csv"]
for sid, script in STAGES[ids.index(start):]:
    if sid == "01":
        missing = [f for f in AHRQ if not (RAW / f).exists() or (RAW / f).stat().st_size == 0]
        if missing:  # ahrq.gov blocks scripted downloads; see README "Manual step"
            sys.exit(f"Download these AHRQ files in a browser into {RAW} first: {missing}")
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(here / script)], cwd=here, check=True)
print("\npipeline complete")
