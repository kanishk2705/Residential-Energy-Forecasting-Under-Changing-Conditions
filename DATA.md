# Data: provenance, licence, attribution

**Dataset.** Appliances Energy Prediction, UCI Machine Learning Repository
(dataset id 374). 19,735 rows, 10-minute resolution, 2016-01-11 17:00 to
2016-05-27 18:00, one low-energy house in Belgium. Single file:
`energydata_complete.csv`.

**Licence (checked 2026-10-15 on the UCI dataset page).** Creative Commons
Attribution 4.0 International (CC BY 4.0): sharing and adaptation for any
purpose with appropriate credit.

**Third-party component.** The outdoor-weather columns come from the Chièvres
airport weather station via Reliable Prognosis (rp5.ru). The dataset page states
that permission for distributing this weather data was obtained by the dataset
authors. Because of that third-party component, **this repository does not
re-host the raw CSV**. `scripts/get_data.py` downloads it and verifies a
SHA-256 checksum.

**Citations (please cite both if you reuse this).**

- Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5VC8G
- Candanedo, L. M., Feldheim, V., Deramaix, D. (2017). Data driven prediction
  models of energy use of appliances in a low-energy house. *Energy and
  Buildings*, 140, 81-97. https://doi.org/10.1016/j.enbuild.2017.01.083

**Where the file used here came from.** The authors' public GitHub repository
`LuisM78/Appliances-energy-prediction-data` (commit
`e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1`), because the build sandbox could not
reach `archive.ics.uci.edu`. That repository carries no licence file of its own;
the UCI CC BY 4.0 statement is the licence relied upon. The row count (19,735)
matches the UCI page.

SHA-256 of the file used: `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`

**Not independently verified:** that the file served by UCI is byte-identical
to the GitHub copy. `scripts/get_data.py` therefore falls back to a structural
check (row count, first/last timestamp, 10-minute grid) when the checksum differs.
