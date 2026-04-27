# Data directory

## Real study data

The response data analysed in the publication is governed by the
Institutional Review Board of Seoul National University Hospital
(IRB No. 2206-008-1328) and cannot be redistributed. It is available
from the corresponding authors on reasonable request, subject to a data
transfer agreement and the terms of the IRB protocol.

To reproduce the paper's numbers exactly, place the CSV at
`data/responses.csv` and run:

```bash
python -m vestibular_classifier --data data/responses.csv
```

## Schema

| column         | dtype  | description                                                   |
| -------------- | ------ | ------------------------------------------------------------- |
| `D_number`     | int    | Anonymous subject identifier                                  |
| `D_Age`        | int    | 6-digit birth date, YYMMDD (century inferred from YY)         |
| `D_Gender`     | int    | 1 = male, 2 = female                                          |
| `D_Time_final` | str    | Date of questionnaire completion (parseable by `pd.to_datetime`) |
| `D_Time`       | int    | Completion time in seconds (0 if not recorded)                |
| `Q01` … `Q50`  | int    | Forced-choice answer codes. Yes/No questions use 1 = Yes, 2 = No. Categorical questions use 1/2/3 in the order presented in the app (see supplementary materials of the paper). |
| `App_isOld`      | int    | App-side age flag (recomputed; value in CSV is ignored)       |
| `App_isVas`      | int    | App-side vascular flag (recomputed; value in CSV is ignored)  |
| `App_vasScore`   | int    | App-side vascular count (recomputed; value in CSV is ignored) |
| `Ref_Dx1`      | str    | Reference otologist's primary diagnosis                       |
| `Ref_Dx2`      | str    | Reference otologist's secondary diagnosis (`OTHERS` if none)  |
| `Ref_Vas`      | bool   | True if reference otologist flagged vascular vertigo          |

Diagnosis labels use the short codes `BPPV`, `MD`, `VM`, `PPPD`, `VEST`,
`OH`, and the out-of-scope placeholder `OTHERS`. Legacy CSVs using the
older codes `PV` (for `VEST`) and `OS` (for `OTHERS`) are automatically
translated on load.
