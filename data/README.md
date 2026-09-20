# Data

The simulation study generates its own data; nothing is needed here for it.

The MTO application expects the Moving to Opportunity adult file at
`data/mto.dta` (Stata format). This file is **not** included in the repository:
MTO data are distributed under a data use agreement and must be obtained from
the data provider directly.

`experiments/mto/run_mto_fna.py` reads the following variables:

| Role | Variable(s) |
|---|---|
| Treatment arm | `ra_group` |
| Mediator | `ps_f_c9010t_perpov_dw` |
| Outcome | `ps_f_mh_idx_z_ad` |
| Site | `ra_site` |
| Baseline covariates | `ps_x_*` (listed in `build_mto_sample`) |

Any file placed in this directory other than this README is ignored by git.
