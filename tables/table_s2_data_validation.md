| check                                       | passed   | observed       |   expected |
|:--------------------------------------------|:---------|:---------------|-----------:|
| all allocations parsed                      | True     | 2500           |       2500 |
| all allocations sum to $1000                | True     | 2500           |       2500 |
| all allocations within [0,1000]             | True     | 2500           |       2500 |
| no duplicate run-scenario-level rows        | True     | 0              |          0 |
| number of input CSV files                   | True     | 10             |         10 |
| number of rows                              | True     | 2500           |       2500 |
| number of run identifiers                   | True     | 50             |         50 |
| number of scenarios                         | True     | 10             |         10 |
| number of levels                            | True     | 5              |          5 |
| balanced 50 observations per run identifier | True     | min=50, max=50 |         50 |
| temperature fixed within run identifier     | True     | 50             |         50 |