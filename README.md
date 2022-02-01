# VAP SIRS Analysis

This repository contains code to generate three figures from the paper
Krueger et al "Risk assessment of COVID-19 epidemic resurgence in relation to SARS-CoV-2 variants and vaccination passes".
## Prerequisities

- Python 3

## Installation

- `python3 -m venv venv`
- `. venv/bin/activate`
- `pip install -r requirements.txt`

Now all should be ready to run `python main.py`. It takes a moment to generate Fig 1b,
around 90 minutes to generate Fig 2, and about 6 hours to generate Fig 3 or Fig 4.

## Data Availability
All the data used for figures generation is provided within the [data](data) directory. 
See [data readme](data/README.md) for further details.