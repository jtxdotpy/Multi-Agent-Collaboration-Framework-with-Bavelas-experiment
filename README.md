# Code for Multi-Agent Collaboration Framework with partially observable information in various communication network topologies

This is the code repository for the paper ASAP: a Multi-Agent Collaboration Framework with partially observable information in various communication network topologies. More code is coming soon.

## Requisite

Our code requires Python library openai>=1.0.0

Replace your openai_key in `management.py` line 13.

## Running Script

To run our full model with a 3-agent chain-structure setting, you can use the following command

```powershell
python main.py -n 3 -s 3 -t chain -M full -r 10
```