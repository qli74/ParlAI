Task: covid
==============
Description: covid-19 QA data and a poly-encoder model

Tags: #covid

1.Download files (including train&valid data and model)
```
git clone https://github.com/qli74/ParlAI
cd ParlAI
python examples/display_data.py -t covid -n 1
```

2.web chat on localhost (default port: 35496, or use --port PORT_NUMBER)
```
./start_browser_service.sh
```
![example](https://github.com/qli74/ParlAI/blob/master/cov1.png)

3.terminal chat
```
./start_terminal_service.sh
```
![example](https://github.com/qli74/ParlAI/blob/master/cov2.png)


4.another api file written with fastapi: ParlAI/fastapi_covid.py\
https://github.com/qli74/ParlAI/blob/master/fastapi_covid.py
