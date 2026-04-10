#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

/venv/bin/xspress_live_merge --sub_ports tcp://127.0.0.1:15500,tcp://127.0.0.1:15501
