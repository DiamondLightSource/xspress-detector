#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

/venv/bin/xspress_live_merge --sub_ports 15500,15501
