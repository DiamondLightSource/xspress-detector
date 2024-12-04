#!/bin/bash

numactl --membind=0 --cpunodebind=0 /venv/bin/xspress_meta_writer -w xspress_detector.data.xspress_meta_writer.XspressMetaWriter --sensor-shape 8 4096 --data-endpoints tcp://127.0.0.1:10008,tcp://127.0.0.1:10018 --static-log-fields beamline=${BEAMLINE},detector="Xspress4"
