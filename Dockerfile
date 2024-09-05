FROM ghcr.io/odin-detector/odin-data-build:1.12.0 AS developer

from developer as build

# Copy xspress-detector source in for build
COPY . /tmp/xspress-detector

# C++
WORKDIR /tmp/xspress-detector
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/odin -DODINDATA_ROOT_DIR=/odin ../cpp && \
    make -j8 VERBOSE=1 && \
    make install

# Python
WORKDIR /tmp/xspress-detector/python
RUN python -m pip install .

# Final image
FROM ghcr.io/odin-detector/odin-data-runtime:1.12.0
COPY --from=build /odin /odin

# Add libxspress here to build xspressControl application
# COPY ./libxspress /libxspress

ENV PATH=/odin/bin:/odin/venv/bin:$PATH
WORKDIR /odin
