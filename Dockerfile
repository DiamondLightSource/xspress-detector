FROM ghcr.io/odin-detector/odin-data-build:1.12.0-xspress-dev4 AS developer

RUN git clone https://github.com/DiamondLightSource/libxspress.git /libxspress

FROM developer AS build

# Copy xspress-detector source in for build
COPY . /tmp/xspress-detector

# C++
WORKDIR /tmp/xspress-detector
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/odin -DODINDATA_ROOT_DIR=/odin -DLIBXSPRESS_ROOT_DIR=/libxspress/src ../cpp && \
    make -j8 VERBOSE=1 && \
    make install

# Python
WORKDIR /tmp/xspress-detector/python
RUN python -m pip install .

# Final image
FROM ghcr.io/odin-detector/odin-data-build:1.12.0-xspress-dev4 AS runtime
COPY --from=build /odin /odin
COPY --from=build /libxspress /libxspress
COPY --from=build /venv /venv

ENV PATH=/odin/bin:/odin/venv/bin:$PATH
WORKDIR /odin
