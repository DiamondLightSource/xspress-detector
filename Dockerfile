FROM ghcr.io/odin-detector/odin-data-build:latest AS build

# Install numactl for the MetaWriter process
RUN apt-get update
RUN apt-get install numactl

# Copy xspress-detector source in for build
COPY . /tmp/xspress-detector

# In here you should copy your version of libxspress inside the container
# COPY ./libxspress /libxspress

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
FROM ghcr.io/odin-detector/odin-data-runtime:latest
COPY --from=build /odin /odin
# COPY --from=build /libxspress /libxspress
ENV PATH=/odin/bin:/odin/venv/bin:$PATH
WORKDIR /odin