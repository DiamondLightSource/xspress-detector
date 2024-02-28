"""Implementation of Xspress Meta Writer

This module is a subclass of the odin_data MetaWriter and handles Xspress
specific meta messages, writing them to disk.

Alan Greer, Diamond Light Source
"""

import struct
from datetime import datetime

from odin_data.meta_writer.hdf5dataset import (
    Float64HDF5Dataset,
    Int32HDF5Dataset,
    Int64HDF5Dataset,
)
from odin_data.meta_writer.meta_writer import MetaWriter
from odin_data.util import construct_version_dict
from xspress_detector._version import get_versions

# Data message types
XSPRESS_SCALARS = "xspress_scalars"
XSPRESS_DTC = "xspress_dtc"
XSPRESS_INP_EST = "xspress_inp_est"

# Number of scalars per channel
XSPRESS_SCALARS_PER_CHANNEL = 9

# Chunk dimension
XSPRESS_SCALAR_CHUNK_DIM = 4000

# Dataset names
DATASET_SCALAR = "scalar_"
DATASET_DTC = "dtc"
DATASET_INP_EST = "inp_est"
DATASET_DAQ_VERSION = "data_version"
DATASET_META_VERSION = "meta_version"


class XspressMetaWriter(MetaWriter):
    """Implementation of MetaWriter that also handles Xspress meta messages"""

    def __init__(self, name, directory, endpoints, config):
        # This must be defined for _define_detector_datasets in base class __init__
        self._sensor_shape = config.sensor_shape
        self._num_channels = self._sensor_shape[0]

        super(XspressMetaWriter, self).__init__(name, directory, endpoints, config)

        self._logger.info("Num channels set to: {}".format(self._num_channels))
        self._series = None
        self._flush_time = datetime.now()
        self._logger.info("Initialised XspressMetaWriter")
        self._scalars = {
            0: {},
            1: {},
            2: {},
            3: {},
            4: {},
            5: {},
            6: {},
            7: {},
            8: {}
        }
        self._dtc = {}
        self._inp = {}

    def _define_detector_datasets(self):
        dsets = []
        for index in range(XSPRESS_SCALARS_PER_CHANNEL):
            scalar_name = "{}{}".format(DATASET_SCALAR, index)
            self._logger.info("Adding dataset: {}".format(scalar_name))
            dsets.append(Int32HDF5Dataset(scalar_name, shape=(0, self._num_channels), maxshape=(None, self._num_channels), chunks=(XSPRESS_SCALAR_CHUNK_DIM, self._num_channels), rank=2, cache=True, block_size=2000))
        self._logger.info("Adding dataset: {}".format(DATASET_DTC))
        dsets.append(Float64HDF5Dataset(DATASET_DTC, shape=(0, self._num_channels), maxshape=(None, self._num_channels), chunks=(XSPRESS_SCALAR_CHUNK_DIM, self._num_channels), rank=2, cache=True, block_size=2000))
        self._logger.info("Adding dataset: {}".format(DATASET_INP_EST))
        dsets.append(Float64HDF5Dataset(DATASET_INP_EST, shape=(0, self._num_channels), maxshape=(None, self._num_channels), chunks=(XSPRESS_SCALAR_CHUNK_DIM, self._num_channels), rank=2, cache=True, block_size=2000))
        dsets.append(Int64HDF5Dataset(DATASET_DAQ_VERSION))
        dsets.append(Int64HDF5Dataset(DATASET_META_VERSION))
        return dsets

    @property
    def detector_message_handlers(self):
        return {
            XSPRESS_SCALARS: self.handle_xspress_scalars,
            XSPRESS_DTC: self.handle_xspress_dtc,
            XSPRESS_INP_EST: self.handle_xspress_inp_est,
        }

    def handle_xspress_scalars(self, header, _data):
        """Handle global header message part 1"""
        self._logger.debug("%s | Handling xspress scalar message", self._name)
        self._logger.debug("{}".format(header))
        # Extract the channel number from the header
        channel = header['channel_index']

        format_str = '{}i'.format(header['qty_scalars']*header['number_of_frames'])
        array = struct.unpack(format_str, _data)

        # Number of channels
        number_of_channels = header['number_of_channels']
        # Number of frames
        number_of_frames = header['number_of_frames']
        # Frame ID
        frame_id = header['frame_id']
        for frame in range(number_of_frames):
            current_frame_id = frame_id + frame
            frame_index = frame*number_of_channels*XSPRESS_SCALARS_PER_CHANNEL
            frame_array = array[frame_index:frame_index+(number_of_channels*XSPRESS_SCALARS_PER_CHANNEL)]
            for index in range(number_of_channels):
                arr_index = index*XSPRESS_SCALARS_PER_CHANNEL
                scalars = frame_array[arr_index:arr_index+XSPRESS_SCALARS_PER_CHANNEL]
                dataset_name = "{}{}".format(DATASET_SCALAR, channel+index)
                for scalar_index in range(9):
                    self.add_scalar_value(current_frame_id, channel+index, scalar_index, scalars[scalar_index])

        if (datetime.now() - self._flush_time).total_seconds() > 0.25:
            self._flush_datasets()
            self._flush_time = datetime.now()

    def add_scalar_value(self, frame, channel, scalar, value):
        # Check if we have an entry for the scalar
        obj = self._scalars[scalar]
        if frame not in obj:
            obj[frame] = {
                'qty': 0,
                'values': [0] * self._num_channels
            }

        obj[frame]['values'][channel] = value
        obj[frame]['qty'] += 1

        if obj[frame]['qty'] == self._num_channels:
            dataset_name = "{}{}".format(DATASET_SCALAR, scalar)
            self._add_value(dataset_name, obj[frame]['values'], offset=frame)
            del obj[frame]

        self._scalars[scalar] = obj

    def handle_xspress_dtc(self, header, _data):
        """Handle global header message part 1"""
        self._logger.debug("%s | Handling xspress dtc message", self._name)
        self._logger.debug("{}".format(header))
        # Extract the channel number from the header
        channel = header['channel_index']
        # Extract Number of channels
        number_of_channels = header['number_of_channels']
        # Number of frames
        number_of_frames = header['number_of_frames']
        # Frame ID
        frame_id = header['frame_id']

        format_str = '{}d'.format(number_of_channels*number_of_frames)
        array = struct.unpack(format_str, _data)

        for frame in range(number_of_frames):
            current_frame_id = frame_id + frame
            frame_index = frame*number_of_channels
            frame_array = array[frame_index:frame_index+number_of_channels]
            for index in range(number_of_channels):
                self.add_dtc_value(current_frame_id, channel+index, frame_array[index])

    def add_dtc_value(self, frame, channel, value):
        # Check if we need to create an entry for this frame
        if frame not in self._dtc:
            self._dtc[frame] = {
                'qty': 0,
                'values': [0] * self._num_channels
            }

        self._dtc[frame]['values'][channel] = value
        self._dtc[frame]['qty'] += 1

        if self._dtc[frame]['qty'] == self._num_channels:
            self._add_value(DATASET_DTC, self._dtc[frame]['values'], offset=frame)
            del self._dtc[frame]

    def handle_xspress_inp_est(self, header, _data):
        """Handle global header message part 1"""
        self._logger.debug("%s | Handling xspress dtc message", self._name)
        self._logger.debug("{}".format(header))
        # Extract the channel number from the header
        channel = header['channel_index']
        # Extract Number of channels
        number_of_channels = header['number_of_channels']
        # Number of frames
        number_of_frames = header['number_of_frames']
        # Frame ID
        frame_id = header['frame_id']

        format_str = '{}d'.format(number_of_channels*number_of_frames)
        array = struct.unpack(format_str, _data)

        for frame in range(number_of_frames):
            current_frame_id = frame_id + frame
            frame_index = frame*number_of_channels
            frame_array = array[frame_index:frame_index+number_of_channels]
            for index in range(number_of_channels):
                self.add_inp_est_value(current_frame_id, channel+index, frame_array[index])

    def add_inp_est_value(self, frame, channel, value):
        # Check if we need to create an entry for this frame
        if frame not in self._inp:
            self._inp[frame] = {
                'qty': 0,
                'values': [0] * self._num_channels
            }

        self._inp[frame]['values'][channel] = value
        self._inp[frame]['qty'] += 1

        if self._inp[frame]['qty'] == self._num_channels:
            self._add_value(DATASET_INP_EST, self._inp[frame]['values'], offset=frame)
            del self._inp[frame]

    @staticmethod
    def get_version():
        return (
            "xspress-detector",
            construct_version_dict(get_versions()["version"]),
        )
