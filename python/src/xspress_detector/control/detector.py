import asyncio
import copy
import logging
import getpass
import inspect
import time
import math

from enum import Enum
from functools import partial
from datetime import datetime
from contextlib import suppress
from numbers import Number as number

from typing import Any

from odin.adapters.adapter import ApiAdapterRequest
from odin.adapters.async_adapter import AsyncApiAdapter
from odin.adapters.async_parameter_tree import AsyncParameterTree
from odin_data.control.ipc_message import IpcMessage

from .client import AsyncClient
from .util import ListModeIPPortGen
from .debug import debug_method


from .xspress_api import XspressApi

def raise_exception(ex_class, message):
    """
    Utility function for raising exceptions from lambdas
    """
    raise ex_class(message)


ENDPOINT_TEMPLATE = "tcp://{}:{}"
UPDATE_PARAMS_PERIOD_SECONDS = 0.5
QUEUE_SIZE_LIMIT = 10

XSPRESS_MODE_MCA = "mca"
XSPRESS_MODE_LIST = "list"
XSPRESS_EXPOSURE_LOWER_LIMIT = 1.0 / 1000000
XSPRESS_EXPOSURE_UPPER_LIMIT = 20.0

NUM_FR_MCA = 9
NUM_FR_LIST = 8

FR_INIT_TIME = {
    XSPRESS_MODE_MCA: 1,  # seconds. Arbitrary wait time.
    XSPRESS_MODE_LIST: 1,  # seconds. This is a macro in the FR List cpp code
}


def nearest_mult_of_5_up(x):
    return math.ceil(x / 5) * 5


def is_coroutine(object):
    """
    see: https://stackoverflow.com/questions/67020609/partial-of-a-class-coroutine-isnt-a-coroutine-why
    """
    while isinstance(object, partial):
        object = object.func
    return inspect.iscoroutinefunction(object)


class AsyncPeriodicJob:
    def __init__(self, callback, time_seconds, time_min=0.001):
        self.callback = callback
        self._time_min = time_min
        self.time = 1
        self.set_time(time_seconds)
        self._running = False
        self._task = None

        self._last_logging_time = 0
        self.logging_interval = (
            30  # testing increase the time necessary for the configuration
        )

    def start(self):
        if not self._running:
            # Start task to call func periodically:
            self._task = asyncio.ensure_future(self._run())
            self._running = True

    async def stop(self):
        if self._running:
            # Stop task and await it stopped:
            self._task.cancel()
            self._running = False
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self):
        while True:
            try:
                await self.callback()
            except Exception as e:
                current = time.time()
                if current - self._last_logging_time > self.logging_interval:
                    logging.error(
                        f"{self.__class__}._run: raised and cought exception{e}"
                    )
                    self._last_logging_time = current
            await asyncio.sleep(self.time)

    def is_running(self):
        return self._running

    def set_time(self, time):
        if time < self._time_min:
            logging.warning(
                f"{self.__class__}.__init__: setting time to minimum allowed = {self._time_min}"
            )
            self.time = self._time_min
        else:
            self.time = time

    def get_time(self):
        return self.time


class MessageType(Enum):
    STATUS = 0
    CMD = 1
    CONFIG = 2
    REQUEST = 3
    APP = 4
    DAQ = 5


class ParamAccess(Enum):
    """Enumeration of parameter access"""

    ReadOnly = 0
    ReadWrite = 1
    WriteOnly = 2


def msgType(message_type: MessageType):
    if message_type == MessageType.APP:
        return XspressDetectorStr.APP
    elif message_type == MessageType.CONFIG:
        return XspressDetectorStr.CONFIG
    elif message_type == MessageType.CMD:
        return XspressDetectorStr.CMD
    elif message_type == MessageType.DAQ:
        return XspressDetectorStr.CONFIG_DAQ
    else:
        raise XspressDetectorException(
            f"xspress_odin._build_message: {message_type} is not type MessageType"
        )

def _build_message(message_type: MessageType, config: dict = None):
    if message_type == MessageType.REQUEST:
        msg = IpcMessage("cmd", XspressDetectorStr.CONFIG_REQUEST)
        return msg
    params_group = msgType(message_type)
    msg = IpcMessage("cmd", "configure")
    msg.set_param(params_group, config)
    return msg


class XspressTriggerMode:
    TM_SOFTWARE = 0
    TM_TTL_RISING_EDGE = 1
    TM_BURST = 2
    TM_TTL_VETO_ONLY = 3
    TM_SOFTWARE_START_STOP = 4
    TM_IDC = 5
    TM_TTL_BOTH = 6
    TM_LVDS_VETO_ONLY = 7
    TM_LVDS_BOTH = 8

    def str2int(mode):
        x = XspressTriggerMode
        map = dict(
            software=x.TM_SOFTWARE,
            ttl_rising_edge=x.TM_TTL_RISING_EDGE,
            burst=x.TM_BURST,
            ttl_veto_only=x.TM_TTL_VETO_ONLY,
            software_start_stop=x.TM_SOFTWARE_START_STOP,
            idc=x.TM_IDC,
            ttl_both=x.TM_TTL_BOTH,
            lvds_veto_only=x.TM_LVDS_VETO_ONLY,
            lvds_both=x.TM_LVDS_BOTH,
        )
        return map[mode]


class XspressDetectorStr:
    STATUS = "status"
    STATUS_ERROR = "error"
    STATUS_STATE = "state"
    STATUS_CONNECTED = "connected"
    STATUS_RECONNECT_REQUIRED = "reconnect_required"
    STATUS_SENSOR = "sensor"
    STATUS_SENSOR_HEIGHT = "height"
    STATUS_SENSOR_WIDTH = "width"
    STATUS_SENSOR_BYTES = "bytes"
    STATUS_MANUFACTURER = "manufacturer"
    STATUS_MODEL = "model"
    STATUS_ACQ_COMPLETE = "acquisition_complete"
    STATUS_FRAMES = "frames_acquired"
    STATUS_SCALAR_0 = "scalar_0"
    STATUS_SCALAR_1 = "scalar_1"
    STATUS_SCALAR_2 = "scalar_2"
    STATUS_SCALAR_3 = "scalar_3"
    STATUS_SCALAR_4 = "scalar_4"
    STATUS_SCALAR_5 = "scalar_5"
    STATUS_SCALAR_6 = "scalar_6"
    STATUS_SCALAR_7 = "scalar_7"
    STATUS_SCALAR_8 = "scalar_8"
    STATUS_DTC = "dtc"
    STATUS_INP_EST = "inp_est"
    STATUS_TEMP_0 = "temp_0"
    STATUS_TEMP_1 = "temp_1"
    STATUS_TEMP_2 = "temp_2"
    STATUS_TEMP_3 = "temp_3"
    STATUS_TEMP_4 = "temp_4"
    STATUS_TEMP_5 = "temp_5"
    STATUS_CH_FRAMES = "ch_frames_acquired"
    STATUS_FEM_DROPPED_FRAMES = "fem_dropped_frames"
    STATUS_CARDS_CONNECTED = "cards_connected"
    STATUS_NUM_CH_CONNECTED = "num_ch_connected"

    ADAPTER = "adapter"
    ADAPTER_RESET = "reset"
    ADAPTER_SCAN = "scan"
    ADAPTER_UPDATE = "update"
    ADAPTER_START_TIME = "start_time"
    ADAPTER_UP_TIME = "up_time"
    ADAPTER_CONNECTED = "connected"
    ADAPTER_USERNAME = "username"
    ADAPTER_DEBUG_LEVEL = "debug_level"

    APP = "app"
    APP_SHUTDOWN = "shutdow"
    APP_DEBUG = "debug_level"
    APP_CTRL_ENDPOINT = "ctrl_endpoint"
    CONFIG_REQUEST = "request_configuration"

    CONFIG = "config"
    CONFIG_NUM_CARDS = "num_cards"
    CONFIG_NUM_TF = "num_tf"
    CONFIG_BASE_IP = "base_ip"
    CONFIG_MAX_CHANNELS = "max_channels"
    CONFIG_MCA_CHANNELS = "mca_channels"
    CONFIG_MAX_SPECTRA = "max_spectra"
    CONFIG_DEBUG = "debug"
    CONFIG_CONFIG_PATH = "config_path"
    CONFIG_CONFIG_SAVE_PATH = "config_save_path"
    CONFIG_USE_RESGRADES = "use_resgrades"
    CONFIG_RUN_FLAGS = "run_flags"
    CONFIG_DTC_ENERGY = "dtc_energy"
    CONFIG_TRIGGER_MODE = "trigger_mode"
    CONFIG_INVERT_F0 = "invert_f0"
    CONFIG_INVERT_VETO = "invert_veto"
    CONFIG_DEBOUNCE = "debounce"
    CONFIG_EXPOSURE_TIME = "exposure_time"
    CONFIG_NUM_IMAGES = "num_images"  # so only "num_images" is used

    CONFIG_MODE = "mode"
    CONFIG_MODE_CONTROL = "mode_control"
    CONFIG_SCA5_LOW = "sca5_low_lim"
    CONFIG_SCA5_HIGH = "sca5_high_lim"
    CONFIG_SCA6_LOW = "sca6_low_lim"
    CONFIG_SCA6_HIGH = "sca6_high_lim"
    CONFIG_SCA4_THRESH = "sca4_threshold"

    CONFIG_DTC_FLAGS = "dtc_flags"
    CONFIG_DTC_ALL_EVT_OFF = "dtc_all_evt_off"
    CONFIG_DTC_ALL_EVT_GRAD = "dtc_all_evt_grad"
    CONFIG_DTC_ALL_EVT_RATE_OFF = "dtc_all_evt_rate_off"
    CONFIG_DTC_ALL_EVT_RATE_GRAD = "dtc_all_evt_rate_grad"
    CONFIG_DTC_IN_WIN_OFF = "dtc_in_win_off"
    CONFIG_DTC_IN_WIN_GRAD = "dtc_in_win_grad"
    CONFIG_DTC_IN_WIN_RATE_OFF = "dtc_in_win_rate_off"
    CONFIG_DTC_IN_WIN_RATE_GRAD = "dtc_in_win_rate_grad"

    CONFIG_DAQ = "daq"
    CONFIG_DAQ_ENABLED = "enabled"
    CONFIG_DAQ_ZMQ_ENDPOINTS = "endpoints"

    CMD = "command"
    CMD_RECONFIGURE = "reconfigure"
    CMD_CONNECT = "connect"
    CMD_DISCONNECT = "disconnect"
    CMD_SAVE = "save"
    CMD_RESTORE = "restore"
    CMD_START = "start"
    CMD_STOP = "stop"
    CMD_TRIGGER = "trigger"
    CMD_START_ACQUISITION = "start_acquisition"
    CMD_STOP_ACQUISITION = "stop_acquisition"

    VERSION = "version"
    VERSION_XSPRESS_DETECTOR = "xspress-detector"
    VERSION_XSPRESS_DETECTOR_MAJOR = "major"
    VERSION_XSPRESS_DETECTOR_MINOR = "minor"
    VERSION_XSPRESS_DETECTOR_PATCH = "patch"
    VERSION_XSPRESS_DETECTOR_SHORT = "short"
    VERSION_XSPRESS_DETECTOR_FULL = "full"

    PROCESS = "process"
    PROCESS_NUM_LIST = "num_list"
    PROCESS_NUM_MCA = "num_mca"
    PROCESS_NUM_CHANS_MCA = "num_chan_mca"
    PROCESS_NUM_CHANS_LIST = "num_chan_list"

    API = "api"


class XspressDetectorException(Exception):
    pass


class NotAcknowledgeException(Exception):
    pass


class XspressDetector(object):
    def __init__(
        self,
        name: str,
        ip: str,
        port: int,
        debug_level=logging.INFO,
        num_process_mca=NUM_FR_MCA,
        num_process_list=NUM_FR_LIST,
    ):
        
        self._name = name

        self.logger = logging.getLogger()
        self.logger.setLevel(debug_level)

        self.endpoint = ENDPOINT_TEMPLATE.format("0.0.0.0", 12000)
        self.start_time = datetime.now()
        self.username = getpass.getuser()

        self._async_client = AsyncClient(ip, port)
        self.timeout = 1
        self.sched = AsyncPeriodicJob(self.read_config, UPDATE_PARAMS_PERIOD_SECONDS)

        # process params
        self.num_process_list = num_process_list
        self.num_process_mca = num_process_mca

        # root level parameter tree members
        self.ctr_endpoint = ENDPOINT_TEMPLATE.format(ip, port)

        self.max_channels: int = 0
        self.mca_channels: int = 0
        self.max_spectra: int = 0
        self.use_resgrades: bool = False
        self.run_flags: int = 0

        self.mode: str = ""  # 'mca' or 'list' for readback
        self.acquisition_complete: bool = False
        self._cache = {} # local cache of the parameter tree
        self._param_tree_waited = False

        self.param = {}

        self.param[XspressApi.status_uri] = self.add_parameters(
            MessageType.STATUS, XspressApi.status_parameters, ParamAccess.ReadOnly, XspressApi.status_uri
        )

        self.param[XspressApi.version_uri] = self.add_parameters(
            MessageType.STATUS, XspressApi.version_parameters, ParamAccess.ReadOnly, XspressApi.version_uri
        )

        self.param[XspressApi.config_uri] = self.add_parameters(
            MessageType.CONFIG, XspressApi.config_parameters, ParamAccess.ReadWrite, XspressApi.config_uri
        )


        self.param["module"] = self._name

        self.param["command"] = {"allowed": ["reconfigure","start_acquisition","stop_acquisition"],
                                 "execute": ("", lambda name: self.run_command(name))
                                 }


    async def run_command(self, name):
        match name:
            case "reconfigure":
                await self.reconfigure(1)
            case "start_acquisition":
                await self.acquire(1)
            case "stop_acquisition":
                await self.acquire(0)
            case _:
                logging.error(f"Unsupported command {name}")

    async def _get_value(self,path):
        return self._cache.get(path,None)

    async def _set_value(self,mtype,path,value):
        return await self._put(mtype,path,value)


    def add_parameters(self, message_type: MessageType,  parameters: Any, writable: ParamAccess, path: str = ""):
        if isinstance(parameters,dict):
            child = {}
            for name, parameter in parameters.items():
                child_path = f"{path}/{name}".strip("/")
                child[name] = self.add_parameters(message_type, parameter, writable, child_path)
            return child
        else:
            return self.add_parameter(message_type, parameters, writable, path)

    def add_parameter(self, message_type: MessageType,  parameter: Any, writable: ParamAccess, path: str = ""):
        # cache the initial value for each param
        self._cache[path] = parameter

        if writable == ParamAccess.ReadOnly:
            leaf = (lambda x=path: self._get_value(x),)
        
        else:
            leaf = (
                    lambda x=path: self._get_value(x),
                    lambda val,x=path: self._set_value(message_type,x,val)
                    )
        return leaf

    @property
    def num_chan_per_process_list(self):
        """
        max_channels: in list mode is mca_channels+1.
        For list mode the last process will sometimes have less 'active' channels, as each card has 10 chans (I believe).
        e.g. when mca_channels = 36 and list_channels = 37,
        then num_chan_per_process_list = 5 if num_process_list = 8,
        or num_chan_per_process_list = 40 if num_process_list = 1,
        """
        return nearest_mult_of_5_up(self.mca_channels) // self.num_process_list

    @property
    def num_chan_per_process_mca(self):
        """
        this is correct if num_process % mca_channels == 0
        """
        return self.mca_channels // self.num_process_mca

    def set_fr_handler(self, handler):
        self.fr_handler: AsyncApiAdapter = handler
        self.logger.error(f"self.fr_handler set to {handler}")
        req = ApiAdapterRequest(None, accept="application/json")
        resp = self.fr_handler.get(path="config", request=req).data
        logging.error(f"recieved from fr adaptor: {resp}")

    def set_fp_handler(self, handler):
        self.fp_handler: AsyncApiAdapter = handler
        self.logger.error(f"self.fp_handler set to {handler}")
        req = ApiAdapterRequest(None, accept="application/json")
        resp = self.fr_handler.get(path="config", request=req).data
        logging.error(f"recieved from fr adaptor: {resp}")

    async def configure_fps(self, mode: str):
        list_command = {"execute": {"index": "list"}}
        mca_command = {"execute": {"index": "mca"}, "hdf": {"dataset": {}}}
        command = mca_command if mode == XSPRESS_MODE_MCA else list_command
        num_process = (
            self.num_process_mca if mode == XSPRESS_MODE_MCA else self.num_process_list
        )
        # must copy otherwise we'll modify the same dict later
        configs = [copy.deepcopy(command) for _ in range(num_process)]

        if mode == XSPRESS_MODE_MCA:
            dataset_values = {"dims": [1, 4096], "chunks": [1, 1, 4096]}
            for i in range(self.mca_channels):
                fp_index = i // self.num_chan_per_process_mca
                configs[fp_index]["hdf"]["dataset"][f"mca_{i}"] = dataset_values

        tasks = ()
        for client, config in zip(self.fp_clients, configs):
            tasks += (asyncio.create_task(self.async_send_task(client, config)),)
        result = await asyncio.gather(*tasks)

    async def configure_frs(self, mode: int):
        if mode == XSPRESS_MODE_LIST:
            configs = [
                {
                    "rx_ports": ",".join([str(p) for p in ports]),
                    "rx_type": "udp",
                    "decoder_type": "XspressListMode",
                    "rx_address": ip,
                    "rx_recv_buffer_size": 30000000,
                }
                for ip, ports in ListModeIPPortGen(
                    self.num_chan_per_process_list, self.num_process_list
                )
            ]
        elif mode == XSPRESS_MODE_MCA:
            configs = [
                {
                    "rx_ports": "{},".format(index + 15150),
                    "rx_type": "zmq",
                    "decoder_type": "Xspress",
                    "rx_address": "127.0.0.1",
                }
                for index in range(self.num_process_mca)
            ]
        else:
            raise ValueError("invalid mode requested")

        tasks = ()
        for client, config in zip(self.fr_clients, configs):
            tasks += (asyncio.create_task(self.async_send_task(client, config)),)
        result = await asyncio.gather(*tasks)

    async def async_send_task(self, client, config):
        msg = IpcMessage("cmd", "configure")
        msg.set_params(config)
        await client.wait_till_connected()
        resp = await client.send_recv(msg)
        if resp.get_msg_type() == resp.NACK:
            raise NotAcknowledgeException()
        return resp

    async def reconfigure(self, *unused):
        mode = (
            self._cache["config/mode"]
        )  # save local copy so the value doesn't change from under our feet

        # This seems to help when dealing with detectors that don't have 8 channels.
        # Apparently when the detector has a number of channels different from that defined in 'DEFAULT_MAX_CHANNELS',
        # at XspressDetector.h, we have some problems when trying to connect to it straigh away using the reconfigure button.
        await self.reset()
        await asyncio.sleep(1)

        resp = await self._async_client.send_recv(self.configuration.get())
        # resp = await self._put(MessageType.CONFIG, XspressDetectorStr.CONFIG_CONFIG_PATH, self.settings_paths[mode])
        resp = await self._put(MessageType.CMD, "command/disconnect", 1) # Here disconnect is inside config
        chans = self.mca_channels if mode == XSPRESS_MODE_MCA else self.mca_channels + 1
        await self._put(
            MessageType.CONFIG, "config/max_channels", chans
        )
        self.max_channels = chans
        await self.configure_frs(mode)
        await self.configure_fps(mode)
        await asyncio.sleep(1)
        resp = await self.connect()
        if mode == XSPRESS_MODE_MCA:
            resp = await self._async_client.send_recv(self.configuration.get_daq())
        return resp

    async def connect(self, value: bool = True, *unused):
        msg = _build_message(MessageType.CMD, {XspressDetectorStr.CMD_CONNECT: value})
        return await self._async_client.send_recv(msg, timeout=20)

    async def acquire(self, value):
        if value:
            reply = await self._put(MessageType.CMD, XspressDetectorStr.CMD_START, 1)
            self.acquisition_complete = False
            return reply
        else:
            return await self._put(MessageType.CMD, XspressDetectorStr.CMD_STOP, 1)

    async def set_mode(self, value):
        if value == 0:
            return await self._put(
                MessageType.CONFIG, XspressDetectorStr.CONFIG_MODE, XSPRESS_MODE_MCA
            )
        else:  # mode == "list"
            return await self._put(
                MessageType.CONFIG, XspressDetectorStr.CONFIG_MODE, XSPRESS_MODE_LIST
            )

    async def do_updates(self, value: int):
        if value:
            self.sched.start()
        else:
            await self.sched.stop()

    def _set(self, attr_name, value):
        setattr(self, attr_name, value)

    async def _put(self, message_type: MessageType, path: str, value: any):
        if not self._param_tree_waited:
            self.parameter_tree = await AsyncParameterTree(self.param)
            self._param_tree_waited = True

        self.logger.debug(debug_method())
        if not self._async_client.is_connected():
            raise ConnectionError(
                "Control server is not connected! Check if it is running and tcp endpoint is configured correctly"
            )

        # field, item = path.split("/")
        
        if isinstance(value,list):
            index = 0
            for i in range(len(value)):
                if value[i] != -1:
                    index = i
                    break
            self._cache[path][index] = value[index]
            msg = _build_message(message_type, {path: self._cache[path]})
            resp = await self._async_client.send_recv(msg)
            return resp

        msg = _build_message(message_type, {path: value})
        resp = await self._async_client.send_recv(msg)
        return resp

    async def get(self, path):
        # return self.parameter_tree.get(path)
        # Initializes the tree
        if not self._param_tree_waited:
            self.parameter_tree = await AsyncParameterTree(self.param)
            self._param_tree_waited = True


        # If asked for the whole parameter tree
        if path == '':
            return await self.parameter_tree.get(path)

        # If asked for element in a list ("param/list/elem")
        elif path.split("/")[-1].isdigit():
            field, param, index = path.split("/")
            newPath = field + "/" + param
            ret = await self.parameter_tree.get(newPath)
            return ret[param][int(index)]
       
        # Value calls
        else:
            ret = await self.parameter_tree.get(path)
            return ret[path.split("/")[-1]]

    def configure(
        self,
        num_cards,
        num_tf,
        base_ip,
        max_channels,
        max_spectra,
        settings_path,
        run_flags,
        debug,
        daq_endpoints,
    ):
        self.logger.critical(debug_method())
        self.max_channels = max_channels
        self.mca_channels = max_channels
        self.max_spectra = max_spectra
        self.run_flags = run_flags
        self.fr_clients = [
            AsyncClient("127.0.0.1", 10000 + (10 * port_offset))
            for port_offset in range(self.num_process_mca)
        ]
        self.fp_clients = [
            AsyncClient("127.0.0.1", 10004 + (10 * port_offset))
            for port_offset in range(self.num_process_mca)
        ]

        x = XspressDetectorStr
        logging.info("MessageType.CONFIG = {}".format(MessageType.CONFIG))

        class Configuration:
            has_been_called_once = False

            def __init__(self):
                self.initial_config_msg = _build_message(
                    MessageType.CONFIG,
                    {
                        x.CONFIG_NUM_CARDS: num_cards,
                        x.CONFIG_NUM_TF: num_tf,
                        x.CONFIG_BASE_IP: base_ip,
                        x.CONFIG_MAX_CHANNELS: max_channels,
                        x.CONFIG_MCA_CHANNELS: max_channels,
                        x.CONFIG_MAX_SPECTRA: max_spectra,
                        x.CONFIG_CONFIG_PATH: settings_path,
                        x.CONFIG_DEBUG: debug,
                    },
                )
                self.re_config_message = _build_message(
                    MessageType.CONFIG,
                    {  # these are constants that the user is unlikely to change
                        x.CONFIG_NUM_CARDS: num_cards,
                        x.CONFIG_BASE_IP: base_ip,
                        x.CONFIG_MAX_SPECTRA: max_spectra,
                    },
                )
                self.initial_daq_msg = _build_message(
                    MessageType.DAQ,
                    {
                        x.CONFIG_DAQ_ZMQ_ENDPOINTS: daq_endpoints,
                        x.CONFIG_DAQ_ENABLED: True,
                    },
                )

            def get(self):
                if self.has_been_called_once:
                    return self.re_config_message
                else:
                    self.has_been_called_once = True
                    return self.initial_config_msg

            def get_daq(self):
                return self.initial_daq_msg

            def get_initial(self):
                return self.initial_config_msg

        self.configuration = Configuration()
        self.sched.start()

    async def reset(self, *unused):
        resp = await self._async_client.send_recv(self.configuration.get_initial())
        if self.mode == XSPRESS_MODE_MCA:
            resp = await self._async_client.send_recv(self.configuration.get_daq())
        return resp

    async def read_config(self, *unused):
        msg = _build_message(MessageType.REQUEST)
        ipc_msg = await self._async_client.send_recv(msg, loud=False)
        BASE_PATH = ""
        if not ipc_msg.is_valid():
            raise XspressDetectorException("IpcMessage recieved is not valid")
        # self._param_tree_set_recursive(BASE_PATH, ipc_msg.get_params())
        dictMsg = ipc_msg.get_params()
        self._param_tree_set_recursive(BASE_PATH, dictMsg)
        return ipc_msg

    def _param_tree_set_recursive(self, path, params):
        if not isinstance(params, dict):
            try:
                path = path.strip("/")
                self._cache[path] = params

            except KeyError as e:
                self.logger.error(e)
                self.logger.warning(
                    (
                        f"XspressDetector._param_tree_set_recursive: parameter path {path} is not in parameter tree\n"
                        f"params = {params}"
                    )
                )
        else:
            for param_name, params in params.items():
                self._param_tree_set_recursive(f"{path}/{param_name}", params)
