
class XspressApi:
    config_uri = "config"
    status_uri = "status"
    adapter_uri = "adapter"
    process_uri = "process"
    version_uri = "version"
    daq_uri = "daq"
    app_uri = "app"

    
    api_items = {
        "api": ""
    }
    app_items = {
        "debug_level": 0,
        "ctrl_endpoint": "",
        "shutdown": 0,
    }
    daq_items = {
        "enabled": False,
        "endpoints": ""
    }
    confi_req_items = {
        "request_configuration": 0,
    }
    adapter_items = {
        "start_time": "",
        "up_time": "",
        "connected": False,
        "username": "",
        "scan": 0,
        "debug_level": 0,
        "update": 0,
        "reset": 0,
    }
    status_items = {
        "sensor": {
            "height": 0,
            "width": 0,
            "bytes": 0,
        },
        "manufacturer": "Quantum Detectors",
        "model": "Xspress 3",
        "acquisition_complete": False,
        "frames_acquired": 0,
        "scalar_0": [],
        "scalar_1": [],
        "scalar_2": [],
        "scalar_3": [],
        "scalar_4": [],
        "scalar_5": [],
        "scalar_6": [],
        "scalar_7": [],
        "scalar_8": [],
        "dtc": [],
        "inp_est": [],
        "error": "",
        "state": "",
        "connected": False,
        "reconnect_required": False,
        "temp_0": [],
        "temp_1": [],
        "temp_2": [],
        "temp_3": [],
        "temp_4": [],
        "temp_5": [],
        "ch_frames_acquired": [],
        "fem_dropped_frames": [],
        "cards_connected": [],
        "num_ch_connected": [],
    }
    config_items = {
        "mode_control": 0,
        "mode": 0,
        "num_cards": 0,
        "num_tf": 0,
        "base_ip": "",
        "max_channels": 0,
        "mca_channels": 0,
        "max_spectra": 0,
        "debug": 0,
        "config_path": "",
        "config_save_path": "",
        "use_resgrades": False,
        "run_flags": 0,
        "dtc_energy": 0.0,
        "trigger_mode": 0,
        "invert_f0": 0,
        "invert_veto": 0,
        "debounce": 0,
        "exposure_time": 1.0,
        "num_images": 1,
        "sca5_low_lim": [],
        "sca5_high_lim": [],
        "sca6_low_lim": [],
        "sca6_high_lim": [],
        "sca4_threshold": [],
        "dtc_flags": [],
        "dtc_all_evt_off": [],
        "dtc_all_evt_grad": [],
        "dtc_all_evt_rate_off": [],
        "dtc_all_evt_rate_grad": [],
        "dtc_in_win_off": [],
        "dtc_in_win_grad": [],
        "dtc_in_win_rate_off": [],
        "dtc_in_win_rate_grad": [],
        "connect": False,
        "disconnect":False,
        "save": False,
        "restore": False,
        "start": False,
        "stop": False,
        "trigger": False,
        "start_acquisition": False,
        "stop_acquisition": False,
        "reconfigure": False,
    }
    version_items = {
        "xspress-detector": {
            "full": "",
            "major": 0,
            "minor": 0,
            "patch": 0,
            "short": "",
        }
    }
    process_items = {
        "num_mca": 0,
        "num_list": 0,
        "num_chan_mca": 0,
        "num_chan_list": 0,
    }

