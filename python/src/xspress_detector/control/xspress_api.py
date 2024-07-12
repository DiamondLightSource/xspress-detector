
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
            "height": 8,
            "width": 4096,
            "bytes": 0,
        },
        "manufacturer": "Quantum Detectors",
        "model": "Xspress 3",
        "acquisition_complete": False,
        "frames_acquired": 0,
        "scalar_0": [0, 0, 0, 0, 0, 0, 0, 0],
        "scalar_1": [0, 0, 0, 0, 0, 0, 0, 0],
        "scalar_2": [0, 0, 0, 0, 0, 0, 0, 0],
        "scalar_3": [0, 0, 0, 0, 0, 0, 0, 0],
        "scalar_4": [0, 0, 0, 0, 0, 0, 0, 0],
        "scalar_5": [0, 0, 0, 0, 0, 0, 0, 0],
        "scalar_6": [0, 0, 0, 0, 0, 0, 0, 0],
        "scalar_7": [0, 0, 0, 0, 0, 0, 0, 0],
        "scalar_8": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc": [0, 0, 0, 0, 0, 0, 0, 0],
        "inp_est": [0, 0, 0, 0, 0, 0, 0, 0],
        "error": "",
        "state": "",
        "connected": False,
        "reconnect_required": False,
        "temp_0": [0, 0, 0, 0, 0, 0, 0, 0],
        "temp_1": [0, 0, 0, 0, 0, 0, 0, 0],
        "temp_2": [0, 0, 0, 0, 0, 0, 0, 0],
        "temp_3": [0, 0, 0, 0, 0, 0, 0, 0],
        "temp_4": [0, 0, 0, 0, 0, 0, 0, 0],
        "temp_5": [0, 0, 0, 0, 0, 0, 0, 0],
        "ch_frames_acquired": [0, 0, 0, 0, 0, 0, 0, 0],
        "fem_dropped_frames": [0, 0, 0, 0, 0, 0, 0, 0],
        "cards_connected": [0, 0, 0, 0, 0, 0, 0, 0],
        "num_ch_connected": [0, 0, 0, 0, 0, 0, 0, 0],
    }
    config_items = {
        "mode_control": 0,
        "mode": 0,
        "num_cards": 0,
        "num_tf": 0,
        "base_ip": "",
        "max_channels": 8,
        "mca_channels": 8,
        "max_spectra": 4096,
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
        "sca5_low_lim": [0, 0, 0, 0, 0, 0, 0, 0],
        "sca5_high_lim": [0, 0, 0, 0, 0, 0, 0, 0],
        "sca6_low_lim": [0, 0, 0, 0, 0, 0, 0, 0],
        "sca6_high_lim": [0, 0, 0, 0, 0, 0, 0, 0],
        "sca4_threshold": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_flags": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_all_evt_off": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_all_evt_grad": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_all_evt_rate_off": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_all_evt_rate_grad": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_in_win_off": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_in_win_grad": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_in_win_rate_off": [0, 0, 0, 0, 0, 0, 0, 0],
        "dtc_in_win_rate_grad": [0, 0, 0, 0, 0, 0, 0, 0],
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
        "num_mca": 2,
        "num_list": 3,
        "num_chan_mca": 8,
        "num_chan_list": 9,
    }

