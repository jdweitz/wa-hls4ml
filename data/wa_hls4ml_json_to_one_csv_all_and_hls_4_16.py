import json, csv, re, os, glob

import orjson # much faster than json module
def data_reader(fn):
    try:
        return orjson.loads(open(fn, 'rb').read())
    except:
        return None

# def data_reader(fn):
#     try:
#         with open(fn) as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"Skipping {fn}: {e}")
#         return None

def extract_model_features(model_config, hls_config):
    # defaults & HLS globals
    d_in1 = d_in2 = d_in3 = 0
    d_out1 = d_out2 = d_out3 = filters = 0

    rf       = hls_config["Model"]["ReuseFactor"]
    strategy = hls_config["Model"]["Strategy"]

    # Model.Precision might be a dict {"default": "..."} or a string "fixed<...>"
    prec_cfg = hls_config["Model"]["Precision"]
    if isinstance(prec_cfg, dict):
        default_prec = prec_cfg.get("default", "")
    else:
        default_prec = prec_cfg

    m = re.search(r'fixed<(\d+),', default_prec)
    bitwidth = int(m.group(1)) if m else 0
    rf_times_precision = rf * bitwidth

    # do we have any conv?
    conv_layers = [L for L in model_config if L['class_name'].startswith("QConv")]
    if conv_layers:
        conv = conv_layers[0]
        inp = next(L for L in model_config if L['class_name']=="InputLayer")['input_shape'][0]

        if len(inp) == 4:        # Conv2D
            _, H, W, C = inp
            d_in1, d_in2, d_in3 = H, W, C
            layer_type = 2
        else:                    # Conv1D
            _, L, C = inp
            d_in1, d_in2, d_in3 = 0, L, C
            layer_type = 1

        out = conv['output_shape']
        if len(out) == 4:
            _, H2, W2, F = out
            d_out1, d_out2, d_out3, filters = H2, W2, F, F
        else:
            _, L2, F = out
            d_out1, d_out2, d_out3, filters = 0, L2, F, F

        # override precision if specified in LayerName
        for key, info in hls_config["LayerName"].items():
            if key.lower().startswith("q_conv"):
                pi = info["Precision"]
                w = pi.get("weight")
                r = pi.get("result")

                raw = None
                if w and w != "auto":
                    raw = w
                elif r and r != "auto":
                    raw = r

                if raw:
                    m2 = re.search(r'fixed<(\d+),', raw)
                    if m2:
                        bitwidth = int(m2.group(1))
                        rf_times_precision = rf * bitwidth
                break

        # activation
        activation_type = 0
        idx = model_config.index(conv)
        for L in model_config[idx+1:]:
            if L['class_name'] in ("Activation","QActivation"):
                a = L.get("activation","linear")
                activation_type = {
                    "linear":0, "relu":1, "tanh":2,
                    "sigmoid":3, "softmax":4
                }[ next(k for k in ("linear","relu","tanh","sigmoid","softmax") if k in a) ]
                break

        kernel_size = stride = padding = pooling = 0

    else:
        # dense path
        widths = [L['input_shape'][1] for L in model_config
                  if L['class_name'] in ("Dense","QDense")]
        d, dout = widths[0], widths[-1]
        d_in1, d_in2, d_in3 = 0, d, 0
        d_out1, d_out2, d_out3 = 0, dout, 0
        filters = 0
        layer_type = 0
        activation_type = 0
        kernel_size = stride = padding = pooling = 0

    return {
        "d_in1": d_in1, "d_in2": d_in2, "d_in3": d_in3,
        "d_out1": d_out1, "d_out2": d_out2, "d_out3": d_out3,
        "prec": bitwidth, "rf": rf, "strategy": strategy,
        "rf_times_precision": rf_times_precision,
        "layer_type": layer_type, "activation_type": activation_type,
        "filters": filters, "kernel_size": kernel_size,
        "stride": stride, "padding": padding, "pooling": pooling
    }

def parse_file(fn):
    data = data_reader(fn)
    if not data:
        return None

    # SKIP FAILED RUNS
    rr = data.get("resource_report", {})
    if not rr:
        print(f"Skipping {fn} (empty resource_report)")
        return None
    # get the HLS estimates
    hrr = data.get("hls_resource_report", {})

    md = data["meta_data"]
    mc = data["model_config"]
    hc = data["hls_config"]
    lat= data["latency_report"]

    f = extract_model_features(mc, hc)
    ms = f"{f['d_in1']}-{f['d_in2']}-{f['d_in3']}-"\
         f"{f['d_out1']}-{f['d_out2']}-{f['d_out3']}"

    row = [
      md["model_name"],
      f["d_in1"], f["d_in2"], f["d_in3"],
      f["d_out1"], f["d_out2"], f["d_out3"],
      f["prec"], md["artifacts_file"], ms,
      f["rf"], f["strategy"],
      lat["target_clock"], lat["estimated_clock"],
      lat["cycles_min"], lat["cycles_max"],
      lat["cycles_min"], lat["cycles_max"],
      # original resource_report fields
      rr.get("bram", 0),
      rr.get("dsp", 0),
      rr.get("ff",  0),
      rr.get("lut", 0),
      rr.get("uram",0),
      # new HLS‐estimate fields
      hrr.get("bram", 0),
      hrr.get("dsp", 0),
      hrr.get("ff",  0),
      hrr.get("lut", 0),
      hrr.get("uram",0),
      # back to the rest
      f["rf_times_precision"], f["layer_type"],
      f["activation_type"], f["filters"],
      f["kernel_size"], f["stride"], f["padding"],
      f["pooling"], "TRUE"
    ]
    return row

def concatenate_json_to_csv(folder, outcsv):
    hdr = [
      'model_name','d_in1','d_in2','d_in3',
      'd_out1','d_out2','d_out3','prec','model_file',
      'model_string','rf','strategy',
      'TargetClockPeriod_hls','EstimatedClockPeriod_hls',
      'BestLatency_hls','WorstLatency_hls',
      'IntervalMin_hls','IntervalMax_hls',
      # original tool outputs
      'BRAM_18K_hls','DSP_hls','FF_hls','LUT_hls','URAM_hls',
      # HLS estimates
      'BRAM_hls_est','DSP_hls_est','FF_hls_est','LUT_hls_est','URAM_hls_est',
      # rest of the features
      'rf_times_precision','layer_type','activation_type',
      'filters','kernel_size','stride','padding',
      'pooling','hls_synth_success'
    ]

    files = [
      f for f in glob.glob(os.path.join(folder,"**/*.json"), recursive=True)
      if "raw_reports" not in f
    ]

    with open(outcsv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for fn in files:
            print("→", fn)
            r = parse_file(fn)
            if r:
                w.writerow(r)
    print("Done:", outcsv)

if __name__=="__main__":
    concatenate_json_to_csv("4_16", "all_models_with_estimates.csv")