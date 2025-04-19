import json
import csv
import re
import os
import glob

def data_reader(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            if data is None:
                print(f"Skipping null JSON: {file}")
            return data
    except json.JSONDecodeError as e:
        print(f"Skipping {file} (JSON error: {e})")
        return None

def extract_conv_features(model_config, hls_config):
    # ── 1) Input dims from InputLayer ─────────────────────────────────────
    d_in1 = d_in2 = d_in3 = 0
    for layer in model_config:
        if layer['class_name'] == "InputLayer":
            shape = layer['input_shape'][0]      # e.g. [null, H, W, C]
            d_in1, d_in2, d_in3 = shape[1], shape[2], shape[3]
            break

    # ── 2) Output dims + filters from first QConv2D ──────────────────────
    d_out1 = d_out2 = d_out3 = filters = 0
    conv_layer = None
    for layer in model_config:
        if layer['class_name'] == "QConv2D":
            out_shape = layer['output_shape']    # e.g. [null, H', W', F]
            d_out1, d_out2, d_out3 = out_shape[1], out_shape[2], out_shape[3]
            filters = out_shape[3]
            conv_layer = layer
            break

    # ── 3) Global HLS settings ────────────────────────────────────────────
    mh = hls_config["Model"]
    default_prec_str = mh["Precision"]["default"]   # "fixed<16,6>"
    rf       = mh["ReuseFactor"]
    strategy = mh["Strategy"]

    # ── 4) Extract bitwidth for the conv weights ─────────────────────────
    prec_str = default_prec_str
    for key, info in hls_config["LayerName"].items():
        if "q_conv2d" in key:
            raw = info["Precision"].get("weight") or info["Precision"].get("result")
            if raw and raw != "auto":
                prec_str = raw
            break
    # pull out the integer bitwidth
    m = re.search(r'fixed<(\d+),', prec_str)
    bitwidth = int(m.group(1)) if m else int(re.search(r'fixed<(\d+),', default_prec_str).group(1))
    rf_times_precision = rf * bitwidth

    # ── 5) layer_type code ────────────────────────────────────────────────
    layer_type = 2  # QConv2D → conv2d

    # ── 6) activation_type code ──────────────────────────────────────────
    activation_type = 0
    if conv_layer:
        idx = model_config.index(conv_layer)
        for L in model_config[idx+1:]:
            if L['class_name'] in ("Activation","QActivation"):
                act = L.get("activation","linear")
                if act == "linear":
                    activation_type = 0
                elif "relu"  in act:
                    activation_type = 1
                elif "tanh"  in act:
                    activation_type = 2
                elif "sigmoid" in act:
                    activation_type = 3
                elif "softmax" in act:
                    activation_type = 4
                break

    # ── 7) conv params defaults ───────────────────────────────────────────
    kernel_size = 0
    stride      = 0
    padding     = 0  # same=0, valid=1
    pooling     = 0  # pooling layer? 2 otherwise 0

    return {
        "d_in1": d_in1, "d_in2": d_in2, "d_in3": d_in3,
        "d_out1": d_out1, "d_out2": d_out2, "d_out3": d_out3,
        "prec": bitwidth,
        "rf": rf,
        "strategy": strategy,
        "rf_times_precision": rf_times_precision,
        "layer_type": layer_type,
        "activation_type": activation_type,
        "filters": filters,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "pooling": pooling
    }

def process_conv_file(file):
    data = data_reader(file)
    if data is None:
        return None

    # Unpack JSON
    md = data["meta_data"]
    mc = data["model_config"]
    hc = data["hls_config"]
    lat = data["latency_report"]
    res = data["resource_report"]

    # Extract conv features
    f = extract_conv_features(mc, hc)

    # Build model_string (optional—you can replicate your dense logic if desired)
    # Here we’ll just note the conv dims: "H-W-C → H'-W'-F"
    model_string = f"{f['d_in1']}-{f['d_in2']}-{f['d_in3']}-{f['d_out1']}-{f['d_out2']}-{f['d_out3']}"

    # Build the row in your dense‐like format + new features
    row = [
        md["model_name"],      # model_name
        f["d_in1"], f["d_in2"], f["d_in3"],  # d_in1..3
        f["d_out1"], f["d_out2"], f["d_out3"],  # d_out1..3
        f["prec"],             # prec (numeric bitwidth)
        md["artifacts_file"],  # model_file
        model_string,          # model_string
        f["rf"],                # rf
        f["strategy"],          # strategy
        lat["target_clock"],    # TargetClockPeriod_hls
        lat["estimated_clock"], # EstimatedClockPeriod_hls
        lat["cycles_min"],      # BestLatency_hls
        lat["cycles_max"],      # WorstLatency_hls
        lat["cycles_min"],      # IntervalMin_hls
        lat["cycles_max"],      # IntervalMax_hls
        res.get("bram", res.get("BRAM")),   # BRAM_18K_hls
        res.get("dsp", res.get("DSP")),     # DSP_hls
        res.get("ff",  res.get("FF")),      # FF_hls
        res.get("lut", res.get("LUT")),     # LUT_hls
        res.get("uram",res.get("URAM")),    # URAM_hls
        f["rf_times_precision"], # rf_times_precision
        f["layer_type"],         # layer_type
        f["activation_type"],    # activation_type
        f["filters"],            # filters
        f["kernel_size"],        # kernel_size
        f["stride"],             # stride
        f["padding"],            # padding
        f["pooling"],            # pooling
        "TRUE"                   # hls_synth_success
    ]
    return row

def concatenate_conv_json_to_csv(json_folder, csv_filename):
    # Gather JSONs
    json_files = [
        f for f in glob.glob(os.path.join(json_folder, "**/*.json"), recursive=True)
        if "raw_reports" not in f and f.endswith(".json")
    ]

    # Header: dense format + new conv features
    header = [
        'model_name',
        'd_in1','d_in2','d_in3',
        'd_out1','d_out2','d_out3',
        'prec','model_file','model_string','rf','strategy',
        'TargetClockPeriod_hls','EstimatedClockPeriod_hls',
        'BestLatency_hls','WorstLatency_hls','IntervalMin_hls','IntervalMax_hls',
        'BRAM_18K_hls','DSP_hls','FF_hls','LUT_hls','URAM_hls',
        'rf_times_precision',
        'layer_type','activation_type','filters',
        'kernel_size','stride','padding','pooling',
        'hls_synth_success'
    ]

    with open(csv_filename, 'w', newline='') as out:
        w = csv.writer(out)
        w.writerow(header)
        for jf in json_files:
            print("Processing", jf)
            row = process_conv_file(jf)
            if row:
                w.writerow(row)
    print("Done — saved to", csv_filename)

if __name__ == '__main__':
    folder = "../added_data_conv2d/conv2d_run_vsynth_2023-2"
    outcsv = "conv2d_combined_dense_format.csv"
    concatenate_conv_json_to_csv(folder, outcsv)