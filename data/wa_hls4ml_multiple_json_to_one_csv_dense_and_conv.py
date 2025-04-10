import os
import glob
import json
import csv
import re

def data_reader(file):
    with open(file, 'r') as json_file:
        try:
            return json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Skipping {file}: {e}")
            return None

def produce_model_string(model_config):
    """
    Build a hyphen‑delimited string of layer sizes.
    Handles both QConv2D (uses output filters) and Dense/QDense (uses units).
    Returns (model_string, first_size, last_size).
    """
    layers = []
    first_layer = None
    last_layer = None

    for layer in model_config:
        cls = layer['class_name']
        # for QConv2D layers
        if cls == 'QConv2D':
            # input_shape = [batch, H, W, in_channels]
            # output_shape = [batch, H', W', out_channels]
            in_ch  = layer['input_shape'][3]
            out_ch = layer['output_shape'][3]
            if first_layer is None:
                first_layer = in_ch
            layers.append(str(in_ch))
            last_layer = out_ch

        # for Dense and QDense
        elif cls in ('Dense', 'QDense'):
            # input_shape = [batch, in_units]
            # output_shape = [batch, out_units]
            in_units  = layer['input_shape'][1]
            out_units = layer['output_shape'][1]
            if first_layer is None:
                first_layer = in_units
            layers.append(str(in_units))
            last_layer = out_units

        # skip everything else (activations, flatten, etc)
        else:
            continue

    # append the final layer size
    layers.append(str(last_layer))
    assert first_layer is not None and last_layer is not None, "No Conv2D or Dense layers found"
    return "-".join(layers), first_layer, last_layer

# def process_json_file(file):
#     data = data_reader(file)
#     if data is None:
#         return []
#     if isinstance(data, dict):
#         data = [data]

#     rows = []
#     for data_point in data:
#         meta_data    = data_point['meta_data']
#         model_name   = meta_data['model_name']
#         model_config = data_point['model_config']

#         # build model string and sizes
#         model_string, d_in, d_out = produce_model_string(model_config)

#         # hls precision: get the first layer’s weight precision
#         hls_config = data_point['hls_config']
#         prec_data  = hls_config["LayerName"]

#         # pull out the model level default: "fixed<16,6>" (using this as a sub for 'auto')
#         default_str = hls_config['Model']['Precision']['default']
#         # extract the integer bit width before the comma: "16"
#         default_bits = default_str.replace('fixed<','').rstrip('>').split(',')[0]

#         # skip the input_ entry, then take the first real layer
#         for key in prec_data: # changes here to work for 'auto'
#             # prec = prec_data[key]["Precision"].get('weight') or prec_data[key]["Precision"].get('result')
#             # # get rid of stuff like "fixed<16,1>"
#             # prec = re.sub(r",(?:[0-9]+)\>", ">", prec)
#             # prec = re.sub(r"fixed\<", "", prec).rstrip(">")
#             raw_prec = prec_data[key]["Precision"].get('weight') \
#                     or prec_data[key]["Precision"].get('result')
#             if raw_prec == 'auto':
#                 # back to the model default bit width
#                 prec = default_bits
#             else:
#                 # strip the "fixed<" and ",<int>>"
#                 prec = re.sub(r",(?:[0-9]+)\>", ">", raw_prec)
#                 prec = re.sub(r"fixed\<", "", prec).rstrip(">")
#             break

#         model_file = meta_data['artifacts_file']
#         rf         = hls_config['Model']['ReuseFactor']
#         strategy   = hls_config['Model']['Strategy']

#         # latency
#         lr = data_point['latency_report']
#         target_clock    = lr['target_clock']
#         estimated_clock = lr['estimated_clock']
#         best_latency    = lr['cycles_min']
#         worst_latency   = lr['cycles_max']

#         # resources
#         rr   = data_point['resource_report']
#         bram = rr.get('BRAM', rr.get('bram'))
#         dsp  = rr.get('DSP', rr.get('dsp'))
#         ff   = rr.get('FF',  rr.get('ff'))
#         lut  = rr.get('LUT', rr.get('lut'))
#         uram = rr.get('URAM',rr.get('uram'))

#         rf_times_precision = int(prec) * int(rf)
#         hls_synth_success  = "TRUE"

#         row = [
#             model_name,
#             d_in,
#             d_out,
#             prec,
#             model_file,
#             model_string,
#             rf,
#             strategy,
#             target_clock,
#             estimated_clock,
#             best_latency,
#             worst_latency,
#             best_latency,       # IntervalMin_hls
#             worst_latency,      # IntervalMax_hls
#             bram,
#             dsp,
#             ff,
#             lut,
#             uram,
#             rf_times_precision,
#             hls_synth_success
#         ]
#         rows.append(row)
#     return rows

def process_json_file(file):
    data = data_reader(file)
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]

    rows = []
    for data_point in data:
        meta_data    = data_point['meta_data']
        model_name   = meta_data['model_name']
        model_config = data_point['model_config']

        # 1) Build model_string (for backward compatibility) and d_in/d_out of first & last layer
        model_string, d_in, d_out = produce_model_string(model_config)

        # 2) Extract precision, falling back on model default if 'auto'
        hls_config  = data_point['hls_config']
        prec_data   = hls_config["LayerName"]
        default_str = hls_config['Model']['Precision']['default']         # e.g. "fixed<16,6>"
        default_bits= default_str.replace('fixed<','').rstrip('>').split(',')[0]  # "16"

        for key in prec_data:
            raw_prec = prec_data[key]["Precision"].get('weight') \
                     or prec_data[key]["Precision"].get('result')
            if raw_prec == 'auto':
                prec = default_bits
            else:
                prec = re.sub(r",(?:[0-9]+)\>", ">", raw_prec)
                prec = re.sub(r"fixed\<", "", prec).rstrip(">")
            break

        # 3) Other global fields
        model_file = meta_data['artifacts_file']
        rf         = hls_config['Model']['ReuseFactor']
        strategy   = hls_config['Model']['Strategy']

        lr = data_point['latency_report']
        target_clock    = lr['target_clock']
        estimated_clock = lr['estimated_clock']
        best_latency    = lr['cycles_min']
        worst_latency   = lr['cycles_max']

        rr   = data_point['resource_report']
        bram = rr.get('BRAM', rr.get('bram'))
        dsp  = rr.get('DSP', rr.get('dsp'))
        ff   = rr.get('FF',  rr.get('ff'))
        lut  = rr.get('LUT', rr.get('lut'))
        uram = rr.get('URAM',rr.get('uram'))

        rf_times_precision = int(prec) * int(rf)
        hls_synth_success  = "TRUE"

        # ─── Unroll up to 3 real layers into flat features ────────────────────
        max_layers = 3
        real_layers = [
            L for L in model_config
            if L['class_name'] in (
                'QConv2D','Conv2D','QConv1D','Conv1D',
                'QDense','Dense'
            )
        ]

        layer_type_map = {
          'Dense':0,'QDense':0,
          'Conv1D':1,'QConv1D':1,
          'Conv2D':2,'QConv2D':2,
          'SeparableConv1D':3,'SeparableConv2D':4,
          'DepthwiseConv1D':5,'DepthwiseConv2D':6,
          'Flatten':7,'MaxPool2D':8,'AveragePool2D':9
        }
        activation_map = {
          None:0,'linear':0,'relu':1,'tanh':2,'sigmoid':3,'softmax':4
        }

        d_ins=[]; d_outs=[]
        l_types=[]; a_types=[]
        filts=[]; k_sizes=[]; strs=[]; pads=[]; pools=[]

        for i in range(max_layers):
            if i < len(real_layers):
                L   = real_layers[i]
                cls = L['class_name']
                inp = L['input_shape']
                out = L['output_shape']

                # in/out units (channels for conv, units for dense)
                if 'Conv' in cls:
                    d_ins .append(inp[-1])
                    d_outs.append(out[-1])
                else:
                    d_ins .append(inp[1])
                    d_outs.append(out[1])

                # layer type code
                l_types.append(layer_type_map.get(cls, 0))

                # activation code: scan ahead for next Activation/QActivation
                act_code = 0
                idx = model_config.index(L)
                for nxt in model_config[idx+1:]:
                    if nxt['class_name'] in ('Activation','QActivation'):
                        act_str = nxt.get('activation', None)
                        act_code = activation_map.get(act_str, 0)
                        break
                a_types.append(act_code)

                # conv params or zeros
                cfg = L.get('config', {})
                if 'Conv' in cls:
                    filts.append(out[-1])
                    k_sizes.append(cfg.get('kernel_size',[0])[0])
                    strs   .append(cfg.get('strides',    [0])[0])
                    pads   .append(0 if cfg.get('padding','valid')=='same' else 1)
                    pools  .append(0)
                else:
                    filts.append(0); k_sizes.append(0)
                    strs  .append(0); pads.append(0); pools.append(0)
            else:
                # pad missing layers
                d_ins .append(0); d_outs.append(0)
                l_types.append(0); a_types.append(0)
                filts.append(0); k_sizes.append(0)
                strs.append(0); pads.append(0); pools.append(0)
        # ────────────────────────────────────────────────────────────────────────

        # 4) Build the final row
        row = [
            model_name,
            *d_ins, *d_outs,
            prec, model_file, model_string, rf, strategy,
            *l_types, *a_types,
            *filts, *k_sizes, *strs, *pads, *pools,
            target_clock, estimated_clock,
            best_latency, worst_latency,
            best_latency, worst_latency,
            bram, dsp, ff, lut, uram,
            rf_times_precision,
            hls_synth_success
        ]
        rows.append(row)

    return rows

# def concatenate_json_to_csv(json_folder, csv_filename):
#     csv_header = [
#         'model_name',
#         'd_in',
#         'd_out',
#         'prec',
#         'model_file',
#         'model_string',
#         'rf',
#         'strategy',
#         'TargetClockPeriod_hls',
#         'EstimatedClockPeriod_hls',
#         'BestLatency_hls',
#         'WorstLatency_hls',
#         'IntervalMin_hls',
#         'IntervalMax_hls',
#         'BRAM_18K_hls',
#         'DSP_hls',
#         'FF_hls',
#         'LUT_hls',
#         'URAM_hls',
#         'rf_times_precision',
#         'hls_synth_success'
#     ]

def concatenate_json_to_csv(json_folder, csv_filename):
    max_layers = 3
    csv_header = [
        'model_name',
        # per‐layer in/out channels or units
        *[f'd_in{i+1}'  for i in range(max_layers)],
        *[f'd_out{i+1}' for i in range(max_layers)],
        # global model‐level features
        'prec','rf','strategy','rf_times_precision',
        # per‐layer categorical/conv params
        *[f'layer_type{i+1}'      for i in range(max_layers)],
        *[f'activation_type{i+1}' for i in range(max_layers)],
        *[f'filters{i+1}'         for i in range(max_layers)],
        *[f'kernel_size{i+1}'     for i in range(max_layers)],
        *[f'stride{i+1}'          for i in range(max_layers)],
        *[f'padding{i+1}'         for i in range(max_layers)],
        *[f'pooling{i+1}'         for i in range(max_layers)],
        # existing metrics
        'TargetClockPeriod_hls','EstimatedClockPeriod_hls',
        'BestLatency_hls','WorstLatency_hls','IntervalMin_hls','IntervalMax_hls',
        'BRAM_18K_hls','DSP_hls','FF_hls','LUT_hls','URAM_hls',
        'hls_synth_success'
    ]

    # find all JSONs, excluding raw_reports
    json_files = [
        f for f in glob.glob(os.path.join(json_folder, "**/*.json"), recursive=True)
        if "raw_reports" not in f and f.endswith(".json")
    ]

    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)
        for jf in json_files:
            print(f"Processing {jf}...")
            for row in process_json_file(jf):
                writer.writerow(row)

    print(f"Parsing successful, file saved as {csv_filename}")

if __name__ == '__main__':
    # point to conv2d JSON directory
    json_folder = "../added_data_conv2d/conv2d_run_vsynth_2023-2"
    csv_filename = "concatenated_conv2d_json_to_csv.csv"
    concatenate_json_to_csv(json_folder, csv_filename)