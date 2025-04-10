import os
import glob
import json
import csv
import re

def data_reader(file):
    with open(file, 'r') as json_file:
        try:
            data = json.load(json_file)
            return data
        except json.JSONDecodeError as e:
            print(f"Skipping {file}: {e}")
            return None

def produce_model_string(model_config):
    layers = []
    first_layer = None
    last_layer = None

    for layer in model_config:
        if layer['class_name'] not in ['Dense', 'QDense']:
            continue
        if first_layer is None:
            first_layer = layer['input_shape'][1]
        layers.append(str(layer['input_shape'][1]))
        last_layer = layer['output_shape'][1]

    layers.append(str(last_layer))
    # Ensure at least one layer was processed
    assert first_layer is not None and last_layer is not None

    return "-".join(layers), first_layer, last_layer

def process_json_file(file):
    data = data_reader(file)
    if data is None: # at least one of the JSONs in 'added_data_2layer' is empty
        return []
    # If data is a dictionary, wrap it in a list
    if isinstance(data, dict):
        data = [data]
    rows = []
    for data_point in data:
        meta_data = data_point['meta_data']
        model_name = meta_data['model_name']
        model_config = data_point['model_config']

        # Create a string for the model
        model_string, d_in, d_out = produce_model_string(model_config)

        # Process HLS config to get precision
        hls_config = data_point['hls_config']
        prec_data = hls_config["LayerName"]
        d = True
        for key in prec_data:
            if d:
                d = False
                continue
            prec = prec_data[key]["Precision"]['weight']
            break
        prec = re.sub(r",[0-9]+\>", "", prec)
        prec = re.sub(r"fixed\<", "", prec)

        model_file = meta_data['artifacts_file']
        rf = hls_config['Model']['ReuseFactor']
        strategy = hls_config['Model']['Strategy']

        # Extract latency values
        latency_report = data_point['latency_report']
        target_clock = latency_report['target_clock']
        estimated_clock = latency_report['estimated_clock']
        best_latency = latency_report['cycles_min']
        worst_latency = latency_report['cycles_max']

        # Extract resource values (handles alternative keys)
        resource_report = data_point['resource_report']
        bram = resource_report.get('BRAM', resource_report.get('bram'))
        dsp = resource_report.get('DSP', resource_report.get('dsp'))
        ff = resource_report.get('FF', resource_report.get('ff'))
        lut = resource_report.get('LUT', resource_report.get('lut'))
        uram = resource_report.get('URAM', resource_report.get('uram'))

        rf_times_precision = int(prec) * int(rf)
        hls_synth_success = "TRUE"

        row = [
            model_name, d_in, d_out, prec, model_file, model_string, rf, strategy,
            target_clock, estimated_clock, best_latency, worst_latency,
            best_latency, worst_latency, bram, dsp, ff, lut, uram, rf_times_precision, hls_synth_success
        ]
        rows.append(row)
    return rows

def concatenate_json_to_csv(json_folder, csv_filename):
    csv_header = [
        'model_name', 'd_in', 'd_out', 'prec', 'model_file', 'model_string', 'rf', 'strategy', 
        'TargetClockPeriod_hls', 'EstimatedClockPeriod_hls', 
        'BestLatency_hls', 'WorstLatency_hls', 'IntervalMin_hls', 'IntervalMax_hls', 
        'BRAM_18K_hls', 'DSP_hls', 'FF_hls', 'LUT_hls', 'URAM_hls', "rf_times_precision", "hls_synth_success"
    ]
    
    # Find all JSON files in the folder
    # Recursively find JSON files, but exclude any that are in a folder named 'raw_reports'
    json_files = [
        f for f in glob.glob(os.path.join(json_folder, "**/*.json"), recursive=True)
        if "raw_reports" not in f and f.endswith(".json")
    ]

    # # if it was just the JSON files in the directory, then it would be:
    # json_files = glob.glob(os.path.join(json_folder, "*.json"))

    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)
        for json_file in json_files:
            print(f"Processing {json_file}...")
            rows = process_json_file(json_file)
            for row in rows:
                writer.writerow(row)
    print(f"Parsing successful, file saved as {csv_filename}")

if __name__ == '__main__':
    # Update the folder path containing the JSON files
    json_folder = "../added_data_conv2d/conv2d_run_vsynth_2023-2"
    csv_filename = "concatenated_conv2d_json_to_csv.csv"
    concatenate_json_to_csv(json_folder, csv_filename)