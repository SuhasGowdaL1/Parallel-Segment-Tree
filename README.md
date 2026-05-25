# Saleae Log Capturer

Small Python CLI for recording a Saleae Logic 2 capture and exporting the result.

## Requirements

- Python 3.8+
- Saleae Logic 2 running with Automation enabled
- `logic2-automation` installed in the active environment
- `PyYAML` for YAML-based configuration files

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python capture_saleae_logs.py --duration 10 --output-dir ./captures --digital-channels 0,1,2,3
```

If you want to target a specific device, pass `--device-id` with the Saleae serial number.

## YAML Config

You can drive both capture settings and channel names from a YAML file:

```yaml
saleae:
	host: 127.0.0.1
	port: 10430
	device_id: F4241

capture:
	duration_seconds: 10
	output_dir: captures
	name: demo_capture
	digital_channels: [0, 1, 2, 3]
	analog_channels: []
	digital_sample_rate: 1000000
	analog_sample_rate: null
	digital_threshold_volts: 3.3
	export_raw_csv: true
	export_data_table: true

channel_names:
	digital:
		"0": CLK
		"1": MOSI
		"2": MISO
		"3": CS
	analog:
		"0": VBUS
```

Run it with:

```bash
python capture_saleae_logs.py --config saleae.yaml
```