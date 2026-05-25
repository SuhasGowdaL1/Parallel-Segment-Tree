#!/usr/bin/env python3
"""Capture Saleae Logic 2 logs and export them to disk.

This script connects to a running Logic 2 instance over the Automation API,
records a timed capture, and exports the result as a .sal file plus raw CSV
log files.

Examples:
  python capture_saleae_logs.py --duration 10 --output-dir ./captures
  python capture_saleae_logs.py --device-id F4241 --digital-channels 0,1,2,3

Prerequisite:
  Saleae Logic 2 must be running with Automation enabled.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import yaml

try:
    import saleae.automation as automation
except ImportError as exc:  # pragma: no cover - import guard for CLI usage
    raise SystemExit(
        "Missing dependency: saleae.automation. Install logic2-automation first."
    ) from exc


@dataclass(frozen=True)
class CaptureArtifacts:
    capture_path: Path
    raw_csv_dir: Path
    data_table_csv: Path


@dataclass(frozen=True)
class DigitalTransition:
    time_seconds: float
    channel_states: Dict[str, str]


@dataclass(frozen=True)
class AnalogSample:
    time_seconds: float
    channel_voltages: Dict[str, float]


@dataclass(frozen=True)
class VoltageChangeEvent:
    time_seconds: float
    previous_voltage: Optional[float]
    current_voltage: float
    delta_voltage: Optional[float] = None


@dataclass(frozen=True)
class SaleaeFileRecord:
    path: Path
    kind: str


@dataclass(frozen=True)
class SaleaeExportBundle:
    source_path: Path
    csv_files: Dict[str, List[Dict[str, str]]]
    binary_files: List[SaleaeFileRecord]
    other_files: List[SaleaeFileRecord]

    @property
    def has_raw_csv(self) -> bool:
        return any(name in self.csv_files for name in ("analog", "digital"))

    @property
    def has_binary(self) -> bool:
        return len(self.binary_files) > 0

    @property
    def has_data_table(self) -> bool:
        return any(name not in ("analog", "digital") for name in self.csv_files)


@dataclass(frozen=True)
class SaleaeCaptureConfig:
    digital_channels: List[int]
    analog_channels: List[int]
    digital_sample_rate: Optional[int]
    analog_sample_rate: Optional[int]
    digital_threshold_volts: Optional[float]
    channel_labels: "ChannelLabelConfig" = field(default_factory=lambda: ChannelLabelConfig())


@dataclass(frozen=True)
class ChannelLabelConfig:
    generic: Dict[str, str] = field(default_factory=dict)
    digital: Dict[str, str] = field(default_factory=dict)
    analog: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SaleaeWorkflowConfig:
    host: str
    port: int
    device_id: Optional[str]
    duration_seconds: float
    output_dir: Path
    name: str
    capture: SaleaeCaptureConfig
    export_raw_csv: bool = True
    export_data_table: bool = True


class SaleaeLogClient:
    """Reusable helper around the Saleae Automation API."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10430):
        self.host = host
        self.port = port
        self._manager: Optional[automation.Manager] = None

    def __enter__(self) -> "SaleaeLogClient":
        self._manager = automation.Manager.connect(address=self.host, port=self.port)
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def manager(self) -> automation.Manager:
        if self._manager is None:
            self._manager = automation.Manager.connect(address=self.host, port=self.port)
        return self._manager

    def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
            self._manager = None

    def list_devices(self, include_simulation_devices: bool = False):
        return self.manager.get_devices(include_simulation_devices=include_simulation_devices)

    def load_capture(self, capture_path: Path):
        return self.manager.load_capture(str(capture_path))

    def build_device_configuration(self, config: SaleaeCaptureConfig) -> automation.LogicDeviceConfiguration:
        if config.analog_channels and config.analog_sample_rate is None:
            raise ValueError("analog_sample_rate is required when analog_channels are enabled")

        return automation.LogicDeviceConfiguration(
            enabled_digital_channels=config.digital_channels,
            enabled_analog_channels=config.analog_channels,
            digital_sample_rate=config.digital_sample_rate if config.digital_channels else None,
            analog_sample_rate=config.analog_sample_rate,
            digital_threshold_volts=config.digital_threshold_volts if config.digital_channels else None,
        )

    def capture_timed(
        self,
        *,
        device_config: SaleaeCaptureConfig,
        duration_seconds: float,
        device_id: Optional[str] = None,
    ):
        capture_config = automation.CaptureConfiguration(
            capture_mode=automation.TimedCaptureMode(duration_seconds=duration_seconds)
        )
        return self.manager.start_capture(
            device_id=device_id,
            device_configuration=self.build_device_configuration(device_config),
            capture_configuration=capture_config,
        )

    def export_capture(
        self,
        capture,
        *,
        output_dir: Path,
        name: str,
        export_data_table: bool = True,
        export_raw_csv: bool = True,
    ) -> CaptureArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_export_dir = output_dir / f"{name}_raw"
        raw_export_dir.mkdir(parents=True, exist_ok=True)

        capture_path = output_dir / f"{name}.sal"
        data_table_path = output_dir / f"{name}.csv"

        capture.save_capture(filepath=str(capture_path))

        if export_raw_csv:
            capture.export_raw_data_csv(directory=str(raw_export_dir))

        if export_data_table:
            capture.export_data_table(filepath=str(data_table_path), analyzers=[])

        return CaptureArtifacts(
            capture_path=capture_path,
            raw_csv_dir=raw_export_dir,
            data_table_csv=data_table_path,
        )

    def parse_digital_csv(self, csv_path: Path) -> List[DigitalTransition]:
        return parse_digital_csv(csv_path, channel_labels=self.capture_channel_labels())

    def parse_analog_csv(self, csv_path: Path) -> List[AnalogSample]:
        return parse_analog_csv(csv_path, channel_labels=self.capture_channel_labels())

    def read_raw_export_directory(
        self,
        raw_export_dir: Path,
    ) -> Dict[str, List[Dict[str, str]]]:
        return read_raw_export_directory(raw_export_dir, channel_labels=self.capture_channel_labels())

    def detect_voltage_changes(
        self,
        samples: Sequence[AnalogSample],
        *,
        channel_name: str,
        delta_threshold: float = 0.0,
    ) -> List[VoltageChangeEvent]:
        return detect_voltage_changes(samples, channel_name=channel_name, delta_threshold=delta_threshold)

    def extract_channel_series(
        self,
        rows: Sequence[Dict[str, str]],
        *,
        channel_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Union[float, Dict[str, Union[str, float]]]]]:
        return extract_channel_series(
            rows,
            channel_names=channel_names,
            channel_labels=self.capture_channel_labels(),
        )

    def load_saleae_export(self, path: Path) -> SaleaeExportBundle:
        return load_saleae_export(path)

    def summarize_saleae_export(self, bundle: SaleaeExportBundle) -> Dict[str, Union[str, int, bool, List[str]]]:
        return summarize_saleae_export(bundle)

    def detect_row_changes(
        self,
        rows: Sequence[Dict[str, str]],
        *,
        key_column: str = "Time [s]",
    ) -> List[Dict[str, Union[float, Dict[str, Tuple[Optional[str], Optional[str]]]]]]:
        return detect_row_changes(rows, key_column=key_column)

    def capture_channel_labels(self) -> ChannelLabelConfig:
        return getattr(self, "_channel_labels", ChannelLabelConfig())

    def set_channel_labels(self, labels: ChannelLabelConfig) -> None:
        self._channel_labels = labels

    def capture_and_export(
        self,
        *,
        device_config: SaleaeCaptureConfig,
        duration_seconds: float,
        output_dir: Path,
        name: str = "saleae_capture",
        device_id: Optional[str] = None,
    ) -> CaptureArtifacts:
        with self.manager.start_capture(
            device_id=device_id,
            device_configuration=self.build_device_configuration(device_config),
            capture_configuration=automation.CaptureConfiguration(
                capture_mode=automation.TimedCaptureMode(duration_seconds=duration_seconds)
            ),
        ) as capture:
            capture.wait()
            return self.export_capture(capture, output_dir=output_dir, name=name)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(csv_path: Path) -> Iterator[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def read_raw_export_directory(raw_export_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    return read_raw_export_directory_with_labels(raw_export_dir)


def read_raw_export_directory_with_labels(
    raw_export_dir: Path,
    *,
    channel_labels: Optional[ChannelLabelConfig] = None,
) -> Dict[str, List[Dict[str, str]]]:
    parsed_exports: Dict[str, List[Dict[str, str]]] = {}
    for csv_path in sorted(raw_export_dir.glob("*.csv")):
        rows = read_csv_rows(csv_path)
        if channel_labels is not None:
            rows = [
                {
                    _remap_channel_name(column, channel_labels, "generic"): value
                    for column, value in row.items()
                }
                for row in rows
            ]
        parsed_exports[csv_path.stem] = rows
    return parsed_exports


def _coerce_str_map(value: object) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _load_channel_labels(raw_value: object) -> ChannelLabelConfig:
    if not isinstance(raw_value, dict):
        return ChannelLabelConfig()

    if "digital" in raw_value or "analog" in raw_value or "generic" in raw_value:
        return ChannelLabelConfig(
            generic=_coerce_str_map(raw_value.get("generic", {})),
            digital=_coerce_str_map(raw_value.get("digital", {})),
            analog=_coerce_str_map(raw_value.get("analog", {})),
        )

    return ChannelLabelConfig(generic=_coerce_str_map(raw_value))


def _coerce_int_list(value: object) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    return [int(value)]


def _remap_channel_name(name: str, labels: Optional[ChannelLabelConfig], kind: str) -> str:
    if labels is None:
        return name

    lookup = labels.generic.copy()
    if kind == "digital":
        lookup.update(labels.digital)
    elif kind == "analog":
        lookup.update(labels.analog)

    return lookup.get(name, name)


def load_workflow_config(config_path: Path) -> SaleaeWorkflowConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("YAML config must contain a mapping at the top level")

    saleae_section = raw_config.get("saleae", {})
    capture_section = raw_config.get("capture", {})
    labels_section = raw_config.get("channel_names", raw_config.get("channel_labels", {}))

    if not isinstance(saleae_section, dict) or not isinstance(capture_section, dict):
        raise ValueError("saleae and capture sections must be mappings")

    channel_labels = _load_channel_labels(labels_section)

    capture = SaleaeCaptureConfig(
        digital_channels=_coerce_int_list(capture_section.get("digital_channels", [])),
        analog_channels=_coerce_int_list(capture_section.get("analog_channels", [])),
        digital_sample_rate=capture_section.get("digital_sample_rate"),
        analog_sample_rate=capture_section.get("analog_sample_rate"),
        digital_threshold_volts=capture_section.get("digital_threshold_volts"),
        channel_labels=channel_labels,
    )

    return SaleaeWorkflowConfig(
        host=str(saleae_section.get("host", "127.0.0.1")),
        port=int(saleae_section.get("port", 10430)),
        device_id=saleae_section.get("device_id"),
        duration_seconds=float(capture_section.get("duration_seconds", 10.0)),
        output_dir=Path(capture_section.get("output_dir", "captures")),
        name=str(capture_section.get("name", "saleae_capture")),
        capture=capture,
        export_raw_csv=bool(capture_section.get("export_raw_csv", True)),
        export_data_table=bool(capture_section.get("export_data_table", True)),
    )


def detect_saleae_file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".bin":
        return "binary"
    if suffix == ".csv":
        if name == "analog.csv":
            return "raw_analog_csv"
        if name == "digital.csv":
            return "raw_digital_csv"
        return "csv"
    if suffix == ".sal":
        return "capture"
    return "other"


def list_saleae_export_files(path: Path) -> Tuple[List[SaleaeFileRecord], List[SaleaeFileRecord]]:
    binary_files: List[SaleaeFileRecord] = []
    other_files: List[SaleaeFileRecord] = []

    if path.is_file():
        record = SaleaeFileRecord(path=path, kind=detect_saleae_file_kind(path))
        if record.kind == "binary":
            binary_files.append(record)
        elif record.kind not in {"capture"}:
            other_files.append(record)
        return binary_files, other_files

    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        kind = detect_saleae_file_kind(child)
        record = SaleaeFileRecord(path=child, kind=kind)
        if kind == "binary":
            binary_files.append(record)
        elif kind != "capture":
            other_files.append(record)

    return binary_files, other_files


def load_saleae_export(path: Path) -> SaleaeExportBundle:
    if not path.exists():
        raise FileNotFoundError(path)

    csv_files: Dict[str, List[Dict[str, str]]] = {}
    binary_files, other_files = list_saleae_export_files(path)

    if path.is_file():
        if path.suffix.lower() == ".csv":
            csv_files[path.stem] = read_csv_rows(path)
    else:
        for csv_path in sorted(path.rglob("*.csv")):
            csv_files[csv_path.stem] = read_csv_rows(csv_path)

    return SaleaeExportBundle(
        source_path=path,
        csv_files=csv_files,
        binary_files=binary_files,
        other_files=other_files,
    )


def _parse_float(value: str) -> Optional[float]:
    text = value.strip()
    if not text:
        return None
    try:
        return float(Decimal(text))
    except (InvalidOperation, ValueError):
        return None


def _parse_time_seconds(row: Dict[str, str]) -> float:
    time_value = row.get("Time [s]", "").strip()
    if not time_value:
        raise ValueError("CSV row is missing a Time [s] value")
    return float(Decimal(time_value))


def parse_digital_csv_rows(
    rows: Sequence[Dict[str, str]],
    *,
    channel_labels: Optional[ChannelLabelConfig] = None,
) -> List[DigitalTransition]:
    return parse_digital_csv_rows_with_labels(rows, channel_labels=channel_labels)


def parse_digital_csv_rows_with_labels(
    rows: Sequence[Dict[str, str]],
    *,
    channel_labels: Optional[ChannelLabelConfig] = None,
) -> List[DigitalTransition]:
    transitions: List[DigitalTransition] = []
    previous_states: Optional[Dict[str, str]] = None

    for row in rows:
        current_states = {
            _remap_channel_name(column, channel_labels, "digital"): value.strip()
            for column, value in row.items()
            if column != "Time [s]" and value.strip() != ""
        }
        if previous_states is None or current_states != previous_states:
            transitions.append(
                DigitalTransition(
                    time_seconds=_parse_time_seconds(row),
                    channel_states=current_states,
                )
            )
            previous_states = current_states

    return transitions


def parse_digital_csv(
    csv_path: Path,
    *,
    channel_labels: Optional[ChannelLabelConfig] = None,
) -> List[DigitalTransition]:
    return parse_digital_csv_rows_with_labels(read_csv_rows(csv_path), channel_labels=channel_labels)


def parse_analog_csv_rows(
    rows: Sequence[Dict[str, str]],
    *,
    channel_labels: Optional[ChannelLabelConfig] = None,
) -> List[AnalogSample]:
    return parse_analog_csv_rows_with_labels(rows, channel_labels=channel_labels)


def parse_analog_csv_rows_with_labels(
    rows: Sequence[Dict[str, str]],
    *,
    channel_labels: Optional[ChannelLabelConfig] = None,
) -> List[AnalogSample]:
    samples: List[AnalogSample] = []

    for row in rows:
        channel_voltages = {
            column: voltage
            for column, value in row.items()
            if column != "Time [s]"
            and (voltage := _parse_float(value)) is not None
        }
        if channel_labels is not None:
            channel_voltages = {
                _remap_channel_name(column, channel_labels, "analog"): voltage
                for column, voltage in channel_voltages.items()
            }
        samples.append(
            AnalogSample(
                time_seconds=_parse_time_seconds(row),
                channel_voltages=channel_voltages,
            )
        )

    return samples


def parse_analog_csv(
    csv_path: Path,
    *,
    channel_labels: Optional[ChannelLabelConfig] = None,
) -> List[AnalogSample]:
    return parse_analog_csv_rows_with_labels(read_csv_rows(csv_path), channel_labels=channel_labels)


def summarize_saleae_export(bundle: SaleaeExportBundle) -> Dict[str, Union[str, int, bool, List[str]]]:
    return {
        "source_path": str(bundle.source_path),
        "csv_files": sorted(bundle.csv_files.keys()),
        "binary_files": [str(record.path) for record in bundle.binary_files],
        "other_files": [str(record.path) for record in bundle.other_files],
        "has_raw_csv": bundle.has_raw_csv,
        "has_binary": bundle.has_binary,
        "has_data_table": bundle.has_data_table,
    }


def detect_voltage_changes(
    samples: Sequence[AnalogSample],
    *,
    channel_name: str,
    delta_threshold: float = 0.0,
) -> List[VoltageChangeEvent]:
    events: List[VoltageChangeEvent] = []
    previous_voltage: Optional[float] = None

    for sample in samples:
        if channel_name not in sample.channel_voltages:
            continue

        current_voltage = sample.channel_voltages[channel_name]
        if previous_voltage is None:
            events.append(
                VoltageChangeEvent(
                    time_seconds=sample.time_seconds,
                    previous_voltage=None,
                    current_voltage=current_voltage,
                    delta_voltage=None,
                )
            )
            previous_voltage = current_voltage
            continue

        delta_voltage = current_voltage - previous_voltage
        if abs(delta_voltage) > delta_threshold:
            events.append(
                VoltageChangeEvent(
                    time_seconds=sample.time_seconds,
                    previous_voltage=previous_voltage,
                    current_voltage=current_voltage,
                    delta_voltage=delta_voltage,
                )
            )
            previous_voltage = current_voltage

    return events


def extract_channel_series(
    rows: Sequence[Dict[str, str]],
    *,
    channel_names: Optional[Sequence[str]] = None,
    channel_labels: Optional[ChannelLabelConfig] = None,
) -> List[Dict[str, Union[float, Dict[str, Union[str, float]]]]]:
    series: List[Dict[str, Union[float, Dict[str, Union[str, float]]]]] = []
    selected = set(channel_names) if channel_names is not None else None

    for row in rows:
        channel_values: Dict[str, Union[str, float]] = {}
        for column, value in row.items():
            if column == "Time [s]":
                continue
            renamed_column = _remap_channel_name(column, channel_labels, "generic")
            if selected is not None and renamed_column not in selected:
                continue

            numeric_value = _parse_float(value)
            channel_values[renamed_column] = numeric_value if numeric_value is not None else value.strip()

        series.append(
            {
                "time_seconds": _parse_time_seconds(row),
                "channels": channel_values,
            }
        )

    return series


def detect_row_changes(
    rows: Sequence[Dict[str, str]],
    *,
    key_column: str = "Time [s]",
) -> List[Dict[str, Union[float, Dict[str, Tuple[Optional[str], Optional[str]]]]]]:
    changes: List[Dict[str, Union[float, Dict[str, Tuple[Optional[str], Optional[str]]]]]] = []
    previous_row: Optional[Dict[str, str]] = None

    for row in rows:
        if previous_row is None:
            previous_row = row
            changes.append({"time_seconds": _parse_time_seconds(row), "changes": {}})
            continue

        changed_columns: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        columns = set(previous_row.keys()) | set(row.keys())
        for column in sorted(columns):
            if column == key_column:
                continue
            previous_value = previous_row.get(column)
            current_value = row.get(column)
            if previous_value != current_value:
                changed_columns[column] = (previous_value, current_value)

        if changed_columns:
            changes.append(
                {
                    "time_seconds": _parse_time_seconds(row),
                    "changes": changed_columns,
                }
            )

        previous_row = row

    return changes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture and export Saleae logs.")
    parser.add_argument("--config", type=Path, help="YAML config file for capture settings and channel names.")
    parser.add_argument("--host", default="127.0.0.1", help="Logic 2 host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=10430, help="Logic 2 automation port (default: 10430)")
    parser.add_argument("--device-id", help="Device serial number. Omit to use the first real device.")
    parser.add_argument("--duration", type=float, default=10.0, help="Capture duration in seconds (default: 10.0)")
    parser.add_argument("--output-dir", default="captures", help="Directory for exported files (default: ./captures)")
    parser.add_argument("--name", default="saleae_capture", help="Base name for exported files (default: saleae_capture)")
    parser.add_argument(
        "--digital-channels",
        default="0",
        help="Comma-separated digital channel indexes to capture (default: 0)",
    )
    parser.add_argument(
        "--analog-channels",
        default="",
        help="Comma-separated analog channel indexes to capture (default: none)",
    )
    parser.add_argument(
        "--digital-sample-rate",
        type=int,
        default=1_000_000,
        help="Digital sample rate in samples/second (default: 1000000)",
    )
    parser.add_argument(
        "--analog-sample-rate",
        type=int,
        default=None,
        help="Analog sample rate in samples/second. Required only when analog channels are enabled.",
    )
    parser.add_argument(
        "--threshold-volts",
        type=float,
        default=3.3,
        help="Digital threshold voltage (default: 3.3)",
    )
    return parser


def parse_channel_list(raw_value: str) -> List[int]:
    value = raw_value.strip()
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_device_configuration(args: argparse.Namespace) -> automation.LogicDeviceConfiguration:
    return SaleaeLogClient().build_device_configuration(
        SaleaeCaptureConfig(
            digital_channels=parse_channel_list(args.digital_channels),
            analog_channels=parse_channel_list(args.analog_channels),
            digital_sample_rate=args.digital_sample_rate,
            analog_sample_rate=args.analog_sample_rate,
            digital_threshold_volts=args.threshold_volts,
        )
    )


def main() -> int:
    args = build_parser().parse_args()

    workflow_config: Optional[SaleaeWorkflowConfig] = None
    if hasattr(args, "config") and args.config is not None:
        workflow_config = load_workflow_config(args.config)

    host = args.host if args.host != "127.0.0.1" or workflow_config is None else workflow_config.host
    port = args.port if args.port != 10430 or workflow_config is None else workflow_config.port
    device_id = args.device_id if args.device_id is not None else (workflow_config.device_id if workflow_config else None)
    output_dir = Path(args.output_dir if args.output_dir != "captures" or workflow_config is None else workflow_config.output_dir).expanduser().resolve()
    name = args.name if args.name != "saleae_capture" or workflow_config is None else workflow_config.name
    duration_seconds = args.duration if args.duration != 10.0 or workflow_config is None else workflow_config.duration_seconds

    if workflow_config is not None:
        digital_channels = parse_channel_list(args.digital_channels) if args.digital_channels != "0" else workflow_config.capture.digital_channels
        analog_channels = parse_channel_list(args.analog_channels) if args.analog_channels != "" else workflow_config.capture.analog_channels
        digital_sample_rate = args.digital_sample_rate if args.digital_sample_rate != 1_000_000 else workflow_config.capture.digital_sample_rate
        analog_sample_rate = args.analog_sample_rate if args.analog_sample_rate is not None else workflow_config.capture.analog_sample_rate
        threshold_volts = args.threshold_volts if args.threshold_volts != 3.3 else workflow_config.capture.digital_threshold_volts
        export_raw_csv = workflow_config.export_raw_csv
        export_data_table = workflow_config.export_data_table
        channel_labels = workflow_config.capture.channel_labels
    else:
        digital_channels = parse_channel_list(args.digital_channels)
        analog_channels = parse_channel_list(args.analog_channels)
        digital_sample_rate = args.digital_sample_rate
        analog_sample_rate = args.analog_sample_rate
        threshold_volts = args.threshold_volts
        export_raw_csv = True
        export_data_table = True
        channel_labels = ChannelLabelConfig()

    client = SaleaeLogClient(host=host, port=port)
    device_config = SaleaeCaptureConfig(
        digital_channels=digital_channels,
        analog_channels=analog_channels,
        digital_sample_rate=digital_sample_rate,
        analog_sample_rate=analog_sample_rate,
        digital_threshold_volts=threshold_volts,
        channel_labels=channel_labels,
    )

    with client:
        client.set_channel_labels(channel_labels)
        with client.manager.start_capture(
            device_id=device_id,
            device_configuration=client.build_device_configuration(device_config),
            capture_configuration=automation.CaptureConfiguration(
                capture_mode=automation.TimedCaptureMode(duration_seconds=duration_seconds)
            ),
        ) as capture:
            capture.wait()
            artifacts = client.export_capture(
                capture,
                output_dir=output_dir,
                name=name,
                export_raw_csv=export_raw_csv,
                export_data_table=export_data_table,
            )

    print(f"Saved capture to {artifacts.capture_path}")
    print(f"Exported raw CSV logs to {artifacts.raw_csv_dir}")
    print(f"Exported data table to {artifacts.data_table_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
