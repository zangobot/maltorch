"""
Trizna, D. and Demetrio, L. and Battista, B. and Fabio, R. (2024)
Nebula: Self-Attention for Dynamic Malware Analysis
IEEE Transactions on Information Forensics and Security (TIFS)
Paper: https://ieeexplore.ieee.org/document/10551436
Source: https://github.com/dtrizna/nebula/blob/main/nebula/preprocessing/pe.py
"""

import os
import json
import logging
from time import time
from pathlib import Path
from typing import Union

import speakeasy
from maltorch.utils.strings import get_alphanum_chars

from .constants import (
    SPEAKEASY_RECORD_FIELDS,
    SPEAKEASY_RECORD_LIMITS
)
from .tokenization import JSONFilter
from .normalization import normalize_table_ip, normalize_table_path


class PEDynamicFeatureExtractor:
    def __init__(
        self,
        speakeasy_config=None,
        speakeasy_record_fields=SPEAKEASY_RECORD_FIELDS,
        record_limits=SPEAKEASY_RECORD_LIMITS,
        emulation_output_folder=None
    ):
        # Setup Speakeasy config
        if speakeasy_config is None:
            speakeasy_config = os.path.join(os.path.dirname(__file__), "speakeasy_config.json")
        if isinstance(speakeasy_config, dict):
            self.speakeasy_config = speakeasy_config
        else:
            if not os.path.exists(speakeasy_config):
                raise FileNotFoundError(f"Speakeasy config file not found: {speakeasy_config}")
            with open(speakeasy_config, "r") as f:
                self.speakeasy_config = json.load(f)

        self.record_limits = record_limits
        self.parser = JSONFilter(fields=speakeasy_record_fields)

        self.output_folder = emulation_output_folder
        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)

    def _create_error_file(self, errfile):
        # Just creating an empty file to indicate failure
        Path(errfile).touch()

    def _emulation(self, config: dict, data: bytes, path: str = None):
        try:
            file_name = path if path else str(data[:15])
            se = speakeasy.Speakeasy(config=config)
            if path:
                module = se.load_module(path=path)
            elif data:
                module = se.load_module(data=data)
            else:
                raise ValueError("[-] Either 'path' or 'data' must be provided to load_module.")
            se.run_module(module)
            return se.get_report()
        except Exception as ex:
            logging.error(f"[-] Failed emulation of {file_name} | Exception:\n{ex}")
            return None

    def emulate(self, raw_pe: Union[str, bytes]):
        if isinstance(raw_pe, str):
            if os.path.exists(raw_pe):
                with open(raw_pe, "rb") as f:
                    raw_pe = f.read()
                sample_name = os.path.splitext(os.path.basename(raw_pe))[0]
            else:
                raise FileNotFoundError(f"[-] File not found: {raw_pe}")
        else:
            sample_name = f"{int(time())}"
        report = self._emulation(self.speakeasy_config, data=raw_pe, path=sample_name)

        if self.output_folder:
            if report:
                output_path = os.path.join(self.output_folder, f"{self.sample_name}.json")
                with open(output_path, "w") as f:
                    json.dump(report, f, indent=4)
            else:
                err_file = os.path.join(self.output_folder, f"{self.sample_name}.err")
                self._create_error_file(err_file)

        if report and 'entry_points' in report:
            api_seq_len = sum(len(entry.get("apis", [])) for entry in report["entry_points"])
        else:
            api_seq_len = 0
        return (
            self.filter_and_normalize_report(report["entry_points"])
            if api_seq_len > 0 else None
        )

    def filter_and_normalize_report(self, entry_points):
        if isinstance(entry_points, str) and os.path.exists(entry_points):
            with open(entry_points, "r") as f:
                entry_points = json.load(f)
        # Clean up report
        record_dict = self.parser.filter_and_concat(entry_points)

        # Filter out events with uninformative API sequences
        if (
            'apis' in record_dict and
            record_dict['apis'].shape[0] == 1 and
            record_dict['apis'].iloc[0].api_name == 'MSVBVM60.ordinal_100'
        ):
            return None

        # Normalize data
        if 'file_access' in record_dict:
            record_dict['file_access'] = normalize_table_path(
                record_dict['file_access'], col='path'
            )
        if (
            'network_events.traffic' in record_dict and
            'server' in record_dict['network_events.traffic'].columns
        ):
            record_dict['network_events.traffic'] = normalize_table_ip(
                record_dict['network_events.traffic'], col='server'
            )
        if (
            'network_events.dns' in record_dict and
            'query' in record_dict['network_events.dns'].columns
        ):
            record_dict['network_events.dns']['query'] = (
                record_dict['network_events.dns']['query']
                .apply(lambda x: ' '.join(x.split('.')))
            )
        # Normalize args to exclude any non-alphanumeric characters
        if 'args' in record_dict['apis'].columns:
            record_dict['apis']['args'] = record_dict['apis']['args'].apply(
                lambda args_list: [get_alphanum_chars(arg) for arg in args_list]
            )

        # Limit verbose fields to a certain number of records
        if self.record_limits:
            for field, limit in self.record_limits.items():
                if field in record_dict:
                    record_dict[field] = record_dict[field].head(limit)
        # Join records
        record_json = self.join_records_to_json(record_dict)
        return record_json

    @staticmethod
    def join_records_to_json(record_dict):
        # Sort keys to ensure consistent order, put 'apis' at the end
        sorted_keys = sorted(record_dict.keys(), reverse=True)
        json_event_parts = [
            f"\"{key}\":{record_dict[key].to_json(orient='records')}"
            for key in sorted_keys
        ]
        json_event = "{" + ",".join(json_event_parts) + "}"
        return json.loads(json_event)
