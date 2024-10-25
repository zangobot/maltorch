"""
Trizna, D. and Demetrio, L. and Battista, B. and Fabio, R. (2024)
Nebula: Self-Attention for Dynamic Malware Analysis
IEEE Transactions on Information Forensics and Security (TIFS)
Paper: https://ieeexplore.ieee.org/document/10551436
Source: https://github.com/dtrizna/nebula/blob/main/nebula/preprocessing/normalization.py
"""

import re
import pandas as pd
from .constants import VARIABLE_MAP


def normalize_string_hash(string: str) -> str:
    if not isinstance(string, str):
        return string
    string = re.sub(r'[0-9a-fA-F]{64}', "<sha256>", string)
    string = re.sub(r'[0-9a-fA-F]{40}', "<sha1>", string)
    string = re.sub(r'[0-9a-fA-F]{32}', "<md5>", string)
    return string


def normalize_string_domain(
    string: str,
    domain_regex: str = r"\b([a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]\.)+",
    top_level_domains_regex: str = r"(com|net|ai|us|uk|cz|gov)\b",
    placeholder: str = "<domain>"
) -> str:
    if not isinstance(string, str):
        return string
    full_regex = domain_regex + top_level_domains_regex
    return re.sub(full_regex, placeholder, string)


def normalize_string_ip(string: str) -> str:
    if not isinstance(string, str):
        return string
    string = re.sub(r'127\.\d{1,3}\.\d{1,3}\.\d{1,3}', "<lopIP>", string)
    string = re.sub(r'169\.254\.169\.254', "<imds>", string)
    string = re.sub(
        r'(10(\.\d{1,3}){3}|(172\.(1[6-9]|2[0-9]|3[0-1])|192\.168)(\.\d{1,3}){2})',
        "<prvIP>",
        string
    )
    string = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', "<pubIP>", string)
    # Exclude IPv6 for now
    # string = re.sub(r'([0-9a-fA-F]{1,4}:){1,7}:', "<IPv6>", string)
    return string


def normalize_string_path(path: str) -> str:
    """
    Normalizes a file path by substituting drive letters, network hosts, user names, and environment variables.
    """
    if not isinstance(path, str):
        return path

    # Clean up auxiliary strings
    path = path.lower().replace("*raw:", "").replace("*amsiprocess:", "").replace("a script started by ", "").strip()

    # Normalize drive letters
    path = re.sub(r"\w:\\{1,2}", r"<drive>\\", path)

    # Normalize network paths
    path = re.sub(r"[\\]{1,2};((lanmanredirector|webdavredirector)\\;)?\w\:[a-z0-9]{16}", r"\\", path)
    path = re.sub(r"\\\\[\w\d.\-]+\\", r"<net>\\", path)

    # Normalize drive if path starts with single backslash
    path = re.sub(r"^\\([^\\])", r"<drive>\\\1", path)

    # Normalize volume GUID paths
    path = re.sub(r"\\[\.\?]\\volume\{[a-z0-9\-]{36}\}", r"<drive>", path)

    # Normalize non-default users
    default_users = ["administrator", "public", "default"]
    if "users\\" in path:
        if not any(f"users\\{user}\\" in path for user in default_users):
            path = re.sub(r"users\\[^\\]+\\", r"users\\<user>\\", path)

    # Replace environment variables
    for k, v in VARIABLE_MAP.items():
        path = path.replace(k, v)

    return path


def normalize_table_path(df: pd.DataFrame, col: str = "path") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df[col] = df[col].apply(normalize_string_path)
    df[col] = df[col].apply(normalize_string_hash)
    return df


def normalize_table_ip(df: pd.DataFrame, col: str = 'auditd.summary.object.primary') -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df[col] = df[col].apply(normalize_string_ip)
    return df
