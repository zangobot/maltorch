JSON_CLEANUP_SYMBOLS = ['"', "'", ":", ",", "[", "]", "{", "}", "\\", "/"]

SPEAKEASY_RECORD_LIMITS = {"network_events.traffic": 256}

SPEAKEASY_RECORD_FIELDS = [
    'file_access.event',
    'file_access.path',
    'network_events.traffic.server',
    'network_events.traffic.port',
    'registry_access.event',
    'registry_access.path',
    'apis.api_name',
    'apis.args',
    'apis.ret_val',
]

SPEAKEASY_TOKEN_STOPWORDS = ['api_name', 'args', 'ret_val', 'event', 'path', 'open_flags', 'access_flags', 'size', 'server', 'proto', 'port', 'method']

QUO_VADIS_LABELMAP = {
    "clean": 0,
    "backdoor": 1,
    "coinminer": 2,
    "dropper": 3,
    "keylogger": 4,
    "ransomware": 5,
    "rat": 6,
    "trojan": 7
}

# good reference:
# https://docs.microsoft.com/en-us/windows/deployment/usmt/usmt-recognized-environment-variables
VARIABLE_MAP = {
    r"%systemdrive%": r"<drive>", 
    r"%systemroot%": r"<drive>\windows",
    r"%windir%": r"<drive>\windows", 
    r"%allusersprofile%": r"<drive>\programdata",
    r"%programdata%": r"<drive>\programdata",
    r"%programfiles%": r"<drive>\program files",
    r"%programfiles(x86)%": r"<drive>\program files (x86)",
    r"%programw6432%": r"<drive>\program files",
    r"%commonprogramfiles%": r"<drive>\program files\common files",
    r"%commonprogramfiles(x86)%": r"<drive>\program files (x86)\common files",
    r"%commonprogramw6432%": r"<drive>\program files\common files",
    r"%commonfiles%": r"<drive>\program files\common files",
    r"%profiles%": r"<drive>\users",
    r"%public%": r"<drive>\users\public",
    r"%userprofile%": r"<drive>\users\<user>"
}
# more user variables
VARIABLE_MAP.update({
    r"%homepath%": VARIABLE_MAP[r"%userprofile%"],
    r"%downloads%": VARIABLE_MAP[r"%userprofile%"] + r"\downloads",
    r"%desktop%": VARIABLE_MAP[r"%userprofile%"] + r"\desktop",
    r"%favorites%": VARIABLE_MAP[r"%userprofile%"] + r"\favorites",
    r"%documents%": VARIABLE_MAP[r"%userprofile%"] + r"\documents",
    r"%mydocuments%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%personal%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%localsettings%": VARIABLE_MAP[r"%userprofile%"] + r"\documents", # obsolete
    r"%mypictures%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my pictures",
    r"%mymusic%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my music",
    r"%myvideos%": VARIABLE_MAP[r"%userprofile%"] + r"\documents\my videos",
    r"%localappdata%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local",
    r"%appdata%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\roaming",
    r"%usertemp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%temp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%tmp%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\temp",
    r"%cache%": VARIABLE_MAP[r"%userprofile%"] + r"\appdata\local\microsoft\windows\temporary internet files"
})    