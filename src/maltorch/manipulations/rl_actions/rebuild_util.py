import lief

def rebuild(binary, imports=False):
    config = lief.PE.Builder.config_t()
    config.imports = imports
    return binary.write_to_bytes(config)
