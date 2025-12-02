PE_HEADER_OFFSET = 60
PE_HEADER_OFFSET_LENGTH = 4

PE_MAGIC_LENGTH = 4
OPT_HEADER_MAGIC_LENGTH = 2
COFF_LENGTH = 20
COFF_N_SECTION = 2

DLL_CHARACTERISTICS_OFFSET_FROM_PE = 70
DLL_CHARACTERISTICS_LENGTH = 2

SECTION_EXECUTABLE = 0x20000000
SECTION_READABLE = 0x40000000
SECTION_WRITEABLE = 0x80000000

DYNAMIC_REBASE = 0x0040
FORCE_INTEGRITY = 0x0080

SECTION_ENTRY_LENGTH = 40

PE_32_MAGIC = 0x10b
PE_64_MAGIC = 0x20b


def align(value, align_on):
    r = value % align_on
    if r > 0:
        return value + (align_on - r)
    return value


class Binary:
    def __init__(self, path=None, bytez=None):
        if path is None and bytez is None:
            raise ValueError("You must pass as input either a path or bytes")
        self.exe_bytes = bytearray()
        if path is not None:
            with open(path, 'rb') as f:
                self.exe_bytes = bytearray(f.read())
        else:
            self.exe_bytes = bytearray(bytez)

    @classmethod
    def load_from_path(cls, path: str):
        return cls(path=path)

    @classmethod
    def load_from_bytes(cls, bytez: bytearray):
        return cls(bytez=bytez)

    @staticmethod
    def flag_is_on(characteristics, flag):
        return (characteristics & flag) == flag

    @staticmethod
    def deactivate_flag(characteristics, flag):
        if Binary.flag_is_on(characteristics, flag):
            characteristics &= ~flag
        return characteristics

    @staticmethod
    def activate_flag(characteristics, flag):
        if not Binary.flag_is_on(characteristics, flag):
            characteristics |= flag
        return characteristics

    def get_bytes(self):
        return self.exe_bytes

    def get_pe_location(self):
        pe_location = self.exe_bytes[PE_HEADER_OFFSET:PE_HEADER_OFFSET + PE_HEADER_OFFSET_LENGTH]
        return int.from_bytes(pe_location, 'little')

    def get_optional_header_location(self):
        pe_location = self.get_pe_location()
        optional_header_location = pe_location + PE_MAGIC_LENGTH + 20
        return optional_header_location

    def get_optional_header_size(self):
        pe_location = self.get_pe_location()
        size_opt_header_location = pe_location + PE_MAGIC_LENGTH + 16
        size_opt_header = self.exe_bytes[size_opt_header_location: size_opt_header_location + 2]
        size_opt_header = int.from_bytes(size_opt_header, 'little')
        return size_opt_header

    def get_section_table_location(self):
        size_opt_header = self.get_optional_header_size()
        return self.get_optional_header_location() + size_opt_header


    def get_total_number_sections(self):
        pe_location = self.get_pe_location()
        n_sections = self.exe_bytes[
                     pe_location + PE_MAGIC_LENGTH + COFF_N_SECTION: pe_location + PE_MAGIC_LENGTH + COFF_N_SECTION + 2]
        n_sections = int.from_bytes(n_sections, 'little')
        return n_sections

    def increase_number_sections(self):
        pe_location = self.get_pe_location()
        n_sections = self.exe_bytes[
                     pe_location + PE_MAGIC_LENGTH + COFF_N_SECTION: pe_location + PE_MAGIC_LENGTH + COFF_N_SECTION + 2]
        n_sections = int.from_bytes(n_sections, 'little') + 1
        self.exe_bytes[
        pe_location + PE_MAGIC_LENGTH + COFF_N_SECTION: pe_location + PE_MAGIC_LENGTH + COFF_N_SECTION + 2] = n_sections.to_bytes(
            2, 'little')

    def get_section_entry_location_from_index(self, index: int):
        n_sections = self.get_total_number_sections()
        if index > n_sections:
            raise ValueError(f"Section with index {index} not found. Only {n_sections} are present.")
        section_table_offset = self.get_section_table_location()
        return section_table_offset + index * 40

    def get_section_entry_from_index(self, index: int):
        n_sections = self.get_total_number_sections()
        if index > n_sections:
            raise ValueError(f"Section with index {index} not found. Only {n_sections} are present.")
        section_table_offset = self.get_section_table_location()
        return self.exe_bytes[section_table_offset + index * 40: section_table_offset + (index + 1) * 40]


    def get_section_alignment(self):
        optional_header_location = self.get_optional_header_location()
        section_alignment = int.from_bytes(
            self.exe_bytes[optional_header_location + 32: optional_header_location + 32 + 4], 'little')
        return section_alignment

    def get_file_alignment(self):
        optional_header_location = self.get_optional_header_location()
        file_alignment = int.from_bytes(
            self.exe_bytes[optional_header_location + 36: optional_header_location + 36 + 4], 'little')
        return file_alignment

    def get_sizeof_headers(self):
        optional_header_location = self.get_optional_header_location()
        sizeof_headers = int.from_bytes(
            self.exe_bytes[optional_header_location + 60: optional_header_location + 60 + 4], 'little')
        return sizeof_headers  # multiple of FileAlignment

    def set_sizeof_headers(self, value: int):
        optional_header_location = self.get_optional_header_location()
        self.exe_bytes[optional_header_location + 60: optional_header_location + 60 + 4:] = value.to_bytes(4, 'little')
        return value  # multiple of FileAlignment

    def get_sizeof_image(self):
        optional_header_location = self.get_optional_header_location()
        sizeof_image = int.from_bytes(self.exe_bytes[optional_header_location + 56: optional_header_location + 56 + 4],
                                      'little')
        return sizeof_image  # multiple of SectionAlignment

    def set_sizeof_image(self, value: int):
        optional_header_location = self.get_optional_header_location()
        self.exe_bytes[optional_header_location + 56: optional_header_location + 56 + 4:] = value.to_bytes(4, 'little')

    def increase_pointer_raw_section(self, section_index: int, value: int):
        section_table_offset = self.get_section_table_location() + section_index * 40
        old_pointer = int.from_bytes(self.exe_bytes[section_table_offset + 20: section_table_offset + 20 + 4], 'little')
        new_pointer = old_pointer + value
        self.exe_bytes[section_table_offset + 20: section_table_offset + 20 + 4] = new_pointer.to_bytes(4, 'little')

    def rva_to_file_offset(self, rva: int):
        # Helper: convert an RVA to file offset using section table
        # iterate sections and find which contains rva
        total = self.get_total_number_sections()
        st = self.get_section_table_location()
        for i in range(total):
            entry = self.exe_bytes[st + i * SECTION_ENTRY_LENGTH: st + (i + 1) * SECTION_ENTRY_LENGTH]
            va = int.from_bytes(entry[12:16], 'little')
            vsz = int.from_bytes(entry[8:12], 'little')
            ptr = int.from_bytes(entry[20:24], 'little')
            # NOTE: VirtualSize may be zero, but SizeOfRawData can still be >0; use VirtualSize if present else SizeOfRawData
            if vsz == 0:
                vsz = int.from_bytes(entry[16:20], 'little')
            if va <= rva < (va + vsz):
                return ptr + (rva - va)
        return None

    def add_robust_section(self, name: str, characteristics: int, content: bytearray):
        n_sections = self.get_total_number_sections()
        if n_sections == 0:
            raise RuntimeError("Binary has no sections; cannot add a new section safely.")

        # Read last section header
        last_section = self.get_section_entry_from_index(n_sections - 1)
        last_virtual_address = int.from_bytes(last_section[12:16], 'little')
        last_virtual_size = int.from_bytes(last_section[8:12], 'little')

        section_alignment = self.get_section_alignment()
        file_alignment = self.get_file_alignment()

        # compute new section VA correctly
        aligned_last_virtual_size = align(last_virtual_size, section_alignment)
        next_virtual_address = last_virtual_address + aligned_last_virtual_size

        virtual_size = len(content)
        size_of_raw_data = align(virtual_size, file_alignment)

        # compose IMAGE_SECTION_HEADER (40 bytes)
        new_section_entry = bytearray(SECTION_ENTRY_LENGTH)
        name_bytes = name.encode('ascii', errors='ignore')[:8]
        new_section_entry[0:len(name_bytes)] = name_bytes
        new_section_entry[8:12] = (virtual_size).to_bytes(4, 'little')            # VirtualSize
        new_section_entry[12:16] = next_virtual_address.to_bytes(4, 'little')    # VirtualAddress
        new_section_entry[16:20] = size_of_raw_data.to_bytes(4, 'little')        # SizeOfRawData
        # PointerToRawData will be set later (we choose EOF aligned to file_alignment)
        new_section_entry[36:40] = characteristics.to_bytes(4, 'little')         # Characteristics

        # where the new section header should go in the section table
        section_table_offset = self.get_section_table_location()
        new_section_header_offset = section_table_offset + n_sections * SECTION_ENTRY_LENGTH

        sizeof_headers = self.get_sizeof_headers()

        # locate DataDirectory base and relevant entries (cert & debug)
        opt_loc = self.get_optional_header_location()
        opt_magic = int.from_bytes(self.exe_bytes[opt_loc: opt_loc + 2], 'little')
        # data directories start offset relative to Optional Header:
        # PE32 -> offset 96, PE32+ -> offset 112
        if opt_magic == PE_32_MAGIC:
            data_dir_base = opt_loc + 96
        else:
            data_dir_base = opt_loc + 112
        # certificate directory entry (IMAGE_DIRECTORY_ENTRY_SECURITY) is index 4, each entry 8 bytes
        cert_dir_offset = data_dir_base + 4 * 8
        # debug directory entry is index 6 (IMAGE_DIRECTORY_ENTRY_DEBUG)
        debug_dir_offset = data_dir_base + 6 * 8



        # If the new section header doesn't fit in SizeOfHeaders, expand headers
        if new_section_header_offset + SECTION_ENTRY_LENGTH > sizeof_headers:
            # compute minimal increment to make room, align to FileAlignment
            needed = (new_section_header_offset + SECTION_ENTRY_LENGTH) - sizeof_headers
            increment = align(needed, file_alignment)

            insert_at = new_section_header_offset  # we insert padding at the end of the current section table
            # insert zeros into file bytes (this shifts everything after insert_at)
            self.exe_bytes = self.exe_bytes[:insert_at] + (b'\x00' * increment) + self.exe_bytes[insert_at:]

            # update SizeOfHeaders
            self.set_sizeof_headers(sizeof_headers + increment)

            # shift all existing section PointerToRawData by increment
            for i in range(n_sections):
                self.increase_pointer_raw_section(i, increment)

            # Certificate directory: VirtualAddress field actually stores a file offset (per PE spec).
            try:
                cert_va = int.from_bytes(self.exe_bytes[cert_dir_offset: cert_dir_offset + 4], 'little')
                cert_size = int.from_bytes(self.exe_bytes[cert_dir_offset + 4: cert_dir_offset + 8], 'little')
                if cert_va != 0 and cert_va >= insert_at:
                    new_cert_va = cert_va + increment
                    self.exe_bytes[cert_dir_offset: cert_dir_offset + 4] = new_cert_va.to_bytes(4, 'little')
            except IndexError:
                # optional header smaller/invalid — ignore safely
                pass

            # Debug directory: if present, it is referenced by a data-directory RVA (index 6)
            try:
                debug_rva = int.from_bytes(self.exe_bytes[debug_dir_offset: debug_dir_offset + 4], 'little')
                debug_size = int.from_bytes(self.exe_bytes[debug_dir_offset + 4: debug_dir_offset + 8], 'little')
                if debug_rva != 0 and debug_size != 0:
                    debug_file_offset = self.rva_to_file_offset(debug_rva)
                    if debug_file_offset is not None:
                        # IMAGE_DEBUG_DIRECTORY is 28 bytes each
                        entry_size = 28
                        count = debug_size // entry_size
                        for di in range(count):
                            entry_off = debug_file_offset + di * entry_size
                            # PointerToRawData is at offset 24 (4 bytes)
                            ptr_field_off = entry_off + 24
                            # Defensive bounds check
                            if ptr_field_off + 4 <= len(self.exe_bytes):
                                old_ptr = int.from_bytes(self.exe_bytes[ptr_field_off: ptr_field_off + 4], 'little')
                                if old_ptr != 0 and old_ptr >= insert_at:
                                    new_ptr = old_ptr + increment
                                    self.exe_bytes[ptr_field_off: ptr_field_off + 4] = new_ptr.to_bytes(4, 'little')
            except (IndexError, ValueError):
                # Not present or malformed - ignore
                pass

            # update sizeof_headers var for subsequent checks
            sizeof_headers = self.get_sizeof_headers()

        # After potential expansion, choose PointerToRawData: append at EOF aligned to FileAlignment
        pointer_to_raw = align(len(self.exe_bytes), file_alignment)
        new_section_entry[20:24] = pointer_to_raw.to_bytes(4, 'little')

        # final safety check: the new header must now fit in SizeOfHeaders
        if new_section_header_offset + SECTION_ENTRY_LENGTH > sizeof_headers:
            raise RuntimeError("Failed to make room for the new section header inside SizeOfHeaders; aborting.")

        # write the new section header
        self.exe_bytes[new_section_header_offset: new_section_header_offset + SECTION_ENTRY_LENGTH] = new_section_entry

        # append raw data at EOF (ensure space)
        needed_len = pointer_to_raw + size_of_raw_data
        if len(self.exe_bytes) < needed_len:
            self.exe_bytes += b'\x00' * (needed_len - len(self.exe_bytes))

        # write content
        self.exe_bytes[pointer_to_raw: pointer_to_raw + len(content)] = content

        # update SizeOfImage properly
        new_sizeof_image = next_virtual_address + align(virtual_size, section_alignment)
        new_sizeof_image = align(new_sizeof_image, section_alignment)
        self.set_sizeof_image(new_sizeof_image)

        # increase NumberOfSections
        self.increase_number_sections()

        return self.exe_bytes
