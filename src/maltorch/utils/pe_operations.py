import itertools
import math
import struct

import lief
import torch


def _shift_pointer_to_section_content(
        liefpe: lief.PE.Binary,
        raw_code: bytearray,
        entry_index: int,
        amount: int,
        pe_shifted_by: int = 0,
) -> bytearray:
    """
    Shifts the section content pointer.

    Parameters
    ----------
    liefpe : lief.PE.Binary
            the binary wrapper by lief
    raw_code : bytearray
            the code of the executable to eprturb
    entry_index : int
            the entry of the section to manipulate
    amount : int
            the shift amount
    pe_shifted_by : int, optional, default 0
            if the PE header was shifted, this value should be set to that amount
    Returns
    -------
    bytearray
            the modified code
    """
    pe_position = liefpe.dos_header.addressof_new_exeheader + pe_shifted_by
    optional_header_size = liefpe.header.sizeof_optional_header
    coff_header_size = 24
    section_entry_length = 40
    size_of_raw_data_pointer = 20
    shift_position = (
            pe_position
            + coff_header_size
            + optional_header_size
            + (entry_index * section_entry_length)
            + size_of_raw_data_pointer
    )
    old_value = struct.unpack("<I", raw_code[shift_position: shift_position + 4])[0]
    new_value = old_value + amount
    new_value = struct.pack("<I", new_value)
    raw_code[shift_position: shift_position + 4] = new_value

    return raw_code


def _shift_pe_header(
        liefpe: lief.PE.Binary, raw_code: bytearray, amount: int
) -> bytearray:
    """
    Shifts the PE header, injecting a default pattern

    Parameters
    ----------
    liefpe : lief.PE.Binary
            the binary wrapper by lief
    raw_code : bytearray
            the code of the executable to perturb
    amount : int
            how much to inject, already aligned to file_alignment

    Returns
    -------
    bytearray
            the modified code
    """
    if amount == 0:
        return raw_code
    pe_position = liefpe.dos_header.addressof_new_exeheader
    # section_alignment = liefpe.optional_header.section_alignment
    # file_alignment = liefpe.optional_header.file_alignment
    raw_code[0x3C:0x40] = struct.pack("<I", pe_position + amount)

    raw_code[pe_position + 60 + 20 + 4: pe_position + 60 + 20 + 4 + 4] = struct.pack(
        "<I", liefpe.optional_header.sizeof_headers + amount
    )
    pattern = itertools.cycle("I love ToucanStrike <3")
    [raw_code.insert(pe_position, ord(next(pattern))) for _ in range(amount)]

    return raw_code


def extend_manipulation(
        x: torch.Tensor, preferable_extension_amount: int
) -> (list, list):
    """
    Applies the DOS header extension to a sample contained inside a list

    Parameters
    ----------
    x : list
            the sample as a torch Tensor
    preferable_extension_amount : int
            how much extension

    Returns
    -------
    list, list
            returns the perturbed sample and which are the indexes that can be perturbed
    """
    if preferable_extension_amount == 0:
        return x, []
    adv_x = x.flatten().tolist()
    liefpe = lief.PE.parse(adv_x)
    section_file_alignment = liefpe.optional_header.file_alignment
    if section_file_alignment == 0:
        return x, []
    first_content_offset = liefpe.dos_header.addressof_new_exeheader
    extension_amount = (
            int(math.ceil(preferable_extension_amount / section_file_alignment))
            * section_file_alignment
    )
    index_to_perturb = list(range(2, 0x3C)) + list(
        range(0x40, first_content_offset + extension_amount)
    )
    adv_x = _shift_pe_header(liefpe, x, extension_amount)
    for i, _ in enumerate(liefpe.sections):
        adv_x = _shift_pointer_to_section_content(
            liefpe, bytearray(adv_x), i, extension_amount, extension_amount
        )
    x = torch.Tensor(adv_x)
    return x, index_to_perturb


def padding_manipulation(x: torch.Tensor, padding: int) -> (torch.Tensor, list):
    """
    Applies the padding to all vectors in x, returning also the indexes to perturb
    """
    index_to_perturb = list(range(x.shape[-1], x.shape[-1] + padding))
    x = torch.hstack((x, torch.zeros([x.shape[0], padding])))
    return x, index_to_perturb


def content_shift_manipulation(
        x: torch.Tensor, preferable_extension_amount: int, pe_shifted_by: int = 0
) -> (torch.Tensor, list):
    """
    Applies the content shifting to a sample contained inside a list

    Parameters
    ----------
    x : list
            single sample as a torch tensor
    preferable_extension_amount : int
            how much extension
    pe_shifted_by : int, optional, default 0
            if the PE header was shifted, this value should be set to that amount

    Returns
    -------
    list, list
            returns the perturbed sample and which are the indexes that can be perturbed
    """
    if not preferable_extension_amount:
        return x, []
    adv_x = x.flatten().tolist()
    liefpe = lief.PE.parse(adv_x)
    section_file_alignment = liefpe.optional_header.file_alignment
    if section_file_alignment == 0:
        return x, []
    first_content_offset = liefpe.sections[0].offset
    extension_amount = (
            int(math.ceil(preferable_extension_amount / section_file_alignment))
            * section_file_alignment
    )
    index_to_perturb = list(
        range(first_content_offset, first_content_offset + extension_amount)
    )
    for i, _ in enumerate(liefpe.sections):
        adv_x = _shift_pointer_to_section_content(
            liefpe, bytearray(adv_x), i, extension_amount, pe_shifted_by
        )
    adv_x = (
            adv_x[:first_content_offset]
            + b"\x00" * extension_amount
            + adv_x[first_content_offset:]
    )
    x = torch.Tensor(adv_x)
    return x, index_to_perturb


def header_fields_manipulations(pe_index: int):
    # COFF manipulations
    # 4 bytes timestamp at PE + 4 (PE size) + 4
    # 4 bytes PointerToSymbolTable at PE + 4 (PE size) + 8
    # 4 bytes NumberOFSymbols at PE + 4 (PE size) + 12
    #
    # Optional header manipulations
    # 1 byte MajorLinkerVersion at PE + 4 + 20 (COFF size) + 2
    # 1 byte MinorLinkerVersion at PE + 4 + 20 (COFF size) + 3
    # 2 byte MajorOperatingSystemVersion at PE + 4 + 20 (COFF size) + 40
    # 2 byte MinorOperatingSystemVersion at PE + 4 + 20 (COFF size) + 42
    # 2 byte MajorImageVersion at PE + 4 + 20 (COFF size) + 44
    # 2 byte MinorImageVersion at PE + 4 + 20 (COFF size) + 46

    coff_indexes = list(range(pe_index + 4 + 4, pe_index + 4 + 12))
    optional_header_indexes = [pe_index + 24 + 2, pe_index + 24 + 2]
    optional_header_indexes.extend(list(range(pe_index + 24 + 40, pe_index + 24 + 48)))
    indexes = coff_indexes + optional_header_indexes
    return indexes


def section_injection_manipulation(
        x: torch.Tensor, how_many_sections: int, size_per_section: int = 0x200
) -> (list, list):
    pe = lief.PE.parse(x.flatten().tolist())
    default_sect_name = "MLADVEXE"
    for _ in range(how_many_sections):
        section = lief.PE.Section(default_sect_name)
        section.content = [0] * size_per_section
        pe.add_section(section)
    builder = lief.PE.Builder(pe)
    builder.build()
    x_init = builder.get_build()
    pe = lief.PE.parse(raw=x_init)
    index_to_perturb = []
    for section in pe.sections:
        if section.name != default_sect_name:
            continue
        indexes = []  # add 8 of name + index of content
        index_to_perturb.extend(indexes)
    x_init = torch.Tensor(x_init)
    return x_init, index_to_perturb
