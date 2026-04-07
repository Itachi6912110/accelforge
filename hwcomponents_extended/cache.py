"""
SimpleSRAMCache: a set-associative cache model built from two SRAM arrays.

The hwcomponents_cacti.Cache class has a known bug where its CACTI design
objective constraints (deviate 20:10:...) are too tight for the tag array
search at all supported tech nodes, making it non-functional. This model
avoids that issue by composing a cache from two hwcomponents_cacti.SRAM
instances (data array + tag array) which use cache_type="ram" internally
and are not affected by the tag-array optimization bug.
"""

import math
from hwcomponents import ComponentModel, action
from hwcomponents_cacti.hwcomponents_cacti import SRAM


class SimpleSRAMCache(ComponentModel):
    """
    Set-associative cache model composed of a data SRAM and a tag SRAM.

    Models a cache as two physical sub-arrays:
      - Data array: stores cache lines (n_sets × ways entries, block_size wide)
      - Tag array:  stores tag + valid + dirty bits (n_sets × ways entries)

    Address breakdown for a 32-bit CPU address:
      [ tag (addr_bits - index - offset) | index (log2 n_sets) | offset (log2 block_size_bytes) ]

    Energy per cache read  = tag_array.read() × ways + data_array.read()
    Energy per cache write = tag_array.write() + data_array.write()

    Parameters
    ----------
    tech_node : float
        Technology node in meters (e.g., 16e-9 for 16 nm).
    cache_size_bytes : int
        Total cache capacity in bytes.
    block_size_bytes : int
        Cache line size in bytes. Must be a power of two >= 8.
    associativity : int
        Number of ways. Must be a power of two >= 1.
    addr_bits : int
        CPU address width in bits (default: 64).
    n_rw_ports : int
        Number of read/write ports on each sub-array (default: 1).
    n_banks : int
        Number of banks on each sub-array (default: 1).
    """

    component_name = ["cache", "Cache", "SimpleSRAMCache"]
    priority = 0.5

    def __init__(
        self,
        tech_node: float,
        cache_size_bytes: int,
        block_size_bytes: int,
        associativity: int,
        addr_bits: int = 64,
        n_rw_ports: int = 1,
        n_banks: int = 1,
        hit_rate: float = 1.0
    ):
        self.tech_node = tech_node
        self.cache_size_bytes = cache_size_bytes
        self.block_size_bytes = block_size_bytes
        self.associativity = associativity
        self.addr_bits = addr_bits
        self.hit_rate = hit_rate

        n_sets = cache_size_bytes // (block_size_bytes * associativity)
        offset_bits = int(math.log2(block_size_bytes))
        index_bits = int(math.log2(n_sets))
        tag_bits = addr_bits - index_bits - offset_bits
        tag_entry_bits = tag_bits + 2  # +1 valid, +1 dirty

        total_lines = n_sets * associativity

        self._data_array = SRAM(
            tech_node=tech_node,
            width=block_size_bytes * 8,
            depth=total_lines,
            size=cache_size_bytes * 8,
            n_rw_ports=n_rw_ports,
            n_banks=n_banks,
        )
        self._tag_array = SRAM(
            tech_node=tech_node,
            width=tag_entry_bits,
            depth=total_lines,
            size=tag_entry_bits * total_lines,
            n_rw_ports=n_rw_ports,
            n_banks=n_banks,
        )

        super().__init__(
            leak_power=self._data_array.leak_power + self._tag_array.leak_power,
            area=self._data_array.area + self._tag_array.area,
        )

    @action
    def read(self) -> tuple[float, float]:
        """
        One cache read: 
            If hit: probe all ways (tag array) then fetch one data line.
            If miss: probe all ways (tag array), (wait next layer), write one tag, write one data, then fetch one data line.
        The energy/latency of next layer cache will be estimated at the next layer.

        Returns
        -------
        (energy_J, latency_s) : tuple[float, float]
            Energy in Joules, latency in seconds (tag and data accessed in parallel).
        """
        tag_rd_e, tag_rd_lat = self._tag_array.read()
        tag_wr_e, tag_wr_lat = self._tag_array.write()
        data_rd_e, data_rd_lat = self._data_array.read()
        data_wr_e, data_wr_lat = self._data_array.write()

        energy_hit = tag_rd_e * self.associativity + data_rd_e
        energy_miss = tag_rd_e * self.associativity + tag_wr_e + data_wr_e + data_rd_e

        latency_hit = max(tag_rd_lat, data_rd_lat)
        latency_miss = tag_rd_lat + max(tag_wr_lat, data_wr_lat) + data_rd_lat

        energy = energy_hit * self.hit_rate + energy_miss * (1-self.hit_rate)
        latency = latency_hit * self.hit_rate + latency_miss * (1-self.hit_rate)
        return energy, latency

    @action
    def write(self) -> tuple[float, float]:
        """
        One cache write: 
            If hit: read one tag, write one data.
            If miss: evict and write. 
                To check, read one tag.
                To evict, read one data (to write to next level).
                To write, write one tag, write one data.

        Returns
        -------
        (energy_J, latency_s) : tuple[float, float]
        """
        tag_rd_e, tag_rd_lat = self._tag_array.read()
        tag_wr_e, tag_wr_lat = self._tag_array.write()
        data_rd_e, data_rd_lat = self._data_array.read()
        data_wr_e, data_wr_lat = self._data_array.write()

        energy_hit = tag_rd_e + data_wr_e
        energy_miss = tag_rd_e + data_rd_e + tag_wr_e + data_wr_e

        latency_hit = max(tag_rd_lat, data_wr_lat)
        latency_miss = tag_rd_lat + data_rd_lat + max(tag_wr_lat, data_wr_lat)

        energy = energy_hit * self.hit_rate + energy_miss * (1-self.hit_rate)
        latency = latency_hit * self.hit_rate + latency_miss * (1-self.hit_rate)

        return energy, latency
