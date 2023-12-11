"""Functions to sample atomic configurations"""
from typing import Type

from .base import StructureSampler
from .random import UniformSampler

methods: dict[str, Type[StructureSampler]] = {
    'uniform': UniformSampler
}
