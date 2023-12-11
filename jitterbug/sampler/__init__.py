"""Functions to sample atomic configurations"""
from typing import Type

from .base import StructureSampler
from .random import RandomSampler

methods: dict[str, Type[StructureSampler]] = {
    'simple': RandomSampler
}
