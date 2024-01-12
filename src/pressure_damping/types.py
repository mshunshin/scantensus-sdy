from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CurveConfig:
    """
    Represents the configuration for a curve.
    """

    color: str
    name: str
    stepSize: str


@dataclass
class Config:
    access: str
    accessEmailsColonInKeyButDotInValue: Dict[str, str]
    curves: Dict[str, CurveConfig]
    imageStore: str
    leaderboardSize: int
    name: str
    projectClass: str


@dataclass 
class Curve:
    project: str
    file: str
    user: str
    time: str
    label: str
    xs: List[float]
    ys: List[float]
    straight_flag: List[int]
    type: str


@dataclass
class MattLabelOutput:
    project: str
    file: str
    user: str
    time: str
    label: str
    vis: str 
    instance_num: int
    value_x: str
    value_y: str
    straight_segment: str 
