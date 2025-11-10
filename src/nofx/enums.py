from enum import Enum


class PositionSide(Enum):
    Long = 0
    Short = 1
    MAX_NUM = 2


class ActionType(Enum):
    DoNothing = 0
    OpenLong = 1
    OpenShort = 2
    CloseLong = 3
    CloseShort = 4
    MAX_NUM = 5