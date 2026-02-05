"""
Author: Redal
Date: 2026-02-02
Todo: 定义了一系列枚举类IntEnum,用于表示不同数据结构中的索引和属性,
      SceneFrameType:表示场景帧的类型(原始帧或合成帧);StateSE2Index:表示二维姿态SE(2)的索引,
      包括位置(X、Y)和朝向(HEADING),并提供了便捷的切片属性;BoundingBoxIndex:表示包围盒的索引,
      包括位置(X、Y、Z)、尺寸(长度、宽度、高度)和朝向HEADING,同样支持切片访问;LidarIndex:表示
      激光雷达点云数据的索引,包括三维坐标(X、Y、Z)、强度、环号和ID等信息,也提供了常用的切片属性
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
from enum import IntEnum


class SceneFrameType(IntEnum):
    """Intenum for scene frame types"""
    ORIGINAL = 0
    SYNTHETIC = 1


class StateSE2Index(IntEnum):
    """Intenum for SE(2) arrays"""
    _X = 0
    _Y = 1
    _HEADING = 2
    @classmethod
    def size(cls):
        """compute and return the number of attributes of class"""
        # select satisfied attributes and return all of them
        # mind that the attribute cannot be callable and should start with "_" but not "__"
        valid_attributes = [ attribute for attribute in dir(cls) if attribute.startswith("_") 
                            and not attribute.startswith("__") and not callable(getattr(cls, attribute))]
        return len(valid_attributes)
    @classmethod
    @property
    def X(cls):
        return cls._X
    @classmethod
    @property
    def Y(cls):
        return cls._Y
    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING
    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)
    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


class BoundingBoxIndex(IntEnum):
    """Intenum of bounding box in logs"""
    _X = 0
    _Y = 1
    _Z = 2
    _LENGTH = 3
    _WIDTH = 4
    _HEIGHT = 5
    _HEADING = 6
    @classmethod
    def size(cls):
        """"""
