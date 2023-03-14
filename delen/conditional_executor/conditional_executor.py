#!/usr/bin/env python3
"""
    Created date: 3/14/23
"""

import json
import time
import numpy as np

from abc import ABC
from enum import Enum
from dataclasses import dataclass
from typing import List

from . import trt_inference_engine


class Metrics(Enum):
    RESPONSE_TIME = "response_time"
    CONFIDENCE = "confidence"
    ACCURACY = "accuracy"
    ENERGY = "energy"
    COMPUTATION = "computation"
    EXIT = "exit"


class CompareOperators(Enum):
    LESS = lambda x, y: x < y
    LESS_EQ = lambda x, y: x <= y
    EQUAL = lambda x, y: x == y
    GREATER = lambda x, y: x > y
    GREATER_EQ = lambda x, y: x >= y
    NULL = lambda x, y: False


class BooleanOperators(Enum):
    AND = lambda x, y: x and y
    OR = lambda x, y: x or y


class TaskState(Enum):
    INIT = 0
    STARTED = 1
    FINISHED = 2

@dataclass
class Status:
    response_time: float = 0
    confidence: float = 0
    accuracy: float = 0
    energy: float = 0
    computation: float = 0
    exit: int = -1


@dataclass
class Task:
    inputs: np.ndarray
    outputs: np.ndarray = None
    start_t: float = 0
    end_t: float = 0
    status: Status = Status()
    state: TaskState = TaskState.INIT


class ConditionBase(ABC):
    def evaluate(self, stat: Status):
        pass


class SingleCondition(ConditionBase):

    def __init__(self, metric: Metrics, op: CompareOperators, threshold: float):
        super().__init__()
        self._metric = metric
        self._op = op
        self._threshold = threshold

    def evaluate(self, stat: Status):
        metric = getattr(stat, self._metric.value)
        return self._op(metric, self._threshold)

    @classmethod
    def load_conditions_from_json(cls, filename):
        with open(filename, 'r') as f:
            cond_data_list = json.loads(f.read())

        conditions = []
        for cond_data in cond_data_list:
            metric = getattr(Metrics, cond_data["metric"].upper())
            op = getattr(CompareOperators, cond_data["op"].upper())
            threshold = float(cond_data["threshold"])
            conditions.append(cls(metric, op, threshold))

        return conditions


class CompoundCondition(ConditionBase):
    def __init__(self, op: BooleanOperators, *conditions: ConditionBase):
        super().__init__()
        if len(conditions) < 1:
            raise ValueError(f"Compound condition requires as least one condition")

        self._op = op
        self._conditions = conditions

    def evaluate(self, stat: Status):
        ret = True if self._op == BooleanOperators.AND else False

        for cond in self._conditions:
            ret = self._op(ret, cond.evaluate())

        return ret


class ConditionalExecutor(object):
    def __init__(self, model_dir: str,
                 condition_file: str,
                 profile_file: str,
                 intermediate_output_name: str = "features"):
        super().__init__()
        self._engine = trt_inference_engine.TRTAnytimeDNNEngine(model_dir, intermediate_output_name)
        self._conditions = SingleCondition.load_conditions_from_json(condition_file)

        with open(profile_file, 'r') as f:
            self._profile = json.loads(f.read())

        assert self._engine.num_subnets == len(self._conditions)
        assert self._engine.num_subnets == len(self._profile)

    def update_conditions(self, new_conditions: List[ConditionBase]):
        assert self._engine.num_subnets == len(new_conditions)
        self._conditions = new_conditions

    def execute(self, task: Task):
        task.state = TaskState.STARTED
        task.start_t = time.time()

        for i in range(self._engine.num_subnets):
            if i == 0:
                res = self._engine.infer_to(task.inputs, i)
            else:
                self._engine.execute_subnet(i)
                res = self._engine.get_subnet_output(i)

            logits = res["output"].flatten()
            prob = np.exp(logits - np.max(logits))
            prob = prob / prob.sum()
            confidence = np.max(prob)

            task.status.response_time = time.time() - task.start_t
            task.status.accuracy = self._profile["accuracy"]
            task.status.energy = self._profile["energy"]
            task.status.computation = self._profile["flops"]
            task.status.confidence = confidence
            task.status.exit = i

            cond = self._conditions[i].evaluate(task.status)

            if cond:
                task.outputs = prob
                break

        task.end_t = time.time()
        task.state = TaskState.FINISHED
