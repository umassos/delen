#!/usr/bin/env python3

import logging
import time
import os
import numpy as np
import argparse
import pycuda.driver as cuda

from delen import conditional_executor
from glob import glob


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument("-m", "--model-dir", required=True, type=str, dest="model_dir", help="Model directory")
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", dest="input_shape", help="Input shape",
                        default=[1, 3, 224, 224])
    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s] %(funcName)s %(message)s")

    model_name = os.path.basename(os.path.dirname(args.model_dir))
    engine_dir = os.path.join(args.model_dir, "trt")
    profile_file = os.path.join(args.model_dir, f"{model_name}_profile.json")

    logger.info(f"Loading model {model_name} from {args.model_dir}")
    condition_files = glob(os.path.join(args.model_dir, "conditions*"))

    input_shape = args.input_shape
    inputs = np.random.random(size=input_shape)

    executor = conditional_executor.ConditionalExecutor(engine_dir, condition_files[0], profile_file)

    for condition_file in condition_files:
        conditions = conditional_executor.SingleCondition.load_conditions_from_json(condition_file)
        logger.info(f"Updating conditions from file {condition_file}")
        executor.update_conditions(conditions)

        start_t = time.time()
        while time.time() - start_t < 10:
            task = conditional_executor.Task(inputs=inputs)
            executor.execute(task)

            logger.info(f"task {task.state}, response time: {task.status.response_time*1000:.2f}")


if __name__ == '__main__':
    main()







