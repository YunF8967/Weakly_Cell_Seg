from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.logger import setup_logger
import json

class DiceEvaluator(DatasetEvaluator):
    def __init__(self):
        self._logger = setup_logger()

    def reset(self):
        self._prediction = []
    
    def process(self, inputs, outputs):
        inPath = "input.json"
        with open(inPath, 'a') as output_json_file:
            json.dump(inputs, output_json_file)
        outPath = "output.json"
        with open(outPath, 'a') as output_json_file:
            json.dump(outputs, output_json_file)
        for input, output in zip(inputs, outputs):
            self._logger.info("input: {}".format(input))
            self._logger.info("output: {}".format(output))
        # self._logger.info("input: {}".format(inputs))
        # self._logger.info("output: {}".format(outputs))

    def evaluate(self):
        return {"dice": -25}

