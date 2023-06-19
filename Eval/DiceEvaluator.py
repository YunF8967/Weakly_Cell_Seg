from detectron2.evaluation import DatasetEvaluator

class DiceEvaluator(DatasetEvaluator):
    
    def reset(self):
        self.count = 0

    def process(self, inputs, outputs):
        for output in outputs:
            self.count += len(output["instances"])

    def evaluate(self):
        # save self.count somewhere, or print it, or return it.
        return {"count": self.count}