from detectron2.evaluation import DatasetEvaluator

class Counter(DatasetEvaluator):
    '''How many instances are detected
    * from: https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/evaluation.md
    '''
    
    def reset(self):
        self.count = 0

    def process(self, inputs, outputs):     # outputs = [batchOutput1[xx,xx,...], batchOutput2[...], ...]
        for output in outputs:
            self.count += len(output["instances"])

    def evaluate(self):
        # save self.count somewhere, or print it, or return it.
        return {"count": self.count}