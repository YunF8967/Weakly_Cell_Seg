from detectron2.engine.hooks import HookBase

class DiceEvalHook(HookBase):

    def __init__(self):
        pass
    
    def after_step(self):
        print("Hellooooo step")
    
    
    def after_train(self):
        print("Hellooooo train")