from deeplab3.evaluators.segmentation_evaluator import SegmentationEvaluator

def make_evaluator(cfg, num_classes):
    if cfg.EVALUATOR.NAME == "segmentation":
        return SegmentationEvaluator(num_classes)
    else:
        raise ValueError("Model not implemented: {}".format(cfg.EVALUATOR.NAME))