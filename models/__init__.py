import importlib
from .trainer import Trainer
from .network import ConvNeXtV2, HRCoMer, HRAdapter, MSPose
from .loss.matcher import HungarianMatcher, BinaryFocalCost, OksCost, JointL1Cost, LimbL1Cost
from .loss.loss_factory import BinaryFocalLoss, OksLoss, JointL1Loss, LimbL1Loss, HeatmapFocalLoss


def build_module(args):
    model_lib = importlib.import_module('models')
    for item in args:
        sub_args = args[item]
        if isinstance(sub_args, dict) and 'type' in sub_args:
            model = getattr(model_lib, sub_args['type'])
            sub_args.pop('type')
            args[item] = model(**build_module(sub_args))

    return args


def build_net(args):
    return eval(args['type'])(**build_module(args))


def build_trainer(args):
    args.model = build_net(args.model)

    if 'matcher' in args:
        cost = build_module(args.matcher.cost)
        args.matcher.pop('cost')
        args.matcher = HungarianMatcher(cost, **args.matcher)

    args.criterion = build_module(args.criterion)

    return Trainer(**args)

