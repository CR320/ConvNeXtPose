import torch
from mmcv import ProgressBar
from models.post_process import parse_results


def inference(model, data, eval_cfg, gpu_id):
    inputs = data['image']
    img_metas = data['img_metas'][0]
    eval_cfg.update(img_metas)

    outputs_list = list()
    for input in inputs:
        input = input.to('cuda:{}'.format(gpu_id))
        with torch.no_grad():
            outputs = model(input, phase='inference')

        outputs_list.append(outputs)
    results = parse_results(outputs_list, eval_cfg=eval_cfg)

    return results


def evaluate(model, data_loader, eval_cfg, gpu_id, distributed=False):
    # test
    model.eval()
    results_list = list()

    if not distributed or gpu_id == 0:
        prog_bar = ProgressBar(len(data_loader))

    for i, data in enumerate(data_loader):
        results = inference(model, data, eval_cfg, gpu_id)
        results_list.append(dict(
            preds=results['poses'],
            scores=results['scores'],
            image_path=data['img_metas'][0]['image_file']
        ))

        if not distributed or gpu_id == 0:
            prog_bar.update()

    return results_list
