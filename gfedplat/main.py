import gfedplat as fp
import numpy as np
import argparse
import torch
import sys
import os
import json
torch.multiprocessing.set_sharing_strategy('file_system')

def outFunc(alg):
    loss_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        training_loss = metric_history['training_loss'][-1]
        if training_loss is None:
            continue
        loss_list.append(training_loss)
    loss_list = np.array(loss_list)

    local_acc_list = []
    for i, metric_history in enumerate(alg.comm_log['client_metric_history']):
        local_acc_list.append(metric_history['test_accuracy'][-1])
    local_acc_list = np.array(local_acc_list)
    p = np.ones(len(local_acc_list))

    # Save local test accuracy mean to JSON file
    accuracy_record = {
        'round': alg.current_comm_round - 1,
        'local_test_acc_mean': float(np.mean(local_acc_list/100))
    }
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    filename = 'local_test_accuracy_{}.json'.format(alg.name)
    save_path = os.path.join(save_dir, filename)

    # Load existing data if file exists
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(accuracy_record)
    stream_log = ""
    stream_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
    stream_log += 'round {}'.format(alg.current_comm_round - 1) + '\n'
    stream_log += 'Mean Global Test loss: ' + format(np.mean(loss_list), ".6f") + \
        '\n' if len(loss_list) > 0 else ''
    stream_log += 'global model test: \n'
    stream_log += 'Local Test Acc: ' + format(np.mean(local_acc_list/100), ".3f") + '\n'
    stream_log += '\n'
    alg.stream_log = stream_log + alg.stream_log
    print(stream_log)

def read_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='seed', type=int, default=1)
    parser.add_argument('--num_runs', help='number of runs with different seeds', type=int, default=3)

    parser.add_argument(
        '--device', help='device: -1, 0, 1, or ...', type=int, default=0)

    parser.add_argument('--module', help='module name;',
                        type=str, default='CNN')

    parser.add_argument('--algorithm', help='algorithm name;',
                        type=str, default='UNITE')

    parser.add_argument('--dataloader', help='dataloader name;',
                        type=str, default='DataLoader_cifar10_dir')

    parser.add_argument('--SN', help='split num', type=int, default=200)

    parser.add_argument('--PN', help='pick num', type=int, default=2)

    parser.add_argument('--B', help='batch size', type=int, default=50)

    parser.add_argument('--NC', help='client_class_num', type=int, default=2)

    parser.add_argument(
        '--Diralpha', help='alpha parameter for dirichlet', type=float, default=0.5)

    parser.add_argument('--types', help='dataloader label types;',
                        type=str, default='default_type')

    parser.add_argument('--N', help='client num', type=int, default=100)

    parser.add_argument(
        '--C', help='select client proportion', type=float, default=1.0)

    parser.add_argument('--R', help='communication round',
                        type=int, default=1000)

    parser.add_argument('--E', help='local epochs', type=int, default=1)

    parser.add_argument('--test_interval',
                        help='test interval', type=int, default=1)

    parser.add_argument('--test_conflicts',
                        help='test conflicts', type=str, default='False')

    parser.add_argument('--sgd_step', help='sgd training',
                        type=str, default='False')

    parser.add_argument('--lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('--decay', help='learning rate decay',
                        type=float, default=0.999)
    parser.add_argument('--momentum', help='momentum', type=float, default=0.0)
        # UNITE hyperparameters
    parser.add_argument('--personalization_alpha', help='alpha for personalized embeddings in UNITE',
                        type=float, default=0.5)
    parser.add_argument('--uncertainty_eps', help='epsilon for uncertainty calculation in UNITE',
                        type=float, default=1e-8)

    try:
        parsed = vars(parser.parse_args())
        return parsed
    except IOError as msg:
        parser.error(str(msg))

def initialize(params):
    import importlib
    fp.setup_seed(seed=params['seed'])
    device = torch.device(
        'cuda:' + str(params['device']) if torch.cuda.is_available() and params['device'] != -1 else "cpu")
    Module = getattr(sys.modules['gfedplat'], params['module'])
    module = Module(device)
    Dataloader = getattr(sys.modules['gfedplat'], params['dataloader'])
    data_loader = Dataloader(
        params=params, input_require_shape=module.input_require_shape)
    module.generate_model(data_loader.input_data_shape,
                          data_loader.target_class_num)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, module.model.parameters(
    )), lr=params['lr'], momentum=params['momentum'])
    train_setting = {'criterion': torch.nn.CrossEntropyLoss(
    ), 'optimizer': optimizer, 'lr_decay': params['decay'], 'sgd_step': eval(params['sgd_step'])}
    test_interval = params['test_interval']
    Algorithm = getattr(sys.modules['gfedplat'], params['algorithm'])
    algorithm = Algorithm(data_loader=data_loader,
                          module=module,
                          device=device,
                          train_setting=train_setting,
                          client_num=data_loader.pool_size,
                          online_client_num=int(
                              data_loader.pool_size * params['C']),
                          metric_list=[fp.Correct()],
                          max_comm_round=params['R'],
                          max_training_num=None,
                          epochs=params['E'],
                          outFunc=outFunc,
                          write_log=True,
                          params=params,)
    algorithm.test_interval = test_interval
    return data_loader, algorithm

if __name__ == '__main__':
    import copy
    params = read_params()
    num_runs = params.get('num_runs', 3)
    seeds = list(range(1, num_runs + 1))  # Seeds for multiple runs
    all_accuracies = []

    for seed in seeds:
        print(f"Run with seed {seed}")
        params_copy = copy.deepcopy(params)
        params_copy['seed'] = seed
        data_loader, algorithm = initialize(params_copy)
        algorithm.run()

        # Collect final accuracy from algorithm.comm_log
        local_acc_list = []
        for metric_history in algorithm.comm_log['client_metric_history']:
            local_acc_list.append(metric_history['test_accuracy'][-1])
        local_acc_list = np.array(local_acc_list)
        mean_acc = float(np.mean(local_acc_list/100))
        all_accuracies.append(mean_acc)



