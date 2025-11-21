# -*- coding: utf-8 -*-
import gfedplat as fp
import os


if __name__ == '__main__':
    import copy
    import numpy as np
    import json
    params = fp.read_params()
    num_runs = params.get('num_runs', 3)
    seeds = list(range(1, num_runs + 1))
    all_accuracies = []

    all_runs_acc_per_round = []

    for seed in seeds:
        print(f"Run with seed {seed}")
        params_copy = copy.deepcopy(params)
        params_copy['seed'] = seed
        data_loader, algorithm = fp.initialize(params_copy)
        algorithm.save_folder = data_loader.nickname + '/C' + str(params_copy['C']) + '/' + params_copy['module'] + '/' + params_copy['algorithm'] + '/'
        if not os.path.exists(algorithm.save_folder):
            os.makedirs(algorithm.save_folder)
        algorithm.save_name = 'seed' + str(params_copy['seed']) + ' N' + str(data_loader.pool_size) + ' C' + str(params_copy['C']) + ' ' + algorithm.save_name
        algorithm.run()

        # Collect accuracy per round from algorithm.comm_log
        rounds_acc = []
        for round_idx in range(algorithm.current_comm_round - 1):
            local_acc_list = []
            for metric_history in algorithm.comm_log['client_metric_history']:
                local_acc_list.append(metric_history['test_accuracy'][round_idx])
            local_acc_list = np.array(local_acc_list)
            mean_acc = float(np.mean(local_acc_list/100))
            rounds_acc.append(mean_acc)
        all_runs_acc_per_round.append(rounds_acc)

    # Pad shorter runs if any (should not happen if all runs have same rounds)
    max_rounds = max(len(acc) for acc in all_runs_acc_per_round)
    for acc in all_runs_acc_per_round:
        while len(acc) < max_rounds:
            acc.append(acc[-1])

    # Compute average per round
    avg_acc_per_round = []
    for round_idx in range(max_rounds):
        round_accs = [run_acc[round_idx] for run_acc in all_runs_acc_per_round]
        avg_acc_per_round.append(float(np.mean(round_accs)))

    avg_accuracy = avg_acc_per_round[-1]

    # Save average accuracy per round to JSON file
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    filename = 'average_accuracy_per_round_{}.json'.format(params['algorithm'])
    save_path = os.path.join(save_dir, filename)

    avg_accuracy_record = {
        'num_runs': len(seeds),
        'average_accuracy_per_round': avg_acc_per_round
    }

    with open(save_path, 'w') as f:
        json.dump(avg_accuracy_record, f, indent=4)
