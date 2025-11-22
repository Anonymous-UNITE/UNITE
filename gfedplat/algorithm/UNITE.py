# -*- coding: utf-8 -*-
import gfedplat as fp
import copy
import numpy as np
import torch
import time
import json
import os

# Check and set device
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU is available. Using CUDA.")
else:
    device = 'cpu'
    print("GPU not available. Using CPU.")

class UNITE(fp.Algorithm):
    def __init__(self,
                 name='UNITE',
                 data_loader=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 client_test=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 params=None,
                 *args,
                 **kwargs):

        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay'])
        # Instantiation is completed by calling the parent class constructor.
        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num, client_test=client_test, max_comm_round=max_comm_round, max_training_num=max_training_num, epochs=epochs, save_name=save_name, outFunc=outFunc, write_log=write_log, dishonest=dishonest, params=params, *args, **kwargs)


        # Parameters for new algorithm
        self.alpha = params.get('personalization_alpha', 0.5) if params else 0.5  # alpha for personalized embeddings, small value e.g. 0.5
        self.eps = params.get('uncertainty_eps', 1e-8) if params else 1e-8  # epsilon for uncertainty calculation
        self.prev_global_embeddings = None  # to store previous global embeddings for personalization

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    def compute_local_uncertainty(self, embeddings):
        """
        Step 2: Local Uncertainty Estimation (Norm-based)
        Args:
            embeddings: list of tensors, one per client, each tensor of shape (num_samples, embedding_dim)
        Returns:
            uncertainties: list of tensors, normalized uncertainties per client
        """
        uncertainties = []
        for emb in embeddings:
            norms = torch.norm(emb, dim=1)  # ||z_i,j||
            u = 1 / (norms + self.eps)  # u_i,j = 1 / (||z_i,j|| + eps)
            u_norm = (u - u.min()) / (u.max() - u.min() + self.eps)  # normalize within client
            uncertainties.append(u_norm)
        return uncertainties

    def compute_disagreement(self, embeddings_per_sample):
        """
        Step 3: Compute Mean Embeddings
        Args:
            embeddings_per_sample: list of lists, each inner list has embeddings from all clients for one sample
        Returns:
            mean_embeddings: tensor of mean embeddings per sample
        """
        mean_embeddings = []
        for sample_embs in embeddings_per_sample:
            emb_tensor = torch.stack(sample_embs)  # (num_clients, embedding_dim)
            mean_emb = emb_tensor.mean(dim=0)  # z_bar_j
            mean_embeddings.append(mean_emb)
        return torch.stack(mean_embeddings)

    def compute_residuals(self, embeddings_per_sample, mean_embeddings, uncertainties):
        """
        Step 4: Residual Extraction
        Args:
            embeddings_per_sample: list of lists, embeddings per sample per client
            mean_embeddings: tensor of mean embeddings per sample
            uncertainties: list of normalized uncertainties per client
        Returns:
            residuals: list of lists, residuals per sample per client
        """
        residuals = []
        for j, sample_embs in enumerate(embeddings_per_sample):
            residuals_j = []
            for i, emb in enumerate(sample_embs):
                r = emb - mean_embeddings[j]  # r_i,j
                residuals_j.append(r)
            residuals.append(residuals_j)
        return residuals

    def aggregate_uncertainty_calibrated(self, embeddings_per_sample, scaled_residuals, uncertainties):
        """
        Step 5: Server-Side Aggregation (Uncertainty-Calibrated Ensemble)
        Args:
            embeddings_per_sample: list of lists, embeddings per sample per client
            scaled_residuals: list of lists, scaled residuals per sample per client
            uncertainties: list of normalized uncertainties per client
        Returns:
            global_embeddings: tensor of global ensemble embeddings per sample
        """
        global_embeddings = []
        for j, sample_embs in enumerate(embeddings_per_sample):
            num_clients = len(sample_embs)
            # z_j* = [sum_i (1 - u_hat_i,j) * z_i,j + sum_i r_i,j'] / [sum_i (1 - u_hat_i,j) + sum_i u_hat_i,j]
            numerator = torch.zeros_like(sample_embs[0])
            denominator = 0.0
            for i in range(num_clients):
                weight_reliable = 1 - uncertainties[i][j]
                weight_uncertain = uncertainties[i][j]
                numerator += weight_reliable * sample_embs[i] + weight_uncertain * scaled_residuals[j][i]
                denominator += weight_reliable + weight_uncertain
            z_star = numerator / denominator
            global_embeddings.append(z_star)
        return torch.stack(global_embeddings)

    def compute_personalized_embeddings(self, embeddings_per_sample, global_embeddings):
        """
        Step 6: Personalized Ensemble Embeddings
        Args:
            embeddings_per_sample: list of lists, local embeddings per sample per client
            global_embeddings: tensor of global embeddings per sample
        Returns:
            personalized_embeddings: list of tensors, one per client, embeddings per sample
        """
        personalized_embeddings = []
        for i in range(len(embeddings_per_sample[0])):  # num_clients
            client_embs = []
            for j in range(len(embeddings_per_sample)):  # num_samples
                z_local = embeddings_per_sample[j][i]
                z_global = global_embeddings[j]
                z_personal = self.alpha * z_local + (1 - self.alpha) * z_global
                client_embs.append(z_personal)
            personalized_embeddings.append(torch.stack(client_embs))
        return personalized_embeddings

    def fine_tune_client_classifier(self, client, embeddings, labels):
        """
        Fine-tune a client's classifier on personalized embeddings.
        Args:
            client: the client object
            embeddings: tensor of personalized embeddings for the client
            labels: tensor of labels
        """
        model = client.module.model
        # Freeze feature extractor
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'predictor'):
            for param in model.predictor.parameters():
                param.requires_grad = True
        if hasattr(model, 'classifier'):
            for param in torch.nn.Sequential(*list(model.classifier.children())[:-1]).parameters():
                param.requires_grad = True

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        features = embeddings
        if hasattr(model, 'classifier'):
            with torch.no_grad():
                temp_seq = torch.nn.Sequential(*list(model.classifier.children())[:-1])
                features = temp_seq(embeddings)

        min_len = min(features.shape[0], labels.shape[0])
        features = features[:min_len]
        labels_trimmed = labels[:min_len]
        labels_trimmed = labels_trimmed.to(self.device)

        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model.predictor(features)
            loss = criterion(outputs, labels_trimmed)
            loss.backward()
            optimizer.step()

        # Unfreeze
        for param in model.parameters():
            param.requires_grad = True



    def run(self):
        print("UNITE run method started")  # Diagnostic print to confirm run is called
        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, _ = self.train_a_round()
            com_time_end = time.time()
            cal_time_start = time.time()
            old_model = self.module.span_model_params_to_vec()

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

            print(f"Online client list length: {len(self.online_client_list)}")  # Diagnostic print for client list size

    def aggregate_models_to_clients(self, m_locals):
        """
        Aggregate feature extractor to each client's model, preserving personalized classifiers.
        Args:
            m_locals: list of local model objects
        """
        # Extract parameter dicts from model objects
        state_dicts = []
        for model in m_locals:
            if hasattr(model, 'state_dict') and callable(getattr(model, 'state_dict')):
                state_dicts.append(model.state_dict())
            elif hasattr(model, 'model'):
                inner_model = model.model
                if hasattr(inner_model, 'state_dict') and callable(getattr(inner_model, 'state_dict')):
                    state_dicts.append(inner_model.state_dict())
                else:
                    raise AttributeError(f"Inner model object {inner_model} has no state_dict method")
            elif isinstance(model, dict):
                state_dicts.append(model)
            else:
                raise AttributeError(f"Model object {model} has no state_dict method and is not a state_dict")

        # Identify excluded keys (personalized classifiers)
        excluded_keys = set()
        if hasattr(self.module.model, 'predictor'):
            for name, _ in self.module.model.predictor.named_parameters():
                excluded_keys.add(f'predictor.{name}')
        if hasattr(self.module.model, 'classifier'):
            for name, _ in self.module.model.classifier.named_parameters():
                excluded_keys.add(f'classifier.{name}')

        # Simple average aggregation only for non-excluded keys
        aggregated = copy.deepcopy(state_dicts[0])
        tensor_keys = [key for key, value in aggregated.items() if torch.is_tensor(value)]

        num_clients = len(state_dicts)
        for key in tensor_keys:
            if key not in excluded_keys:
                aggregated[key] = torch.zeros_like(aggregated[key], dtype=torch.float32)
                for local_params in state_dicts:
                    aggregated[key] += local_params[key].float() / num_clients

        # Update each client's model with aggregated feature extractor
        for client in self.online_client_list:
            client_state = client.module.model.state_dict()
            for key in tensor_keys:
                if key not in excluded_keys:
                    client_state[key] = aggregated[key]
            client.module.model.load_state_dict(client_state)

    def train_a_round(self):
        """
        Conduct a round of training.
        """

        # Reset accuracy sums and counts for the current round to avoid accumulation across rounds
        round_num = self.current_comm_round

        com_time_start = time.time()

        # Call parent train() to get model objects and losses
        m_locals, l_locals = super().train()

        com_time_end = time.time()

        # Collect embeddings from client data for new algorithm
        embeddings = []  # list of tensors, one per client
        labels_list = []  # list of labels for samples
        for client in self.online_client_list:
            # Use client's local training data to get embeddings
            if hasattr(client, 'local_training_data') and client.local_training_data is not None:
                data_loader = client.local_training_data
            elif hasattr(client, 'local_test_data') and client.local_test_data is not None:
                data_loader = client.local_test_data
            else:
                # Skip client if no data available
                continue

            client_embeddings = []
            client_labels = []
            with torch.no_grad():
                for batch in data_loader:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    # Use the model's return_embedding option to get embeddings
                    emb = self.module.model(inputs, return_embedding=True)
                    client_embeddings.append(emb)
                    client_labels.append(labels)
                    break  # only one batch for simplicity, or collect all
            if client_embeddings:  # Only add if we got embeddings
                embeddings.append(torch.cat(client_embeddings, dim=0))
                labels_list.append(torch.cat(client_labels, dim=0))

        # Transpose to per sample - use minimum batch size across all clients
        if not embeddings:
            # Skip the UNITE algorithm if no embeddings were collected
            print("Warning: No embeddings collected from clients, skipping UNITE algorithm")
            return m_locals, l_locals

        # Find minimum number of samples across all clients to avoid index errors
        min_samples = min(emb.shape[0] for emb in embeddings)
        embeddings_per_sample = []
        for j in range(min_samples):
            sample_embs = [emb[j] for emb in embeddings]
            embeddings_per_sample.append(sample_embs)

        # Step 2: Compute local uncertainties
        uncertainties = self.compute_local_uncertainty(embeddings)

        # Step 3: Compute mean embeddings
        mean_embeddings = self.compute_disagreement(embeddings_per_sample)

        # 1 Compute personalized local embeddings (light)
        if self.prev_global_embeddings is not None:
            prev_len = self.prev_global_embeddings.shape[0]
            z_personal_local = []
            for j in range(len(embeddings_per_sample)):
                sample_embs = embeddings_per_sample[j]
                if j < prev_len:
                    z_prev_global = self.prev_global_embeddings[j]
                    personalized_sample = [self.alpha * emb + (1 - self.alpha) * z_prev_global for emb in sample_embs]
                else:
                    personalized_sample = sample_embs  # no personalization for extra samples beyond previous round's min_samples
                z_personal_local.append(personalized_sample)
        else:
            z_personal_local = embeddings_per_sample  # first round, no previous global

        # Compute calibration based on personalized local embeddings
        # Step 3: Compute mean embeddings on personalized local
        mean_embeddings = self.compute_disagreement(z_personal_local)

        # Step 4: Compute residuals on personalized local
        residuals = self.compute_residuals(z_personal_local, mean_embeddings, uncertainties)

        # Step 5: Aggregate uncertainty-calibrated global embeddings
        global_embeddings = self.aggregate_uncertainty_calibrated(z_personal_local, residuals, uncertainties)

        # Store for next round
        self.prev_global_embeddings = global_embeddings.clone()

        # Step 6: Compute personalized embeddings (final)
        personalized_embeddings = self.compute_personalized_embeddings(embeddings_per_sample, global_embeddings)

        # Step 7: Fine-tune each client's classifier on their personalized embeddings
        for i, client in enumerate(self.online_client_list):
            if i < len(personalized_embeddings):
                labels = labels_list[i] if i < len(labels_list) else labels_list[0]
                self.fine_tune_client_classifier(client, personalized_embeddings[i], labels)

        # Evaluate gradients and losses for custom logic
        g_locals, l_locals_eval = self.evaluate()

        # Perform aggregation to clients
        self.aggregate_models_to_clients(m_locals)

        self.current_training_num += 1

        self.communication_time += com_time_end - com_time_start

        return m_locals, l_locals
