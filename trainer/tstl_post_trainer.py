import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

class tstlIntegrator:
    def __init__(self, model, device): 
        self.model = model.double().to(device)
        self.device = device

    @staticmethod
    def _weighted_average_model_weights(existing_soup, new_state_dict, alpha):
        """
        Performs weighted averaging of model weights:
        soup = (1-alpha) * existing_soup + alpha * new_state_dict
        
        Args:
            existing_soup: Average of previous state dictionaries
            new_state_dict: New state dictionary to add
            alpha: Weight for the new state dictionary (0-1)
        
        Returns:
            Weighted average state dictionary
        """
        if existing_soup is None:
            # If this is the first model, just return it
            return new_state_dict.copy()
        
        weighted_state_dict = {}
        for key in existing_soup.keys():
            if key not in new_state_dict:
                raise ValueError(f"Key {key} missing in new state dictionary")
            
            weighted_state_dict[key] = (1 - alpha) * existing_soup[key] + alpha * new_state_dict[key]
            
        return weighted_state_dict
    
    @staticmethod
    def _average_model_weights(state_dicts):
        if not state_dicts:
            raise ValueError("Empty state dictionaries list")
        
        avg_state_dict = {}
        keys = state_dicts[0].keys()
        
        if not all(set(d.keys()) == set(keys) for d in state_dicts[1:]):
            raise ValueError("Inconsistent state dictionary keys")
        
        for key in keys:
            avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)

        return avg_state_dict

    @staticmethod
    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        medae = np.median(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, medae, r2

    def create_model_soup(self, checkpoint_paths, train_loader, valid_loader, 
                         scaler, ensemble_path, alpha_values=None): 
        """
        Create a model soup with weighted averaging.
        
        Args:
            checkpoint_paths: List of checkpoint paths to load
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            scaler: Dictionary with 'mean' and 'std' for inverse scaling
            ensemble_path: Path to save the final model soup
            alpha_values: List of alpha values to try (default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        """
        if alpha_values is None:
            alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            
        state_dicts = [torch.load(path, map_location=self.device, weights_only=True) 
                      for path in checkpoint_paths]
        
        # Evaluate models
        metrics = []
        for idx, state_dict in enumerate(state_dicts): 
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            with torch.no_grad():
                preds = [self.model(*[b.to(self.device) for b in batch[:-1]]).cpu().numpy() 
                        for batch in train_loader]
            y_pred = np.vstack(preds) * scaler['std'] + scaler['mean']
            y_true = np.vstack([batch[-1].numpy() for batch in train_loader])
            
            train_mae, train_medae, train_r2 = self.metrics(y_true, y_pred)
            normalized_mae = 1 / (1 + train_mae)
            balanced_score = 0.5 * normalized_mae + 0.5 * train_r2
            metrics.append(balanced_score)
            print(f'--- Model {idx + 1} - Train MAE: {train_mae:.4f}, MedAE: {train_medae:.4f}, R2: {train_r2:.4f}, Score: {balanced_score:.4f}')

        # Sort state_dicts according to metrics
        state_dicts = [state_dict for _, state_dict in sorted(zip(metrics, state_dicts), key=lambda x: x[0], reverse=True)]
        print('--- Sort models according to score')
        
        # Initialize variables
        current_soup = None
        best_soup = None
        best_score = float('-inf')
        used_models = []
        
        # Try adding each model with different alpha values
        for i, new_state_dict in enumerate(state_dicts):
            print(f"\n--- Trying to add Model {i+1} to soup")
            best_alpha = None
            best_alpha_score = float('-inf')
            best_alpha_soup = None
            
            # If this is the first model, just add it without trying different alphas
            if current_soup is None:
                print(f"--- First model in soup, adding Model {i+1} directly")
                self.model.load_state_dict(new_state_dict)
                self.model.eval()
                
                # Evaluate on validation set
                with torch.no_grad(): 
                    preds = [self.model(*[b.to(self.device) for b in batch[:-1]]).cpu().numpy() 
                            for batch in valid_loader]
                y_pred = np.vstack(preds) * scaler['std'] + scaler['mean']
                y_true = np.vstack([batch[-1].numpy() for batch in valid_loader])
                
                valid_mae, valid_medae, valid_r2 = self.metrics(y_true, y_pred)
                normalized_mae = 1 / (1 + valid_mae)
                best_alpha_score = 0.5 * normalized_mae + 0.5 * valid_r2
                best_alpha_soup = new_state_dict.copy()
                best_alpha = 1.0
                
                print(f'--- First model - Valid MAE: {valid_mae:.4f}, MedAE: {valid_medae:.4f}, R2: {valid_r2:.4f}, Score: {best_alpha_score:.4f}')
            else:
                # Try different alpha values for this model
                for alpha in alpha_values:
                    # Create weighted soup
                    weighted_soup = self._weighted_average_model_weights(
                        current_soup, new_state_dict, alpha)
                    
                    # Load and evaluate the weighted soup
                    self.model.load_state_dict(weighted_soup)
                    self.model.eval()
                    
                    # Evaluate on training set
                    with torch.no_grad():
                        preds = [self.model(*[b.to(self.device) for b in batch[:-1]]).cpu().numpy() 
                                for batch in train_loader]
                    y_pred = np.vstack(preds) * scaler['std'] + scaler['mean']
                    y_true = np.vstack([batch[-1].numpy() for batch in train_loader])
                    
                    train_mae, train_medae, train_r2 = self.metrics(y_true, y_pred)
                    train_normalized_mae = 1 / (1 + train_mae)
                    train_score = 0.5 * train_normalized_mae + 0.5 * train_r2
                    
                    print(f'--- Alpha: {alpha:.2f} - Train MAE: {train_mae:.4f}, MedAE: {train_medae:.4f}, R2: {train_r2:.4f}, Score: {train_score:.4f}')
                    
                    # Evaluate on validation set
                    with torch.no_grad(): 
                        preds = [self.model(*[b.to(self.device) for b in batch[:-1]]).cpu().numpy() 
                                for batch in valid_loader]
                    y_pred = np.vstack(preds) * scaler['std'] + scaler['mean']
                    y_true = np.vstack([batch[-1].numpy() for batch in valid_loader])
                    
                    valid_mae, valid_medae, valid_r2 = self.metrics(y_true, y_pred)
                    normalized_mae = 1 / (1 + valid_mae)
                    valid_score = 0.5 * normalized_mae + 0.5 * valid_r2
                    
                    print(f'--- Alpha: {alpha:.2f} - Valid MAE: {valid_mae:.4f}, MedAE: {valid_medae:.4f}, R2: {valid_r2:.4f}, Score: {valid_score:.4f}')
                    
                    # Keep track of best alpha for this model
                    if train_score > best_alpha_score:
                        best_alpha = alpha
                        best_alpha_score = train_score
                        best_alpha_soup = weighted_soup
            
            # Check if this model with its best alpha improves overall score
            if best_alpha_score > best_score:
                print(f'--- Improvement found! Adding Model {i+1} with alpha={best_alpha:.2f}, new score: {best_alpha_score:.4f}')
                current_soup = best_alpha_soup
                best_soup = best_alpha_soup
                best_score = best_alpha_score
                used_models.append((i+1, best_alpha))
            else: 
                print(f'--- No improvement with Model {i+1}, keeping previous soup')
        
        # If we found a good soup, save it
        if best_soup:
            self.model.load_state_dict(best_soup)
            torch.save(best_soup, ensemble_path)
            print(f'--- Save ensemble model to {ensemble_path}')
            print(f'--- Used models with alphas: {used_models}')
        else: 
            raise ValueError("No models selected for model soup")

        # Final evaluation on validation set
        self.model.eval()
        with torch.no_grad():
            preds = [self.model(*[b.to(self.device) for b in batch[:-1]]).cpu().numpy() 
                    for batch in valid_loader]
        y_pred = np.vstack(preds) * scaler['std'] + scaler['mean']
        y_true = np.vstack([batch[-1].numpy() for batch in valid_loader])

        best_metrics = {}
        best_metrics['MAE'], best_metrics['MedAE'], best_metrics['R2'] = self.metrics(y_true, y_pred)
        print('--- Best metrics:', best_metrics)
        return best_metrics