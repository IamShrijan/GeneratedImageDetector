from typing import Dict, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import mlflow
from sklearn.model_selection import KFold, ParameterGrid
import torchvision
from tqdm import tqdm
from src.models.model_factory import ModelFactory
import torch.nn as nn
import random
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose
import torch

class TestImageCSVLoader(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['id']  # 'id' column has the image path
        img_path = f'dataset/{img_path}'
        image = read_image(img_path).float() / 255.0  # Normalize to [0, 1]

        if self.transform:
            image = self.transform(image)

        return image, img_path  # Return image path so you can display it if needed


class ModelTrainer:
    def __init__(self, config, model_factory: ModelFactory):
        self.config = config
        self.model_factory = model_factory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        mlflow.set_tracking_uri('file:./experiments/mlruns')

    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, # Allow model to take multiple types of models
                    optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Modify the target tensor to have shape [batch_size, 1]
            targets: torch.Tensor = targets.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = self.criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            predictions = (outputs >= self.config['training']['classification_threshold']).float()  # If output >= 0.5, predict 1, else 0
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
            
        return total_loss/len(train_loader), correct/total


    def _evaluate(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Modify the target tensor to have shape [batch_size, 1]
            targets: torch.Tensor = targets.float().unsqueeze(1)
            
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()

            predictions = (outputs >= self.config['training']['classification_threshold']).float()
            total += targets.size(0)

            correct += predictions.eq(targets).sum().item()
        
        return total_loss/len(val_loader), correct/total

    def _train_fold(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, params: Dict) -> float:
        """Train and evaluate one fold"""
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=params['learning_rate'])
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            print(f'Epoch {epoch + 1}')
            # Train
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer)
            
            # Evaluate
            val_loss, val_acc = self._evaluate(model, val_loader)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            # Update the history for tracking metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['training']['early_stopping']['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
                
        return best_val_acc
    

    def train_and_tune(self, train_dataset, hyperparameter_tuning: bool = False):
        """
        Train and tune the model.

        :param train_dataset: The training dataset
        :param hyperparameter_tuning: Boolean flag to enable/disable hyperparameter tuning
        :return: Best hyperparameters and corresponding score if tuning is enabled, else None
        """
        mlflow.set_experiment("model_tuning")
        
        best_params = None
        best_score = float('-inf')
        
        if hyperparameter_tuning:
            # Perform hyperparameter tuning
            for model_name in self.config['model_names']:
                print(f'Training {model_name}')

                # Safely access 'parameter_search' if it exists, else use an empty dictionary
                parameter_search = self.config['models'].get(model_name, {}).get('parameter_search', {})

                # Combine the hyperparameter search spaces
                combined_param = {**self.config['hyperparameter_search'], **parameter_search}

                model_hyperparams = parameter_search.keys()
                param_grid = ParameterGrid(combined_param)
                
                for params in param_grid:
                    print(f'Hyperparameters: {params}')
                    with mlflow.start_run():
                        mlflow.log_params(params)
                        
                        # Choose the model and its parameters from config
                        model_params = {
                            **self.config['models'][model_name].get('base_params', {}), 
                        }
                        
                        # Passing the model hyperparameters to model factory for instantiating the model
                        for param in params:
                            if param in model_hyperparams:
                                model_params[param] = params[param]

                        # Create and train the model
                        model = self.model_factory.get_model(
                            model_name,
                            **model_params
                        ).to(self.device)

                        # Restore batch_size for cross-validation
                        score = self._cross_validate(model, train_dataset, params)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_model = model
                        
                        mlflow.log_metric("cv_score", score)
            
        else:
            # Hyperparameter tuning is disabled, train the model with default config
            print("Skipping hyperparameter tuning and using default parameters.")

            for model_name in self.config['model_names']:
                print(f'Training {model_name} with default parameters')
                
                # Safely access model configuration
                model_config = self.config['models'][model_name]
                
                # If the model configuration is None or empty, skip this model
                if model_config is None or not model_config:
                    print(f"Skipping {model_name} as it is not properly configured.")
                    continue
                
                # Safely access 'parameter_search' if it exists, else use an empty dictionary
                parameter_search = model_config.get('parameter_search', {})
                if parameter_search:
                    model_hyperparams = parameter_search.keys()
                else: 
                    parameter_search = {}
                    model_hyperparams = None
                
                # Build the hyperparameters dictionary
                hyperparams = {
                    **{key: val[0] for key, val in self.config['hyperparameter_search'].items()},
                    **{k: v[0] for k, v in parameter_search.items()}
                }
                
                model_params = model_config.get('base_params', {})
                if not model_params:
                    model_params = {}

                # Update model_params with the hyperparameters from the configuration
                for param in hyperparams:
                    if model_hyperparams and param in model_hyperparams:
                        model_params[param] = hyperparams[param]
                
                # Create and train the model
                model = self.model_factory.get_model(
                    model_name,
                    **model_params
                ).to(self.device)

                score = self._cross_validate(model, train_dataset, hyperparams)
                
                if score > best_score:
                    best_score = score
                    best_params = model_params
                    best_model = model

        return best_params, best_score, best_model


    
    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def _cross_validate(self, model, dataset, params):
        # Cross-validation logic
        n_folds = self.config['training']['cross_validation']['n_splits']
        kfold = KFold(
            n_splits=n_folds,
            shuffle=self.config['training']['cross_validation']['shuffle']
        )

        print(f'Performing KFold Cross Validation with {n_folds} folds')
        
        scores = []
        for fold, (train_idxs, val_idxs) in enumerate(kfold.split(dataset)):
            # print(f'train indices: {len(train_idxs)}')
            # print(f'val indices: {len(val_idxs)}')
            # # Verifying common indices, if any
            # common_indices = set(train_idxs).intersection(val_idxs)
            # print(f'common indices: {common_indices}')

            train_loader = DataLoader(
                dataset,
                batch_size=params['batch_size'],
                sampler=SubsetRandomSampler(train_idxs)
            )
            # print(f"batch size: {train_loader.batch_size}")
            # print(f'batches in train loader: {len(train_loader)}')
            # print(f'total samples in train loader: {len(train_loader.dataset)}')

            val_loader = DataLoader(
                dataset,
                batch_size=params['batch_size'],
                sampler=SubsetRandomSampler(val_idxs)
            )

            # print(f'batches in val loader: {len(val_loader)}')
            # print(f'total samples in val loader: {len(val_loader.dataset)}')
            
            fold_score = self._train_fold(model, train_loader, val_loader, params)
            print(f'Accuracy obtained on Fold {fold + 1}: {fold_score}')
            scores.append(fold_score)
            
        print(f'Mean score obtained from KFold: {np.mean(scores)}')
        return np.mean(scores)


    def train_final_model(self, model: nn.Module, train_loader: DataLoader, 
                            val_loader: DataLoader) -> nn.Module:
            """Train the final model with the best parameters"""
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.config['training']['optimizer']['learning_rate']
            )
            best_val_acc = 0
            best_state_dict = None
            
            for epoch in range(self.config['training']['epochs']):
                train_loss, train_acc = self._train_epoch(model, train_loader, optimizer)
                val_loss, val_acc = self._evaluate(model, val_loader)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state_dict = model.state_dict().copy()
                    
                # Log metrics
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                mlflow.log_metrics({
                    "final_train_loss": train_loss,
                    "final_train_acc": train_acc,
                    "final_val_loss": val_loss,
                    "final_val_acc": val_acc
                }, step=epoch)
                
            # Load best model
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)
                
            return model
    
    @staticmethod
    def test_random(model_path: str, model: nn.Module, num_images: int = 5, csv_path: str = "test.csv"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])

        dataset = TestImageCSVLoader(csv_path, transform=transform)
        indices = random.sample(range(len(dataset)), num_images)
        sample_subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(sample_subset, batch_size=num_images, shuffle=False)

        images, paths = next(iter(loader))
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = (outputs >= 0.5).float().squeeze().cpu().numpy()

        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            img = images[i].cpu().numpy().transpose((1, 2, 0)) * 0.5 + 0.5  # Unnormalize
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.title(f"Pred: {'AI' if preds[i] else 'Human'}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
