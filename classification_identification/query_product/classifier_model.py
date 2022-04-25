# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
#  
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import random
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score, f1_score
import os
import pathlib

def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class QueryProductClassifier(nn.Module):

    def __init__(self, size_petrained=768, dense_hidden_dim=126, num_dense_layers=1, num_labels=1, dropout_rate=0.1):
        super(QueryProductClassifier, self).__init__()
        self.num_labels = 1 if num_labels <= 2 else num_labels
        self.size_petrained = size_petrained * 2
        fc_layers = []
        prev_dim = self.size_petrained
        self.dropout_embedding = nn.Dropout(dropout_rate)
        for _ in range(num_dense_layers):
            fc_layers.append(nn.Linear(prev_dim, dense_hidden_dim, bias=True))
            prev_dim = dense_hidden_dim
        fc_layers.append(nn.Linear(prev_dim, self.num_labels))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, query_embedding, Product_embedding):
        # query_embedding: [batch_size, num_features]
        # product_embedding: [batch_size, num_features]
        embedding = torch.cat((query_embedding, Product_embedding), 1) # [batch_size, num_features * 2]
        embedding = self.dropout_embedding(embedding) # [batch_size, num_features * 2]
        logits = self.fc(embedding).squeeze(-1) # [batch_size, num_labels]
        return logits

def train(model, train_inputs, validation_inputs, path_model, device='cpu', batch_size=128, weight_decay=0.01, num_train_epochs=4, 
    lr=5e-5, eps=1e-8, num_warmup_steps=0, max_grad_norm=1, validation_steps=250, random_seed=42):
    
    set_seed(random_seed=random_seed)

    """ Step 0: prapare data loaders and model """
    train_sampler = RandomSampler(train_inputs)
    train_dataloader = DataLoader(train_inputs, sampler=train_sampler, batch_size=batch_size)
    validation_sampler = SequentialSampler(validation_inputs)
    validation_dataloader = DataLoader(validation_inputs, sampler=validation_sampler, batch_size=batch_size)
    model.to(device)
    
    """ Step 1: preparere optimizer """
    num_training_batches = len(train_dataloader)
    total_training_steps = num_training_batches * num_train_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    """ Step 2: preparere variables """
    validation_metric = np.empty(len(validation_dataloader))
    validation_loss = np.empty_like(validation_metric)
    
    best_metric_value = 0.0
    best_model = None
    input_metric = {
        'y_true' : None, 
        'y_pred' : None,
    }

    if model.num_labels > 2:
        criterion = nn.CrossEntropyLoss()
        metric = accuracy_score
    else:
        criterion = nn.BCELoss()
        metric = f1_score
        input_metric['average'] = 'macro'
    
    """ Step 3: experiments """
    
    for idx_epoch in range(0, num_train_epochs):
        
        """ Step 3.1: Training """
        for (idx_train_batch, train_batch) in enumerate(train_dataloader):
            model.train()
            # 0: query_embedding, 1: product_embedding, 2: labels 
            labels = train_batch[2].to(device)
            optimizer.zero_grad()
            logits = model(train_batch[0].to(device), train_batch[1].to(device))

            if model.num_labels > 2:
                loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
                logits = logits.detach().cpu().numpy()
                hypothesis = np.argmax(logits, axis=1)
            else:
                output = torch.sigmoid(logits)
                output, labels = output.type(torch.FloatTensor), labels.type(torch.FloatTensor)
                loss = criterion(output, labels)
                output = output.detach().cpu().numpy()
                hypothesis = np.digitize(output, [0.5])
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm) # clipping gradient for avoiding exploding gradients
            optimizer.step()
            scheduler.step()
            expected_predictions = labels.detach().cpu().numpy()

            input_metric['y_true'] = expected_predictions
            input_metric['y_pred'] = hypothesis

            training_metric = metric(**input_metric)

            if idx_train_batch % validation_steps == 0:
                model.eval()
                print(f"Training - Epoch {idx_epoch+1}/{num_train_epochs}, Batch: {idx_train_batch+1}/{num_training_batches}, Loss: {loss:.3f} Metric:{training_metric:.3f}")
                """ Step 3.2: evaluating """
                for (idx_validation_batch, validation_batch) in enumerate(validation_dataloader):
                    # 0: query_embedding, 1: product_embedding, 2: labels
                    labels = validation_batch[2].to(device)
                    with torch.no_grad():
                        logits = model(validation_batch[0].to(device), validation_batch[1].to(device))
                    if model.num_labels > 2:
                        loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
                        logits = logits.detach().cpu().numpy()
                        hypothesis = np.argmax(logits, axis=1)
                    else:
                        output = torch.sigmoid(logits)
                        output, labels = output.type(torch.FloatTensor), labels.type(torch.FloatTensor)
                        loss = criterion(output, labels)
                        output = output.detach().cpu().numpy()
                        hypothesis = np.digitize(output, [0.5])
                    expected_predictions = labels.detach().cpu().numpy()
                    input_metric['y_true'] = expected_predictions
                    input_metric['y_pred'] = hypothesis
                    validation_metric[idx_validation_batch] = metric(**input_metric)
                    validation_loss[idx_validation_batch] = loss
                current_validation_metric = np.mean(validation_metric)

                print(f"Validation - Epoch {idx_epoch+1}/{num_train_epochs}, Batch: {idx_train_batch+1}/{num_training_batches}, Loss: {np.mean(validation_loss):.3f}, Metric:{np.mean(validation_metric):.3f}")
                
                if current_validation_metric > best_metric_value:
                    best_metric_value = current_validation_metric
                    best_model = model
                    """ Step 4: store model """
                    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True)
                    torch.save(best_model.state_dict(), os.path.join(path_model, "pytorch_model.bin"))