import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from itertools import cycle

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, drop=0.2):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SELU(),
            nn.Dropout(drop)
        )

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out += residual
        return F.selu(out)


class ResNetTissueClassifier(nn.Module):
    def __init__(self, input_dim, num_tissues, hidden_dims=[512, 256, 128, 64], drop=0.2):
        super(ResNetTissueClassifier, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SELU(),
            nn.Dropout(drop)
        )

        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], drop)
            )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dims[-1] // 2, num_tissues)
        )

        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )

    def forward(self, x, alpha=1.0):
        features = self.input_layer(x)

        for res_block in self.res_blocks:
            features = res_block(features)

        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        tissue_pred = self.classifier(attended_features)

        reversed_features = GradientReversal.apply(attended_features, alpha)
        domain_pred = self.domain_discriminator(reversed_features)
        return tissue_pred, domain_pred, attended_features


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def calculate_class_weights(dataloader):
    all_labels = []
    for batch in dataloader:
        all_labels.extend(batch[1].numpy())

    num_classes = len(config.tissue_map)
    class_counts = np.bincount(all_labels, minlength=num_classes)
    total_samples = len(all_labels)

    class_weights = total_samples / (num_classes * (class_counts + 1))

    class_weights = np.clip(class_weights, None, 10)

    return torch.FloatTensor(class_weights)

def train_tissue_classifier(model, s_dataloaders, t_dataloaders, device,
                            num_epochs=100, lr=0.001, alpha=1.0, weight_decay=1e-4):
    s_train = s_dataloaders[0]
    s_test = s_dataloaders[1]
    t_train = t_dataloaders[0]
    t_test = t_dataloaders[1]

    source_weights = calculate_class_weights(s_train).to(device)
    target_weights = calculate_class_weights(t_train).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    tissue_criterion = nn.CrossEntropyLoss(weight=source_weights)
    target_tissue_criterion = nn.CrossEntropyLoss(weight=target_weights)
    domain_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for (source_batch, target_batch) in zip(s_train, cycle(t_train)):
            source_inputs = source_batch[0].to(device)
            source_labels = source_batch[1].to(device)

            target_inputs = target_batch[0].to(device)
            target_labels = target_batch[1].to(device)

            source_domain_labels = torch.zeros(source_inputs.size(0), 1).to(device)
            target_domain_labels = torch.ones(target_inputs.size(0), 1).to(device)

            optimizer.zero_grad()

            source_tissue_pred, source_domain_pred, _ = model(source_inputs)
            source_tissue_loss = tissue_criterion(source_tissue_pred, source_labels)
            source_domain_loss = domain_criterion(source_domain_pred, source_domain_labels)

            target_tissue_pred, target_domain_pred, _ = model(target_inputs)
            target_tissue_loss = target_tissue_criterion(target_tissue_pred, target_labels)
            target_domain_loss = domain_criterion(target_domain_pred, target_domain_labels)

            total_loss = source_tissue_loss + target_tissue_loss + alpha * (source_domain_loss + target_domain_loss)

            total_loss.backward()
            optimizer.step()

            _, predicted = source_tissue_pred.max(1)
            train_total += source_labels.size(0)
            train_correct += predicted.eq(source_labels).sum().item()
            train_loss += source_tissue_loss.item()

        train_acc = train_correct / train_total

        model.eval()

        # Evaluate source domain training set
        s_train_correct = 0
        s_train_total = 0
        with torch.no_grad():
            for batch in s_train:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)
                s_train_total += labels.size(0)
                s_train_correct += predicted.eq(labels).sum().item()
        s_train_acc = s_train_correct / s_train_total

        # Evaluate source domain test set
        s_test_correct = 0
        s_test_total = 0
        with torch.no_grad():
            for batch in s_test:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)
                s_test_total += labels.size(0)
                s_test_correct += predicted.eq(labels).sum().item()
        s_test_acc = s_test_correct / s_test_total

        # Evaluate target domain training set
        t_train_correct = 0
        t_train_total = 0
        with torch.no_grad():
            for batch in t_train:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)
                t_train_total += labels.size(0)
                t_train_correct += predicted.eq(labels).sum().item()
        t_train_acc = t_train_correct / t_train_total

        # Evaluate target domain test set
        t_test_correct = 0
        t_test_total = 0
        with torch.no_grad():
            for batch in t_test:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                outputs, _, _ = model(inputs)
                _, predicted = outputs.max(1)
                t_test_total += labels.size(0)
                t_test_correct += predicted.eq(labels).sum().item()
        t_test_acc = t_test_correct / t_test_total

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'Train Loss: {train_loss / len(s_train):.4f}, Train Acc: {train_acc:.4f}')
            print(f'Source Domain - Train Acc: {s_train_acc:.4f}, Test Acc: {s_test_acc:.4f}')
            print(f'Target Domain - Train Acc: {t_train_acc:.4f}, Test Acc: {t_test_acc:.4f}')

    return model



def save_feat(model, s_dataloaders, t_dataloaders, device):
    model.eval()
    s_train = s_dataloaders[0]
    t_train = t_dataloaders[0]

    source_features = []
    source_labels = []
    target_features = []
    target_labels = []

    with torch.no_grad():
        for batch in s_train:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            features = model.input_layer(inputs)
            for res_block in model.res_blocks:
                features = res_block(features)

            source_features.append(features)
            source_labels.append(labels)

    with torch.no_grad():
        for batch in t_train:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            features = model.input_layer(inputs)
            for res_block in model.res_blocks:
                features = res_block(features)

            target_features.append(features)
            target_labels.append(labels)

    source_features = torch.cat(source_features, dim=0)
    target_features = torch.cat(target_features, dim=0)

    np.save('source_features.npy',source_features.cpu().detach().numpy())
    np.save('target_features.npy',target_features.cpu().detach().numpy())




