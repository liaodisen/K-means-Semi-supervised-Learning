from typing import Any, Dict, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class KmeanLoss(nn.Module):
    def __init__(self):
        super(KmeanLoss, self).__init__()

    def forward(self, dist):
        loss = torch.mean(dist.min(dim=1)[0])
        return loss
    
class KmeanNet(nn.Module):
    def __init__(self, kmean):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.kmean = kmean
        resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        out = self.features(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        deepfeat = out.view(out.size(0), -1)
        clusters = self.kmean(deepfeat)
        return clusters, clusters
    

class KmeanNetLightning(LightningModule):
    def __init__(
            self,
            kmean : nn.modules, 
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler
            ):
        
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.kmean_net = KmeanNet(kmean)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kmean_loss = KmeanLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x):
        return self.kmean_net(x)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        
        # Find indices of labeled data (where y != -1)
        labeled_indices = (y != -1).nonzero(as_tuple=True)[0]
        
        # Filter x and y for labeled data
        labeled_x = x[labeled_indices]
        labeled_y = y[labeled_indices]
        
        # Forward pass using only labeled data
        logits, _ = self.forward(x)
        labeled_logits = logits[labeled_indices]
        
        # Compute loss using only labeled data
        loss = self.ce_loss(labeled_logits, labeled_y) + 0.1 * self.kmean_loss(logits)
        
        # Compute predictions for all data
        preds = torch.argmax(logits, dim=1)

        # compute predictions for labeled_data
        labeled_preds = torch.argmax(labeled_logits, dim=1)
        
        return loss, preds, labeled_y, labeled_preds

    def training_step(self, batch, batch_idx):
        loss, preds, targets, labeled_preds = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(labeled_preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets, labeled_preds = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(labeled_preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, labeled_preds = self.model_step(batch)

        assert preds == labeled_preds, "there is unlabeled data"
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}



if __name__ == "__main__":
    _ = KmeanNetLightning(None, None, None, None)
