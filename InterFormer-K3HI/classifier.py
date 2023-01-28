import torch
import torch.nn as nn

from model_new_classifier import DeepGRU
from dataset.datafactory import DataFactory
from utils.average_meter import AverageMeter  # Running average computation
from utils.logger import log


def get_true_label(class_to_idx):
    idx_orig = {0: "approaching",
                1: "departing",
                2: "kicking",
                3: "pushing",
                4: "shaking",
                5: "exchanging",
                6: "punching",
                7: "pointing"}
    idx_true = []
    for i in range(len(idx_orig)):
        name_orig =idx_orig[i]
        for j in range(len(class_to_idx)):
            name_new =class_to_idx[j]
            if name_new ==name_orig:
                idx_true.append(j)
    return idx_true

def run_batch(batch, model, criterion):
    """
    Runs the forward pass on a batch and computes the loss and accuracy
    """
    examples, lengths, labels = batch

    if use_cuda:
        examples = examples.cuda()
        labels = labels.cuda()

    # Forward and loss computation
    outputs = model(examples, lengths)
    loss = criterion(outputs, labels)

    # Compute the accuracy
    predicted = outputs.argmax(1)
    correct = (predicted == labels).sum().item()
    curr_batch_size = labels.size(0)
    accuracy = correct / curr_batch_size * 100.0

    class_accuracies=[]
    for k in range(8):
        indices = (labels==k)
        pred_class = predicted[indices]
        real_class = labels[indices]
        correct_class = (pred_class == real_class).sum().item()
        class_accuracies.append(correct_class/len(real_class) *100.0)



    return accuracy, curr_batch_size, loss,class_accuracies


use_cuda=True
log.set_dataset_name('sbu')
dataset = DataFactory.instantiate('sbu_test_react', 1)
criterion = nn.CrossEntropyLoss()
model = DeepGRU(dataset.num_features, dataset.num_classes)
if use_cuda:
    model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(f'classifier_model/model_weights'))
model.eval()

train_loader, test_loader = dataset.get_data_loaders(0,
                                                     shuffle=True,
                                                     random_seed=1570254494,
                                                     normalize=True,num_worker=0)
test_meter = AverageMeter()
test_loss_meter = AverageMeter()

for batch in test_loader:

    accuracy, curr_batch_size, loss,class_accuracies = run_batch(batch, model, criterion)
    test_loss_meter.update(loss.item(), curr_batch_size)
    test_meter.update(accuracy, curr_batch_size)

test_accuracy = test_meter.avg

idx_label = get_true_label(dataset.idx_to_class)

log('       accuracy average {top1:.6f} '.format(top1=test_accuracy))
log('       accuracy approaching {top1:.6f} '.format(top1=class_accuracies[idx_label[0]]))
log('       accuracy departing {top1:.6f} '.format(top1=class_accuracies[idx_label[1]]))
log('       accuracy kicking {top1:.6f} '.format(top1=class_accuracies[idx_label[2]]))
log('       accuracy pushing {top1:.6f} '.format(top1=class_accuracies[idx_label[3]]))
log('       accuracy shaking {top1:.6f} '.format(top1=class_accuracies[idx_label[4]]))
log('       accuracy exchanging {top1:.6f} '.format(top1=class_accuracies[idx_label[5]]))
log('       accuracy punching {top1:.6f} '.format(top1=class_accuracies[idx_label[6]]))
log('       accuracy pointing {top1:.6f} '.format(top1=class_accuracies[idx_label[7]]))




