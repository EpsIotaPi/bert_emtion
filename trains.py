import torch
from torch import nn
from tqdm import tqdm
from ax import optimize

from dataset import *
from sklearn.metrics import classification_report
from models import *

# 判断是否能使用gpu版本
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


params = HyperParams()

model = BertWwmModel(params, "", False).to(device)

def trains(model, newParams, train_loader, optimizer, loss_func, isOptimzeHP):
    if isOptimzeHP:
        params.learn_rate = newParams["learn_rate"]
        params.dropout_rate = newParams["dropout_rate"]
        params.rnn_layers = newParams["rnn_layers"]
        logger.info("change params to:")
        logger.info(f"learn rate:   {params.learn_rate}")
        logger.info(f"dropout rate: {params.dropout_rate}")
        logger.info(f"rnn layers:   {params.rnn_layers}")

    for epoch in range(params.epochs):
        losses, y_trues, y_predicts = [], [], []
        for sample_text, sample_label in tqdm(train_loader):
            optimizer.zero_grad()
            sample_sentence = {k: v.squeeze(1).long().to(device) for k, v in sample_text.items()}
            sample_label = sample_label.long().to(device)
            prob = model(sample_sentence)
            loss = loss_func(prob, sample_label.view(-1))
            losses.append(loss.item())
            y_trues += sample_label.cpu().view(-1).numpy().tolist()
            y_predicts += prob.cpu().argmax(1).numpy().tolist()
            nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
            loss.backward()
            optimizer.step()
        wa = weighted_accuracy(y_trues, y_predicts)
        ua = unweighted_accuracy(y_trues, y_predicts)
        logger.info(f"epoch: {epoch}")
        logger.info(f"train    loss: {np.mean(losses):.3f} \t wa: {wa:.3f} \t ua: {ua:.3f} \t")
    return model


def evaluate(model, data_loader):
    model.eval()
    losses, y_trues, y_predicts = [], [], []
    for sample_text, sample_label in tqdm(data_loader):
        sample_sentence = {k: v.squeeze(1).long().to(device) for k, v in sample_text.items()}
        sample_label = sample_label.long().to(device)
        prob = model(sample_sentence)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(prob, sample_label.view(-1))
        losses.append(loss.item())
        y_trues += sample_label.cpu().view(-1).numpy().tolist()
        y_predicts += prob.cpu().argmax(1).numpy().tolist()

    wa = weighted_accuracy(y_trues, y_predicts)
    ua = unweighted_accuracy(y_trues, y_predicts)
    logger.info(f"valid    loss: {np.mean(losses):.3f} \t wa: {wa:.3f} \t ua: {ua:.3f} \t")
    classification_report(y_trues, y_predicts, zero_division=1)
    model.train()
    return wa


def make_data():
    # load dataset
    titles, contents, labels = load_dataset(params.data_path)

    # split dataset
    train_data, test_data = train_test_splite(titles, contents, labels, params.test_size)

    # create data loader
    train_data, test_data = map(BertDataset, [train_data, test_data])
    train_loader = DataLoader(train_data, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params.vaild_batch_size, shuffle=True)

    # create loss funcation
    loss_function = nn.CrossEntropyLoss()
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params.learn_rate, eps=params.adam_epsilon)
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=params.learn_rate, betas=(0.9, 0.99), weight_decay=params.weight_decay)

    return train_loader, test_loader, loss_function, optimizer


def evaluateModel(parameterization):
    global model
    train_loader, test_loader, loss_function, optimizer = make_data()
    logger.info("-------------------------------------------------")
    logger.info(f"begin to train the model: {model.name}")
    model = trains(model=model, newParams=parameterization, train_loader=train_loader,
                   optimizer=optimizer, loss_func=loss_function,
                   isOptimzeHP=params.isOptimzeHP)
    logger.info(f"model({model.name}) is train finish")
    logger.info("-------------------------------------------------")
    return evaluate(model, test_loader)



def optmizeModel():
    best_parameters, values, experiment, net = optimize(
        parameters=[
            {"name": "learn_rate", "type": "range", "bounds": [1e-5, 8e-5], "log_scale": True},
            {"name": "dropout_rate", "type": "range", "bounds": [0.1, 0.5]},
            {"name": "rnn_layers", "type": "range", "bounds": [1, 3]}
        ],
        evaluation_function=evaluateModel,
        objective_name='accuracy',
        total_trials=5
    )



def changeModel(new_model):
    global model
    model = new_model

