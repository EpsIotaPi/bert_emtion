import torch
from torch import nn
from tqdm import tqdm
from ax import optimize

from dataset import *
from sklearn.metrics import classification_report


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


def trains(model, params, train_loader, optimizer, loss_func):
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
    classification_report(y_trues, y_predicts)
    model.train()
    return wa


def train_evaluate(model):
    logger.info("-------------------------------------------------")

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

    logger.info(f"begin to train the model: {model.name}")

    model = trains(model, params, train_loader, optimizer, loss_function)

    logger.info(f"model({model.name}) is train finish")
    logger.info("-------------------------------------------------")

    return evaluate(model, test_loader)

def eva_import torch
from torch import nn
from tqdm import tqdm
from ax import optimize

from dataset import *
from sklearn.metrics import classification_report


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


def trains(model, params, train_loader, optimizer, loss_func):
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
    classification_report(y_trues, y_predicts)
    model.train()
    return wa


def train_evaluate(model):
    logger.info("-------------------------------------------------")

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

    logger.info(f"begin to train the model: {model.name}")

    model = trains(model, params, train_loader, optimizer, loss_function)
    # evaluate(model, test_loader)

    best_parameters, values, experiment, net = optimize(
        parameters=[
            {"name": "learn_rate", "type": "range", "bounds": [8e-6, 1e-3], "log_scale": True},
            {"name": "dropout_rate", "type": "range", "bounds": [0.0, 0.9]},
            {"name": "rnn_layers", "type": "range", "bounds": [1, 10]}
        ],
        evaluation_function=evaluate(model, test_loader),
        objective_name='accuracy',
        total_trials=15
    )

    print(best_parameters)

    logger.info(f"model({model.name}) is train finish")
    logger.info("-------------------------------------------------")

    return









def train_evaluate(parameterization):






# model = BertWwmModel(params).to(device)









func(params):





best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "learn_rate", "type": "range", "bounds": [8e-6, 1e-3], "log_scale": True},
        {"name": "dropout_rate", "type": "range", "bounds": [0.0, 0.9]},
        {"name": "rnn_layers", "type": "range", "bounds": [1, 10]}
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
    total_trials=15
)







def train_evaluate(parameterization):






# model = BertWwmModel(params).to(device)









