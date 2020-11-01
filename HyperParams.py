class HyperParams:
    # maxlen sequence length
    max_sequence_length = 512

    # 测试集所占百分比
    test_size = 0.2
    batch_size = 5
    vaild_batch_size = 5
    epochs = 5
    # 类别数
    n_classes = 3

    # 维度，与bert保持一致
    embed_dim = 768
    hidden_size = 768
    rnn_layers = 3
    '''-------------------调参-------------------'''
    adam_epsilon = 1e-6
    patience = 10
    decay_rate = 0.90
    grad_clip = 3
    learn_rate = 1e-5
    min_learn_rate = 8e-6
    dropout_rate = 0.8
    '''-------------------调参-------------------'''

    bert_path = "./models/chinese_wwm_pytorch"
    bert_config_path = "./models/chinese_wwm_pytorch/bert_config.json"

    roberta_path = "./models/chinese_roberta_wwm_ext_pytorch"
    roberta_config_path = "./models/chinese_roberta_wwm_ext_pytorch/bert_config.json"

    data_path = "./data/replacement/train.csv"
    log_file = "./log/log.txt"




