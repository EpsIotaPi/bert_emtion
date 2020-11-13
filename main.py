from HyperParams import HyperParams
from models import BertWwmModel, RoBertaModel
from trains import optmizeModel, changeModel, evaluateModel, device

from dataset import logger

params = HyperParams()

logger.info("the flowing is HyperParams:")
logger.info(f"epoch: {params.epochs}")
logger.info(f"batch: {params.batch_size}")
logger.info(f"rnn_layers: {params.rnn_layers}")
logger.info(f"drop_rate: {params.dropout_rate}")
logger.info("HyperParams is over")


changeModel(new_model=BertWwmModel(params).to(device))
evaluateModel(params)
changeModel(new_model=RoBertaModel(params).to(device))
evaluateModel(params)
changeModel(new_model=BertWwmModel(params, model='lstm', bidirectional=False).to(device))
evaluateModel(params)
# changeModel(new_model=RoBertaModel(params, model='lstm', bidirectional=False).to(device))
# changeModel(new_model=BertWwmModel(params, model='lstm', bidirectional=True).to(device))
# changeModel(new_model=RoBertaModel(params, model='lstm', bidirectional=True).to(device))
changeModel(new_model=BertWwmModel(params, model='gru', bidirectional=False).to(device))
evaluateModel(params)
# changeModel(new_model=RoBertaModel(params, model='gru', bidirectional=False).to(device))
changeModel(new_model=BertWwmModel(params, model='gru', bidirectional=True).to(device))
evaluateModel(params)
changeModel(new_model=RoBertaModel(params, model='gru', bidirectional=True).to(device))
evaluateModel(params)
