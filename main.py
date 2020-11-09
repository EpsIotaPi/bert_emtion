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
# optmizeModel()
