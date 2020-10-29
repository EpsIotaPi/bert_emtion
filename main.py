from HyperParams import HyperParams
from models import BertWwmModel, RoBertaModel
from trains import trains, device
from dataset import logger

params = HyperParams()

logger.info("the flowing is HyperParams:")
logger.info(f"epoch: {params.epochs}")
logger.info(f"batch: {params.batch_size}")
logger.info(f"num_layers: {params.rnn_layers}")
logger.info(f"drop_rate: {params.dropout_rate}")
logger.info("HyperParams is over")


# trains(model=BertWwmModel(params).to(device))
# trains(model=RoBertaModel(params).to(device))
# trains(model=BertWwmModel(params, 'lstm', False).to(device))
# trains(model=RoBertaModel(params, 'lstm', False).to(device))
# trains(model=BertWwmModel(params, 'gru', False).to(device))
trains(model=RoBertaModel(params, 'gru', False).to(device))
trains(model=BertWwmModel(params, 'lstm', True).to(device))
trains(model=RoBertaModel(params, 'lstm', True).to(device))
trains(model=BertWwmModel(params, 'gru', True).to(device))
trains(model=RoBertaModel(params, 'gru', True).to(device))
