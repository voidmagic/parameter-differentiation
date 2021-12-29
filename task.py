import logging
from fairseq import utils
from .view import ModelView
from fairseq.trainer import Trainer
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask


logger = logging.getLogger(__name__)


@register_task('parameter_differentiation_task')
class ParameterDifferentiationTask(MultilingualTranslationTask):
    _view: ModelView = None

    @property
    def view(self):
        if self._view is None:
            self._view = ModelView(get_trainer().model)
        return self._view

    def record_gradient(self, model):
        logger.info("Start accumulating gradient")
        criterion = get_trainer().get_criterion()
        model.eval()  # disable dropout
        for lang_pair, dataset in self.dataset(self.args.valid_subset).datasets.items():
            batch_iterator = self.get_batch_iterator(
                dataset=dataset, max_tokens=self.args.max_tokens_valid, seed=self.args.seed).next_epoch_itr()
            model.zero_grad()
            for sample in batch_iterator:
                sample = utils.move_to_cuda(sample)
                loss, _, _ = criterion(model.models[lang_pair], sample)
                loss = loss / len(batch_iterator)
                loss.backward()
            self.view.accum_gradient(lang_pair)
            model.zero_grad()
        model.train()  # enable dropout
        logger.info("End accumulating gradient")

    def begin_valid_epoch(self, epoch, model):
        self.record_gradient(model)
        logger.info("num. model params before: {}".format(sum(p.numel() for p in model.parameters())))
        _ = list(self.view.auto_split())
        logger.info("num. model params after: {}".format(sum(p.numel() for p in model.parameters())))
        self.view.clear_gradient()
        get_trainer()._optimizer = None


def get_trainer() -> Trainer:
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, Trainer):
            return obj
