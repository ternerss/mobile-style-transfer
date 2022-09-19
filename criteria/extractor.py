from functools import partial

from torch import nn


class FeatureExtractor(nn.Module):
    def _model_to_list(self, model):
        modules = []

        if list(model.children()) == []:
            modules.append(model)

        for ch in model.children():
            modules.extend(self._model_to_list(ch))

        return modules

    def _prepare_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _add_hooks(self, layer_idx):
        for i, idx in enumerate(layer_idx):
            fun = partial(self._hook_fn, idx=i)
            hook = self.__modules[idx].register_forward_hook(fun)
            self.__hooks.append(hook)

    def __init__(self, model, layer_idx):
        super().__init__()

        self.__hooks = []
        self.__list = [0] * len(layer_idx)

        self.model = self._prepare_model(model)
        self.layer_idx = layer_idx
        self.__modules = self._model_to_list(self.model)
        self._add_hooks(layer_idx)

    def _hook_fn(self, module, input, output, idx):
        self.__list[idx] = output

    def forward(self, x):
        self.model(x)

        return self.__list.copy()

    def remove_hooks(self):
        [x.remove() for x in self.__hooks]
