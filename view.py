import copy
import torch
import logging
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering


logger = logging.getLogger(__name__)


def name2module(module, name):
    def _generator(_module):
        for part in name.split('.'):
            _module = getattr(_module, part)
            yield _module
    return [module] + list(_generator(module))


def get_module_names(model: nn.Module):
    all_param_names = [name for name in dict(model.named_parameters()).keys() if 'layers' in name]
    all_param_names = [param.rstrip('.weight') for param in all_param_names if 'weight' in param]
    return all_param_names


def extract_gradient_from_module(module):
    grads = [p.grad.view(-1) for _, p in sorted(module.named_parameters(), key=lambda pair: pair[0])]
    return torch.cat(grads).data.cpu()


class ModelView:
    def __init__(self, model):
        self.model = model
        self.container = {name: model.keys for name in get_module_names(model)}
        self.gradients = {lang_pair: {} for lang_pair in model.keys}

    def clear_gradient(self):
        self.gradients = {lang_pair: {} for lang_pair in self.model.keys}

    def accum_gradient(self, lang_pair):
        cur_model = self.model.models[lang_pair]
        for name in get_module_names(cur_model):
            module_tree = name2module(cur_model, name)
            grad = extract_gradient_from_module(module_tree[-1])
            self.gradients[lang_pair][name] = grad + self.gradients[lang_pair].get(name, 0)

    def auto_split(self):
        logger.info('Detect split parameters by grad')
        # 根据梯度，计算每个模块的散度
        # calculate distance (or divergence) of each module.
        divergences = {}
        for name, lang_pairs in self.container.items():
            # name是模块的全名，lang_pairs是这个模块被多少语言对共享。
            # name is the full name of a module. lang_pairs is all languages that share this module.
            # 如果把name中的lang_pair变为lang_pairs中的lang_pair，那实际指向的是同一个模块
            # if we change the `lang_pair` in `name` to `lang_pair` in `lang_pairs`, they actual point to the same module.
            short_name = ".".join(name.split('.')[2:])    # name: 'models.en-de.encoder.layers.0'  short_name: 'encoder.layers.0'
            module_gradients = {lang_pair: self.gradients[lang_pair][short_name] for lang_pair in lang_pairs}
            divergences[name] = calculate_div(module_gradients)

        # 按距离排序，从大到小，-1表示距离最小
        # sorted by distance from large to small. -1 means the smallest distance.
        sorted_divergences = [d for d in sorted(divergences.items(), key=lambda item: -item[1][1]) if d[1][1] > 0]
        for best_name, (best_lang_pairs, best_score) in sorted_divergences[:2]:
            logger.info('Split shared parameters: {}'.format(best_name))
            logger.info('This parameter is shared by {}'.format(','.join(best_lang_pairs[0] + best_lang_pairs[1])))
            logger.info('After split: {}   {}'.format(','.join(best_lang_pairs[0]), ','.join(best_lang_pairs[1])))
            logger.info('Cosine distance is {}'.format(best_score))
            yield self.split_module(best_name, best_lang_pairs)

    def split_module(self, module_to_split, split_lang_pairs):
        # 1. 修改container的内容. Change the content in the container.
        # 旧的参数以lang_pairs[0][i]为base. Old parameters take lang_pairs[0][i] as base.
        if module_to_split.split(".")[1] in split_lang_pairs[1]:
            split_lang_pairs[0], split_lang_pairs[1] = split_lang_pairs[1], split_lang_pairs[0]

        self.container[module_to_split] = split_lang_pairs[0]
        # 新的参数以lang_pairs[1][0]为base. New parameters take lang_pairs[1][0] as base.
        new_name = ".".join([module_to_split.split(".")[0], split_lang_pairs[1][0]] + module_to_split.split(".")[2:])
        self.container[new_name] = split_lang_pairs[1]

        # 2. 新建参数. Create new parameters
        module_tree = name2module(self.model, module_to_split)
        new_module = copy.deepcopy(module_tree[-1]).cuda()

        # 3. 给第二个聚类中的语言，赋予该模块. assign the new parameter to languages in the second cluster.
        # 第一个聚类还是原来的参数。 the languages in the first cluster use the origin parameters.
        for lang_pair in split_lang_pairs[1]:
            module_name = ".".join([module_to_split.split(".")[0], lang_pair] + module_to_split.split(".")[2:])
            module_tree = name2module(self.model, module_name)
            setattr(module_tree[-2], module_name.split(".")[-1], new_module)
        return new_name, module_to_split


def calculate_div(module_gradients):
    """
    对于一个特定模块，由L种语言共享，module_gradients就是在这个模块上，每个语言对应的梯度。
    本函数对其进行聚类，最后分为两个类别，并返回类间距离。
    For a specific module that shared by L languages, `module_gradients` means the gradient of each language on this module.
    This function clusters the languages into two clusters, return the two clusters and their inter-cluster distance.
    :param module_gradients: dict of {lang_pair: gradient}
    :return: [[cluster_1], [cluster_2]], distance
    """
    if len(module_gradients) < 2:
        return [], -1

    cluster = AgglomerativeClustering(linkage='average', affinity='cosine', n_clusters=2, compute_distances=True)
    lang_pairs, gradients = zip(*module_gradients.items())
    labels = cluster.fit_predict(torch.stack(gradients).numpy() + 1e-5)
    cluster_0 = [lang_pair for lang_pair, label in zip(lang_pairs, labels) if label == 0]
    cluster_1 = [lang_pair for lang_pair, label in zip(lang_pairs, labels) if label == 1]
    return [cluster_0, cluster_1], cluster.distances_[-1]
