"""Parameterize the tested click champion's schema without changing its four-task function."""
import argparse
from pathlib import Path


def adapt(source, output):
    text = source.read_text()
    replacements = [
        ('if len(gradients) != 3:', 'if len(gradients) < 1:'),
        ("Expected ordered cart/fav/buy cotangents", 'Expected at least one ordered auxiliary cotangent'),
        ("Expected cart/fav/buy cotangents", 'Expected at least one auxiliary cotangent'),
        ('        total = (active[0] + active[1]) + active[2]',
         '        total = active[0]\n        for contribution in active[1:]:\n            total = total + contribution'),
        ('for omitted in range(3):', 'for omitted in range(len(gradients)):'),
        ('    reduced = (torch.zeros_like(gradients[0]), gradients[1],\n               torch.zeros_like(gradients[2]))',
         '    reduced = tuple(gradient if index == 1 else torch.zeros_like(gradient)\n                    for index, gradient in enumerate(gradients))'),
        ('                click_relative_transfer, orthogonal_assignment):\n        if context.ndim',
         '                click_relative_transfer, orthogonal_assignment, auxiliary_count):\n        if context.ndim'),
        ('return tuple(context.view_as(context) for _ in range(4))',
         'return tuple(context.view_as(context) for _ in range(1 + auxiliary_count))'),
        ('    def backward(ctx, click_gradient, cart_gradient, fav_gradient, buy_gradient):\n        (occupied,) = ctx.saved_tensors\n        total = _reduce_context_transfer(\n            click_gradient, (cart_gradient, fav_gradient, buy_gradient), occupied,',
         '    def backward(ctx, *task_gradients):\n        (occupied,) = ctx.saved_tensors\n        click_gradient, *auxiliary_gradients = task_gradients\n        total = _reduce_context_transfer(\n            click_gradient, tuple(auxiliary_gradients), occupied,'),
        ('        return total, None, None, None, None, None\n\n\nclass _PrimaryPopulationJacobian',
         '        return total, None, None, None, None, None, None\n\n\nclass _PrimaryPopulationJacobian'),
        ("                 task='binary_classification', **kwargs):\n        if num_tasks",
         "                 primary_task='is_click', match_fields=('item_id', 'cate_id', 'brand'),\n                 task='binary_classification', **kwargs):\n        if num_tasks"),
        ("        if set(feature_map.labels) != {'is_click', 'cart', 'fav', 'buy'}:\n            raise ValueError('This study requires the frozen four-task schema')",
         "        labels = list(feature_map.labels)\n        if len(labels) < 2 or len(set(labels)) != len(labels) or primary_task not in labels:\n            raise ValueError('At least two distinct tasks and a declared primary task are required')\n        self.primary_task = primary_task\n        self.auxiliary_labels = [label for label in labels if label != primary_task]"),
        ("        self.auxiliary_indices = [list(feature_map.labels).index(label)\n                                  for label in ('cart', 'fav', 'buy')]",
         '        self.auxiliary_indices = [labels.index(label) for label in self.auxiliary_labels]'),
        ("        self.match_fields = ('item_id', 'cate_id', 'brand')",
         "        self.match_fields = tuple(match_fields)\n        if len(self.match_fields) != 3:\n            raise ValueError('The inherited three metadata match slots must be retained')"),
        ('        for name in self.match_fields:\n            if name not in self.item_fields:',
         '        for name in self.match_fields:\n            if name is None:\n                continue\n            if name not in self.item_fields:'),
        ('        for name in self.match_fields:\n            ids = item[name]',
         '        for name in self.match_fields:\n            if name is None:\n                matches.append(torch.zeros_like(mask, dtype=torch.bool))\n                continue\n            ids = item[name]'),
        ('                self.orthogonal_assignment)\n            product_context_by_label = dict(zip((\'cart\', \'fav\', \'buy\'), collected_contexts[1:]))',
         '                self.orthogonal_assignment, len(self.auxiliary_labels))\n            product_context_by_label = dict(zip(self.auxiliary_labels, collected_contexts[1:]))'),
        ("label == 'is_click'", 'label == self.primary_task'),
        ("label != 'is_click'", 'label != self.primary_task'),
    ]
    for before, after in replacements:
        if before not in text:
            raise ValueError('Unexpected source layout: ' + before[:100])
        text = text.replace(before, after)
    output.write_text('# Fixed interface transfer of g1-r0088-membership_full_first_mean_only.\n'
                      '# Native task order controls a left-associated FP64 auxiliary gradient sum.\n'
                      '# Three exact-ID match slots are retained; absent metadata produces zeros.\n' + text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    adapt(args.source, args.output)
