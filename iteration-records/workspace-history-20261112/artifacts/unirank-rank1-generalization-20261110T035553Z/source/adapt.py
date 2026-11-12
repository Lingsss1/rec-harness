"""Schema-only adaptation of the frozen g3-r0018 top-topic representative."""
import argparse
from pathlib import Path


def adapt(source, output):
    text = Path(source).read_text()
    changes = [
        ("def __init__(self, item_width, width, heads, source, layout):",
         "def __init__(self, item_width, width, heads, source, layout, num_tasks=4, group_count=7):"),
        ("        self.item_width = item_width\n",
         "        self.item_width = item_width\n        self.num_tasks = num_tasks\n        self.group_count = group_count\n"),
        ("torch.empty(1, 5, 4, width)", "torch.empty(1, 5, self.num_tasks, width)"),
        ("(batch, 5, 4, self.item_width)", "(batch, 5, self.num_tasks, self.item_width)"),
        ("(batch, 5, 4, 1)", "(batch, 5, self.num_tasks, 1)"),
        ("math.sqrt(7.0)", "math.sqrt(self.group_count)"),
        ("expand(-1, -1, 4)", "expand(-1, -1, self.num_tasks)"),
        ("batch, 20, self.heads, self.head_width", "batch, 5 * self.num_tasks, self.heads, self.head_width"),
        ("        if list(feature_map.labels) != ['is_click', 'cart', 'fav', 'buy']:\n            raise ValueError('This study requires the frozen four-task order')",
         "        labels = list(feature_map.labels)\n        if len(labels) < 2 or len(set(labels)) != len(labels):\n            raise ValueError('Native tasks must contain at least two distinct labels')"),
        ("        expected = {'item_id', 'cate_id', 'campaign_id', 'customer_id',\n                    'brand', 'price_bucket', 'action'}\n        if set(value_fields) != expected or len(value_fields) != 7:\n            raise ValueError('Expected the frozen six item fields plus action')",
         "        if len(set(value_fields)) != len(value_fields) or len(self.item_fields) < 2:\n            raise ValueError('Pair relations require at least two distinct native item fields')"),
        ("torch.zeros(4, self.key_width, token_dim)", "torch.zeros(self.num_tasks, self.key_width, token_dim)"),
        ("self.key_width, token_dim, num_heads, memory_source, memory_layout)",
         "self.key_width, token_dim, num_heads, memory_source, memory_layout,\n                num_tasks=self.num_tasks, group_count=self.group_count)"),
        ("        if len(task_masks) != 4:\n            raise ValueError('Expected four historical task masks')",
         "        if len(task_masks) != self.num_tasks:\n            raise ValueError('Expected one historical mask per native task')"),
    ]
    for before, after in changes:
        if before not in text:
            raise ValueError('Frozen source layout changed: ' + before[:90])
        text = text.replace(before, after)
    Path(output).write_text(
        '# Schema adaptation of validation rank 1: g3-r0018-pair_occupancy_task_private_linear.\n'
        '# Native field pairs and task cardinalities; five temporal bins and open gradients retained.\n'
        + text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    adapt(args.source, args.output)
