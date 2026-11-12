"""Native-field/task interface for the frozen g3-r0006 relation/moment-token model."""
import argparse
from pathlib import Path


def adapt(source, output):
    text = source.read_text()
    changes = [
        ('                 num_tasks=None,', "                 primary_task='is_click', num_tasks=None,"),
        ("        if list(feature_map.labels) != ['is_click', 'cart', 'fav', 'buy']:\n            raise ValueError('This study requires the frozen four-task order')",
         "        labels = list(feature_map.labels)\n        if len(labels) < 2 or len(set(labels)) != len(labels) or primary_task not in labels:\n            raise ValueError('Native tasks must be distinct and include the primary task')\n        self.primary_task = primary_task"),
        ("label == 'is_click'", 'label == self.primary_task'),
        ("        expected = {'item_id', 'cate_id', 'campaign_id', 'customer_id',\n                    'brand', 'price_bucket', 'action'}\n        if set(value_fields) != expected or len(value_fields) != 7:\n            raise ValueError('Expected the frozen six item fields plus action')",
         "        if len(set(value_fields)) != len(value_fields):\n            raise ValueError('Native item/action feature groups must be distinct')"),
        ("        if len(task_masks) != 4:\n            raise ValueError('Expected four historical task masks')",
         "        if len(task_masks) != self.num_tasks:\n            raise ValueError('Expected one historical mask per native task')"),
    ]
    for before, after in changes:
        if before not in text:
            raise ValueError('Source layout changed: ' + before[:80])
        text = text.replace(before, after)
    output.write_text('# Fixed schema adaptation of the validation-selected g3-r0006-linear_private.\n'
                      '# Five temporal slots stay fixed; relation groups follow the native item fields.\n'
                      '# Native task heads and primary scorer-input gradient role are parameterized.\n' + text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    adapt(args.source, args.output)
