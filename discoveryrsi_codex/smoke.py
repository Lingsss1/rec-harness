"""Check the real Codex backend separately from the 44-attempt experiment."""
from discoveryrsi_codex.run import CodexController, WORKSPACE, config, object_schema, TEXT


def main():
    root = WORKSPACE / 'artifacts/discoveryrsi-native-smoke'
    controller = CodexController(config(root))
    value = controller.query('backend-v1',
        'Transport smoke only: return status="ok" and purpose="DiscoveryRSI Codex backend". Do not use tools.',
        object_schema({'status': TEXT, 'purpose': TEXT}))
    assert value['status'] == 'ok', value
    print(value)


if __name__ == '__main__':
    main()
