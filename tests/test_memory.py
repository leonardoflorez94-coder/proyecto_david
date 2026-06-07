from app.memory import MemoryStore


def test_recent_messages_keep_order_and_limit(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.init()

    store.add_message("abc", "user", "uno")
    store.add_message("abc", "assistant", "dos")
    store.add_message("abc", "user", "tres")

    messages = store.recent_messages("abc", limit=2)

    assert [message.content for message in messages] == ["dos", "tres"]


def test_rejects_invalid_role(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.init()

    try:
        store.add_message("abc", "system", "no")
    except ValueError as exc:
        assert "role" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
