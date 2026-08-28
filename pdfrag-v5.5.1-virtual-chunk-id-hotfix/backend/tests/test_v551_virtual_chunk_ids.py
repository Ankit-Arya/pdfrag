from app.rag.v5.assistant_retrieval import (
    _physical_uuid_chunk_ids,
    _section_keys_for_chunks,
)


PHYSICAL_CHUNK = "20b28c16-2cc1-51e2-9564-e834f818d482"
OTHER_CHUNK = "bfd959c1-e761-5f03-a000-da76286224bc"
DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return list(self._rows)


class _FakeDB:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.calls = []

    def execute(self, _statement, params):
        self.calls.append(dict(params))
        return _FakeResult(self.rows)


def test_physical_uuid_filter_ignores_current_and_future_virtual_ids():
    values = [
        f"v55-procedure:{PHYSICAL_CHUNK}:table:1",
        f"v54-headerless:{PHYSICAL_CHUNK}",
        "virtual:future-evidence:abc",
        PHYSICAL_CHUNK,
        PHYSICAL_CHUNK.upper(),
        OTHER_CHUNK,
        "",
    ]
    assert _physical_uuid_chunk_ids(values) == [PHYSICAL_CHUNK, OTHER_CHUNK]


def test_section_lookup_sends_only_physical_uuids_to_postgres():
    db = _FakeDB([
        {"id": PHYSICAL_CHUNK, "document_id": DOCUMENT_ID, "parent_key": "sec:brakes"}
    ])
    keys = _section_keys_for_chunks(
        db,
        [
            f"v55-procedure:{PHYSICAL_CHUNK}:table:1",
            PHYSICAL_CHUNK,
            f"v54-headerless:{OTHER_CHUNK}",
        ],
    )
    assert db.calls == [{"chunk_ids": [PHYSICAL_CHUNK]}]
    assert keys == [(DOCUMENT_ID, "sec:brakes")]


def test_all_virtual_ids_skip_database_lookup_entirely():
    db = _FakeDB()
    keys = _section_keys_for_chunks(
        db,
        [
            f"v55-procedure:{PHYSICAL_CHUNK}:table:1",
            f"v54-headerless:{OTHER_CHUNK}",
            "future-aggregate:not-a-uuid",
        ],
    )
    assert keys == []
    assert db.calls == []
