from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

PATCH_MODELS_LINES = ['def patch_models(source: str) -> str:', '    if \'answer_strategy: str = "direct_lookup"\' in source:', '        return source', '', '    class_start = source.find("class AnswerResponse(BaseModel):")', '    if class_start < 0:', '        raise RuntimeError("AnswerResponse class not found in backend/app/models.py")', '', '    class_end = source.find("\\n\\nclass ", class_start + 1)', '    if class_end < 0:', '        class_end = len(source)', '', '    block = source[class_start:class_end]', '    anchor = \'    answer_policy_version: str = ""\\n\'', '    count = block.count(anchor)', '    if count != 1:', '        raise RuntimeError(', '            f"AnswerResponse synthesis metadata: expected exactly one "', '            f"answer_policy_version field inside AnswerResponse, found {count}"', '        )', '', '    addition = (', "        \'    # Query/answer strategy metadata for the v5.2 inspector and persisted chats.\\n\'", '        \'    answer_strategy: str = "direct_lookup"\\n\'', "        \'    synthesis_dimensions: list[str] = Field(default_factory=list)\\n\'", '        \'    search_scope: str = "focused"\\n\'', "        \'    relevant_documents: list[str] = Field(default_factory=list)\\n\'", "        \'    contributing_documents: list[str] = Field(default_factory=list)\\n\'", "        \'    search_rounds: int = 1\\n\'", '        \'    evidence_coverage_status: str = "unknown"\\n\'', "        \'    conflicts: list[dict[str, Any]] = Field(default_factory=list)\\n\'", '    )', '    block = block.replace(anchor, addition + anchor, 1)', '    return source[:class_start] + block + source[class_end:]']
PATCH_API_LINES = ['def patch_api(source: str) -> str:', '    if \'"answer_strategy": response.answer_strategy\' in source:', '        return source', '', '    # backend/app/api.py intentionally contains search_queries in both persisted', '    # chat-message metadata and audit metadata. v5.2 fields must be inserted into', '    # the assistant message metadata used when chats are reloaded, so scope the', '    # anchor to that block instead of requiring a globally unique line.', '    block_start = source.find("message_metadata={")', '    if block_start < 0:', '        raise RuntimeError("assistant ChatMessage message_metadata block not found")', '', '    block_end = source.find("\\n    db.add(assistant_message)", block_start)', '    if block_end < 0:', '        raise RuntimeError("end of assistant message_metadata block not found")', '', '    block = source[block_start:block_end]', '    anchor = \'            "search_queries": response.search_queries,\\n\'', '    count = block.count(anchor)', '    if count != 1:', '        raise RuntimeError(', '            f"persist synthesis metadata: expected exactly one search_queries "', '            f"field inside assistant message_metadata, found {count}"', '        )', '', '    addition = (', '        \'            "answer_strategy": response.answer_strategy,\\n\'', '        \'            "synthesis_dimensions": response.synthesis_dimensions,\\n\'', '        \'            "search_scope": response.search_scope,\\n\'', '        \'            "relevant_documents": response.relevant_documents,\\n\'', '        \'            "contributing_documents": response.contributing_documents,\\n\'', '        \'            "search_rounds": response.search_rounds,\\n\'', '        \'            "evidence_coverage_status": response.evidence_coverage_status,\\n\'', '        \'            "conflicts": response.conflicts,\\n\'', '    )', '', '    block = block.replace(anchor, anchor + addition, 1)', '    return source[:block_start] + block + source[block_end:]']

def replace_function(source: str, name: str, next_name: str, lines: list[str]) -> str:
    start = source.find(f'def {name}(source: str) -> str:')
    end = source.find(f'\n\ndef {next_name}(source: str) -> str:', start)
    if start < 0 or end < 0:
        raise RuntimeError(f'Could not locate {name}/{next_name} boundaries in v5.2 installer')
    replacement = '\n'.join(lines)
    return source[:start] + replacement + source[end:]

def main() -> int:
    parser = argparse.ArgumentParser(description='Repair v5.2 installer anchors for current pdfrag files.')
    parser.add_argument('--repo', default='.', help='pdfrag repository root')
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    installer = repo / 'apply_ims_ui_v2_synthesis_v52_patch.py'
    if not installer.exists():
        raise SystemExit(f'Missing installer: {installer}')

    source = installer.read_text(encoding='utf-8-sig')
    source = replace_function(source, 'patch_models', 'patch_api', PATCH_MODELS_LINES)
    source = replace_function(source, 'patch_api', 'patch_service', PATCH_API_LINES)
    source = source.replace('frontend/src/style.css', 'frontend/src/styles.css')

    installer.write_text(source, encoding='utf-8', newline='\n')
    py_compile.compile(str(installer), doraise=True)

    check = installer.read_text(encoding='utf-8')
    required = [
        'class_start = source.find("class AnswerResponse(BaseModel):")',
        'block_start = source.find("message_metadata={")',
        'frontend/src/styles.css',
    ]
    missing = [item for item in required if item not in check]
    if missing:
        raise RuntimeError(f'Repair verification failed; missing markers: {missing}')

    print('[fixed] patch_models is scoped to AnswerResponse')
    print('[fixed] patch_api is scoped to assistant message_metadata')
    print('[fixed] frontend style path is frontend/src/styles.css')
    print('[verified] repaired v5.2 installer compiles successfully')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
