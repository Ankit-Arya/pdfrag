#!/usr/bin/env python3
"""Apply the pdfrag per-message chat timestamp patch.

The patch is intentionally applied as targeted source edits so it preserves the
previous audit-log patch and any unrelated local changes.
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
from pathlib import Path

BACKUP_SUFFIX = ".before-chat-timestamps.bak"


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def patch_backend_models(text: str) -> str:
    old = (
        "    request_id: str | None = None\n"
        "    chat_session_id: uuid.UUID | None = None\n"
    )
    new = (
        "    request_id: str | None = None\n"
        "    chat_session_id: uuid.UUID | None = None\n"
        "    question_created_at: datetime | None = None\n"
        "    response_created_at: datetime | None = None\n"
    )
    return replace_once(text, old, new, "backend AnswerResponse timestamps")


def patch_backend_api(text: str) -> str:
    old_user_message = '''    db.add(
        ChatMessage(
            chat_session_id=chat_session.id,
            role="user",
            content=payload.question,
        )
    )
    chat_session.updated_at = datetime.now(UTC)
    db.commit()
'''
    new_user_message = '''    user_message = ChatMessage(
        chat_session_id=chat_session.id,
        role="user",
        content=payload.question,
    )
    db.add(user_message)
    chat_session.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(user_message)
'''
    text = replace_once(
        text,
        old_user_message,
        new_user_message,
        "persisted user-message timestamp",
    )

    old_response_ids = '''    response.chat_session_id = chat_session.id
    response.request_id = getattr(request.state, "request_id", None)
'''
    new_response_ids = '''    response.chat_session_id = chat_session.id
    response.request_id = getattr(request.state, "request_id", None)
    response.question_created_at = user_message.created_at
'''
    text = replace_once(
        text,
        old_response_ids,
        new_response_ids,
        "question timestamp in live response",
    )

    if "    assistant_message = ChatMessage(\n" not in text:
        response_offset = text.find(new_response_ids)
        if response_offset < 0:
            raise PatchError("assistant message: could not locate chat response block")
        tail = text[response_offset + len(new_response_ids) :]
        match = re.search(
            r"    db\.add\(\n        ChatMessage\(\n(?P<body>.*?)        \)\n    \)\n",
            tail,
            flags=re.DOTALL,
        )
        if not match:
            raise PatchError("assistant message: expected database insert block not found")
        body = match.group("body")
        dedented_body = "".join(
            line[4:] if line.startswith("    ") else line
            for line in body.splitlines(keepends=True)
        )
        replacement = (
            "    assistant_message = ChatMessage(\n"
            f"{dedented_body}"
            "    )\n"
            "    db.add(assistant_message)\n"
        )
        absolute_start = response_offset + len(new_response_ids) + match.start()
        absolute_end = response_offset + len(new_response_ids) + match.end()
        text = text[:absolute_start] + replacement + text[absolute_end:]

    old_return = '''    db.commit()
    return response
'''
    new_return = '''    db.commit()
    db.refresh(assistant_message)
    response.response_created_at = assistant_message.created_at
    return response
'''
    if new_return not in text:
        chat_offset = text.find('@router.post("/chat", response_model=AnswerResponse)')
        if chat_offset < 0:
            raise PatchError("chat endpoint marker not found")
        return_offset = text.find(old_return, chat_offset)
        if return_offset < 0:
            raise PatchError("response timestamp: final chat commit/return block not found")
        text = text[:return_offset] + new_return + text[return_offset + len(old_return) :]

    return text


def patch_frontend_api(text: str) -> str:
    old = "  chat_session_id?: string | null\n"
    new = (
        "  chat_session_id?: string | null\n"
        "  question_created_at?: string | null\n"
        "  response_created_at?: string | null\n"
    )
    return replace_once(text, old, new, "frontend AnswerResponse timestamps")


def patch_frontend_app(text: str) -> str:
    old_interface = '''  role: 'user' | 'assistant'
  text: string
  response?: AnswerResponse
'''
    new_interface = '''  role: 'user' | 'assistant'
  text: string
  createdAt: string
  response?: AnswerResponse
'''
    text = replace_once(text, old_interface, new_interface, "frontend Message timestamp")

    old_history = '''        role: message.role as 'user' | 'assistant',
        text: message.content,
        response:
'''
    new_history = '''        role: message.role as 'user' | 'assistant',
        text: message.content,
        createdAt: message.created_at,
        response:
'''
    text = replace_once(text, old_history, new_history, "saved chat timestamps")

    old_ask = '''async function ask(value: string): Promise<void> {
  messages.value.push({ id: id(), role: 'user', text: value })
  question.value = ''
'''
    new_ask = '''async function ask(value: string): Promise<void> {
  const questionMessageId = id()
  messages.value.push({
    id: questionMessageId,
    role: 'user',
    text: value,
    createdAt: new Date().toISOString(),
  })
  question.value = ''
'''
    text = replace_once(text, old_ask, new_ask, "live question timestamp")

    old_response = '''    const response = await askQuestion(value, activeChatId.value, controller.signal)
    activeChatId.value = response.chat_session_id ?? activeChatId.value
    messages.value.push({
'''
    new_response = '''    const response = await askQuestion(value, activeChatId.value, controller.signal)
    activeChatId.value = response.chat_session_id ?? activeChatId.value
    if (response.question_created_at) {
      const savedQuestion = messages.value.find(
        (message) => message.id === questionMessageId,
      )
      if (savedQuestion) savedQuestion.createdAt = response.question_created_at
    }
    messages.value.push({
'''
    text = replace_once(text, old_response, new_response, "server question timestamp")

    old_assistant = '''      role: 'assistant',
      text: response.answer,
      response,
'''
    new_assistant = '''      role: 'assistant',
      text: response.answer,
      createdAt: response.response_created_at ?? new Date().toISOString(),
      response,
'''
    text = replace_once(text, old_assistant, new_assistant, "live response timestamp")
    return text


def patch_chat_panel(text: str) -> str:
    old_interface = '''  role: 'user' | 'assistant'
  text: string
  response?: AnswerResponse
'''
    new_interface = '''  role: 'user' | 'assistant'
  text: string
  createdAt: string
  response?: AnswerResponse
'''
    text = replace_once(text, old_interface, new_interface, "ChatPanel Message timestamp")

    old_state = "const libraryError = ref('')\n"
    new_state = '''const libraryError = ref('')
const messageDateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})

function formatMessageTime(value: string): string {
  const timestamp = new Date(value)
  if (Number.isNaN(timestamp.getTime())) return ''
  return messageDateTimeFormatter.format(timestamp)
}
'''
    text = replace_once(text, old_state, new_state, "message time formatter")

    old_label = '''          <span class="message-label">
            {{ message.role === 'user' ? 'You' : 'DMRC Q&A' }}
          </span>
'''
    new_label = '''          <div class="message-heading">
            <span class="message-label">
              {{ message.role === 'user' ? 'You' : 'DMRC Q&A' }}
            </span>
            <time
              v-if="formatMessageTime(message.createdAt)"
              class="message-timestamp"
              :datetime="message.createdAt"
              :title="formatMessageTime(message.createdAt)"
            >
              {{ formatMessageTime(message.createdAt) }}
            </time>
          </div>
'''
    text = replace_once(text, old_label, new_label, "message timestamp display")

    old_style = "<style scoped>\n"
    new_style = '''<style scoped>
.message-heading {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
}

.message-timestamp {
  color: #89958f;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.4;
  white-space: nowrap;
}

.message.user .message-heading {
  justify-content: flex-end;
}

'''
    text = replace_once(text, old_style, new_style, "message timestamp styles")
    return text


PATCHES = {
    Path("backend/app/models.py"): patch_backend_models,
    Path("backend/app/api.py"): patch_backend_api,
    Path("frontend/src/services/api.ts"): patch_frontend_api,
    Path("frontend/src/App.vue"): patch_frontend_app,
    Path("frontend/src/components/ChatPanel.vue"): patch_chat_panel,
}


def validate(root: Path) -> None:
    ast.parse((root / "backend/app/models.py").read_text(encoding="utf-8"))
    ast.parse((root / "backend/app/api.py").read_text(encoding="utf-8"))

    app_text = (root / "frontend/src/App.vue").read_text(encoding="utf-8")
    panel_text = (root / "frontend/src/components/ChatPanel.vue").read_text(encoding="utf-8")
    api_text = (root / "frontend/src/services/api.ts").read_text(encoding="utf-8")
    required = {
        "App.vue live question timestamp": "savedQuestion.createdAt = response.question_created_at",
        "App.vue live answer timestamp": "response.response_created_at ?? new Date().toISOString()",
        "ChatPanel timestamp element": 'class="message-timestamp"',
        "frontend API response timestamp": "response_created_at?: string | null",
    }
    combined = app_text + "\n" + panel_text + "\n" + api_text
    missing = [label for label, marker in required.items() if marker not in combined]
    if missing:
        raise PatchError("validation failed: " + ", ".join(missing))


def apply(root: Path, dry_run: bool) -> list[str]:
    planned: list[tuple[Path, Path, str]] = []
    for relative_path, patcher in PATCHES.items():
        path = root / relative_path
        if not path.is_file():
            raise PatchError(f"required file not found: {path}")
        original = path.read_text(encoding="utf-8")
        updated = patcher(original)
        if updated != original:
            planned.append((relative_path, path, updated))

    changed = [str(relative_path) for relative_path, _, _ in planned]
    if dry_run or not planned:
        if not dry_run:
            validate(root)
        return changed

    written: list[tuple[Path, Path]] = []
    try:
        for _, path, updated in planned:
            backup = path.with_name(path.name + BACKUP_SUFFIX)
            if not backup.exists():
                shutil.copy2(path, backup)
            # pathlib.Path.write_text gained the newline parameter in Python 3.10.
            # Use Path.open so the installer also works with Python 3.9 on Windows.
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(updated)
            written.append((path, backup))
        validate(root)
    except Exception:
        for path, backup in reversed(written):
            if backup.exists():
                shutil.copy2(backup, path)
        raise
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path, help="Path to the pdfrag repository root")
    parser.add_argument("--dry-run", action="store_true", help="Check applicability without writing")
    args = parser.parse_args()

    root = args.repository.expanduser().resolve()
    if not (root / "backend/app").is_dir() or not (root / "frontend/src").is_dir():
        raise PatchError(f"not a pdfrag repository root: {root}")

    changed = apply(root, args.dry_run)
    action = "Would update" if args.dry_run else "Updated"
    if changed:
        print(f"{action} {len(changed)} file(s):")
        for item in changed:
            print(f"  - {item}")
    else:
        print("Patch is already applied; no files changed.")
    if not args.dry_run:
        print("Source validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
