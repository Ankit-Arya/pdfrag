Summary of source changes (the installer applies these edits safely):

1. Backend AnswerResponse adds:
   - question_created_at
   - response_created_at
2. The chat endpoint refreshes both saved ChatMessage rows and returns their
   database-created timestamps.
3. The frontend Message view model carries createdAt.
4. Saved chats map ChatMessageOut.created_at to createdAt.
5. Live questions use a temporary client timestamp, replaced by the persisted
   server timestamp when the response arrives.
6. ChatPanel formats timestamps with Intl.DateTimeFormat and renders a <time>
   element beside each message label.
