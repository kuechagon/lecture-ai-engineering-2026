"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

def print_usage(response) -> None:
    """入力トークン数と出力トークン数を表示する。"""
    u = response.usage_metadata
    print(f"\n[tokens] input={u.prompt_token_count}, output={u.candidates_token_count}")

def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[tokens] input={usage.prompt_token_count}, "
            f"output={usage.candidates_token_count}"
        )

    return response.text

def chat() -> None:
    """会話履歴を保ったまま続けて質問する（exit で終了）。"""
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL)  # 履歴は session が保持する
    print(f"[model] {MODEL}  (exit で終了)")
    while True:
        try:
            q = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in {"exit", "quit"}:
            break
        if not q:
            continue
        try:
            response = session.send_message(q)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            continue
        print(response.text)
        print_usage(response)

def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()
    if args.chat:
        chat()
        return

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

