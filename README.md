# hallguard

LLM の出力をそのまま信用しない前提で、複数段のチェックを通った結果だけを
下流に流す Python フレームワーク。Pydantic スキーマで型を縛り、confidence と
出典 URL を検査し、別の LLM に矛盾判定させ、通らないものは `max_retries` まで
再生成。超えたら `ErrorOutput` を値として返す（例外は投げない）。

LangChain / CrewAI とは方向性が違って、機能を増やすより「怪しい出力を下流に
流さない」ほうに振ってある。RAG パイプラインの最終ゲート、医療・法務 QA、
評価データセットの自動フィルタなど、間違いが具体的に困る用途を想定。

PyPI distribution 名は `hallguard`、Python の import 名は `hallucination_guard`:

```bash
pip install hallguard[all]
```

---

## アーキテクチャ

```
Input
  │
  ▼
StructuredNode         ← DomainConfig.output_schema() で型強制 / temperature=0
  │
  ▼
FactCheckGate          ← DomainConfig.confidence_threshold / is_valid_source
  ├─ FAIL ──▶ RetryNode ──▶ retry_count >= max_retries → ErrorOutput
  │               └──────────────────────────────────▶ StructuredNode（再試行）
  └─ PASS
        ▼
     CriticNode         ← DomainConfig.critic_prompt()
        ├─ FAIL ──▶ RetryNode（同上）
        └─ PASS ──▶ FinalOutput
```

---

## ディレクトリ構成

```
hallguard/
├── hallucination_guard/
│   ├── __init__.py
│   ├── state.py            ← GraphState / FailReason（イミュータブル）
│   ├── exceptions.py       ← GraphError 階層
│   ├── schemas.py          ← Claim / GroundedOutput / CriticVerdict
│   ├── graph.py            ← Graph（LangGraph 組み立て）
│   ├── llm/
│   │   ├── protocols.py    ← StructuredLLM / JudgeLLM プロトコル
│   │   └── openai_adapter.py ← OpenAI Structured Outputs アダプタ
│   ├── nodes/
│   │   ├── structured_node.py  ← schema 強制 + retry directive 注入
│   │   ├── factcheck_gate.py   ← confidence / source 検証
│   │   ├── critic_node.py      ← 独立判定 + final_output 確定
│   │   ├── retry_node.py       ← retry_count++ / 信号リセット
│   │   └── error_output.py     ← max_retries 超過時の終端
│   ├── domain/
│   │   ├── base.py         ← DomainConfig 抽象クラス
│   │   ├── general.py      ← GeneralDomain（許容的なデフォルト）
│   │   └── medical.py      ← MedicalDomain（厳格なデモ）
│   └── retry/
│       ├── directive.py    ← RetryDirective（注入型を frozen で制限）
│       └── hint_builder.py ← RetryHintBuilder（プロンプト汚染防止）
├── tests/
│   ├── test_state.py
│   ├── test_hint_builder.py
│   ├── test_general_domain.py
│   ├── test_medical_domain.py
│   ├── test_protocols.py
│   ├── test_structured_node.py
│   ├── test_factcheck_gate.py
│   ├── test_critic_node.py
│   ├── test_retry_node.py
│   ├── test_error_output.py
│   └── test_graph_integration.py
├── examples/
│   └── research_agent.py   ← Graph をモック LLM で回すデモ
├── benchmarks/
│   ├── hallucination_rate.py ← 成功率 / 再試行数 / 失敗種別を集計する CLI
│   └── datasets/
│       ├── synthetic_qa.json ← デフォルトのベンチマーク用 QA セット
│       └── medical_qa.json   ← MedicalDomain 用 QA セット
├── CHANGELOG.md
├── Makefile                ← build / release / quality-gate ターゲット
├── pyproject.toml
└── README.md
```

---

## 必要環境

- Python **3.11 以上**（開発・動作確認は 3.13 で実施）
- macOS / Linux を想定（Windows は未検証）

---

## セットアップ

コアの依存は `pydantic` のみ。LangGraph / OpenAI は extras で必要な分だけ
追加します。

### A. State / Retry 層だけ使う（最軽量）

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
```

`Graph` は不要だが `GraphState` / `RetryHintBuilder` 等は使う、という構成。

### B. `Graph` を回す（LangGraph 込み）

```bash
.venv/bin/pip install -e ".[graph]"
```

### C. OpenAI アダプタ込み

```bash
.venv/bin/pip install -e ".[openai]"
# graph と両方欲しければ:
.venv/bin/pip install -e ".[all]"
```

### D. 開発用フルセット（テスト・mypy 含む）

```bash
.venv/bin/pip install -e ".[dev]"
```

### 仮想環境の有効化（任意）

```bash
source .venv/bin/activate
# 以降は `pytest` や `mypy` をそのまま打てる
deactivate    # 抜けるとき
```

---

## 動作確認

### テスト

```bash
.venv/bin/python -m pytest tests/ -v
```

193 件 passing。内訳:

- **ノード単体** (`test_state.py` / `test_structured_node.py` /
  `test_factcheck_gate.py` / `test_critic_node.py` / `test_retry_node.py` /
  `test_error_output.py`、計 51 件)
- **統合** (`test_graph_integration.py`、40 件) — `Graph` をモック LLM で
  回し、成功経路 / リトライ / `max_retries` 超過 / `max_retries=0` /
  checkpointer 永続化 / `stream` / `astream` / `arun` / async-native /
  sync↔async フォールバック / `asyncio.wait_for` / 明示キャンセル /
  `asyncio.Semaphore(32)` 下の fan-out で state 分離と `fail_history`
  非共有を検証
- **ドメイン** (`test_general_domain.py` 19 件 / `test_medical_domain.py`
  22 件) — `locale="ja"` 込み。Medical は `_ALLOWED_HOSTS` ↔ retry
  テンプレ整合性も
- **その他** (`test_protocols.py` 6 / `test_openai_adapter.py` 9 /
  `test_hint_builder.py` 12 / `test_serde.py` 11 /
  `test_benchmark_smoke.py` 23、計 61 件) — `@runtime_checkable` の
  sync/async 判定、OpenAI Structured Outputs の in-memory fake、
  `INSTRUCTION_MAPS["en"|"ja"]` parametrize、msgpack allow-list、
  ベンチ CLI 引数群（`--async --concurrency N` の並列度上限・直列退行
  検出を含む）

### 型チェック

```bash
.venv/bin/mypy hallucination_guard/
```

`Success: no issues found in 24 source files` が出れば OK。

### デモ実行

```bash
.venv/bin/python -m examples.research_agent
```

モック LLM を使って 5 つのシナリオを順に流します:
1. 1 回目は低 confidence で `FactCheckGate` が差し戻し、リトライで成功
2. 常に低 confidence を返す LLM で `max_retries` を使い切り、`ErrorOutput` 経路
3. `Graph.stream()` でノード単位の進捗イベントを観測（同じシナリオ #1）
4. `Graph.astream()` を `asyncio.run()` 経由で消費（#3 と同じイベント列を非同期 API で）
5. `AsyncStructuredLLM` / `AsyncJudgeLLM` を直接実装したクライアントを
   `Graph.arun()` で駆動（ブリッジ不要のネイティブ async 経路）

### ベンチマーク

```bash
.venv/bin/python -m benchmarks.hallucination_rate                       # 集計のみ
.venv/bin/python -m benchmarks.hallucination_rate --verbose             # 各クエリの結果も
.venv/bin/python -m benchmarks.hallucination_rate --dataset my_qa.json  # 任意 JSON で
.venv/bin/python -m benchmarks.hallucination_rate \
    --domain medical \
    --dataset benchmarks/datasets/medical_qa.json                       # MedicalDomain で
.venv/bin/python -m benchmarks.hallucination_rate --async --concurrency 8  # 非同期並列で
```

出力は `is_success` 率、平均 `retry_count`、失敗種別の内訳
(`low_confidence` / `no_source` / `critic_rejected`)。決定論的な疑似 LLM
を使うので、同じデータセット + 同じドメインなら再現可能。デフォルトは
`GeneralDomain` + `benchmarks/datasets/synthetic_qa.json`。

- `--dataset` で JSON を切り替え
- `--domain {general,medical}` でドメインを切り替え
- データセット JSON の `suggested_domain` と `--domain` が食い違うと
  stderr に `WARNING:`。`--strict-domain` を付けると `ERROR:` 扱いで
  非ゼロ終了
- `--real --model <openai-model-id>` で実 LLM 経路（`OPENAI_API_KEY`
  必須）
- `--async --concurrency N` で並列実行（デフォルト 4、`--async` 無しでは
  無視）。sync 版と集計値が一致することはテストで担保

---

## 使い方

### エンドツーエンドで回す（モック LLM）

```python
from pydantic import BaseModel

from hallucination_guard.domain.general import GeneralDomain
from hallucination_guard.graph import Graph
from hallucination_guard.schemas import Claim, CriticVerdict, GroundedOutput


class MyStructuredLLM:
    def generate(self, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        return GroundedOutput(
            claims=[
                Claim(
                    text="The capital of France is Paris",
                    confidence=0.98,
                    sources=["https://en.wikipedia.org/wiki/Paris"],
                )
            ]
        )


class MyJudgeLLM:
    def judge(self, *, system: str, content: str) -> CriticVerdict:
        return CriticVerdict(verdict="PASS")


result = Graph(
    domain=GeneralDomain(),
    structured_llm=MyStructuredLLM(),
    judge_llm=MyJudgeLLM(),
).run("What is the capital of France?")

print(result.is_success)     # True
print(result.final_output)   # JSON 文字列
print(result.retry_count)    # 0
```

実 LLM を繋ぐ場合は、`StructuredLLM` / `JudgeLLM` プロトコル
（`hallucination_guard/llm/protocols.py`）を満たすクラスを書けば差し替え可能です。
OpenAI 用のアダプタは同梱しています:

```python
from hallucination_guard.llm.openai_adapter import (
    OpenAIJudgeAdapter,
    OpenAIStructuredAdapter,
)

graph = Graph(
    domain=GeneralDomain(),
    structured_llm=OpenAIStructuredAdapter(model="<openai-model-id>"),
    judge_llm=OpenAIJudgeAdapter(model="<openai-model-id>"),
)
```

`OPENAI_API_KEY` 環境変数からキーを読みます。`model` は明示指定が必須
（デフォルトを設けないことで、誤ったモデルが裏で使われる事故を防ぐ）。
Structured Outputs（`chat.completions.parse` + `response_format=<Pydantic class>`）
を内部で使うため、対応モデルが必要です。

### State を永続化する（checkpointer）

LangGraph の checkpointer を渡すと、各ノード境界でのスナップショットが
保存されます。スレッド ID で世帯を分け、後から `get_state()` で復元可能:

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = Graph(
    domain=GeneralDomain(),
    structured_llm=MyStructuredLLM(),
    judge_llm=MyJudgeLLM(),
    checkpointer=InMemorySaver(),
)

graph.run("first question",  thread_id="alice")
graph.run("second question", thread_id="bob")

snapshot = graph.get_state("alice")
print(snapshot.is_success, snapshot.retry_count)
```

`checkpointer` を渡した場合は `run(..., thread_id=...)` 必須です。

LangGraph 1.1+ は checkpoint に書き戻されない型を deserialize すると
`Deserializing unregistered type ... This will be blocked in a future
version.` という警告を出し、将来のリリースでハードエラーになります。
本フレームワークの Pydantic 型（`GroundedOutput` / `Claim` / `CriticVerdict`
/ `GraphState` / `FailReason`）を許可リストに登録した serializer を
`hallucination_guard.serde.build_serializer()` から取得できます:

```python
from langgraph.checkpoint.memory import InMemorySaver
from hallucination_guard.serde import build_serializer

saver = InMemorySaver(serde=build_serializer())
graph = Graph(
    domain=GeneralDomain(),
    structured_llm=MyStructuredLLM(),
    judge_llm=MyJudgeLLM(),
    checkpointer=saver,
)
```

`DomainConfig.output_schema()` を独自クラスで上書きしているなら
`build_serializer(MyCustomOutput)` で追加もできます。

`Graph(auto_serialize=True)` を渡せば、`Graph` 側が checkpointer の
デフォルト serializer をフレームワーク用 allow-list 付きに自動で
差し替えます（カスタム serializer がすでに設定されている場合は
誤って上書きしないよう `ValueError` を出します）:

```python
saver = InMemorySaver()
graph = Graph(
    domain=GeneralDomain(),
    structured_llm=MyStructuredLLM(),
    judge_llm=MyJudgeLLM(),
    checkpointer=saver,
    auto_serialize=True,    # saver.serde を build_serializer() に置換
)
```

### ストリーミング（ノード単位の進捗）

`Graph.stream()` は各ノードの実行が終わるたびに `StreamEvent` を
yield します。`StreamEvent.state` には **その時点までの累積
`GraphState`** が入っているため、最終イベントの state は `run()` の
戻り値と同じになります。

```python
for event in graph.stream("What is the capital of France?"):
    s = event.state
    print(f"{event.node}: retry={s.retry_count} gate={s.gate_result}")
```

進行中の UI 表示や per-node テレメトリに使えます。`thread_id` の扱いは
`run()` と同じ規約です（checkpointer がある場合は必須）。

`async` 文脈では `Graph.astream()` を使うと、LangGraph の `astream`
を経由して同じ `StreamEvent` 列を非同期に受け取れます:

```python
async for event in graph.astream("What is the capital of France?"):
    print(event.node, event.state.retry_count)
```

`stream()` と同じ累積方式で `research_output` の Pydantic インスタンスを
保持します。実 LLM を `asyncio` で叩く場合の進捗観測に使えます。

単発実行が欲しいだけなら `Graph.arun()` が `run()` の async 版です:

```python
result = await graph.arun("What is the capital of France?")
```

`AsyncStructuredLLM` / `AsyncJudgeLLM` を実装したクライアントを
コンストラクタに渡せば、`Graph` は async-native 経路（LangGraph の async
ノードラッパー）に切り替わります。この場合、誤って `run()` / `stream()`
を呼んだら `RuntimeError` が即時に投げられます（LangGraph から
coroutine が透過的に返るのを防ぐため）。

```python
from hallucination_guard.llm.openai_adapter import (
    AsyncOpenAIJudgeAdapter,
    AsyncOpenAIStructuredAdapter,
)

graph = Graph(
    domain=GeneralDomain(),
    structured_llm=AsyncOpenAIStructuredAdapter(model="<your-model-id>"),
    judge_llm=AsyncOpenAIJudgeAdapter(model="<your-model-id>"),
)
result = await graph.arun("Who painted Guernica?")
```

混在モード（structured だけ async、judge は sync、など）も許容されます。
どちらか一方でも async-only なら `graph.is_async is True` になり、async
エントリポイント（`arun` / `astream`）のみが利用できます。

### 低レベル API（State / Retry 層）

```python
from hallucination_guard.state import GraphState, FailReason
from hallucination_guard.retry.hint_builder import RetryHintBuilder

# 1) イミュータブルな State
state = GraphState(user_query="緑茶は癌を予防しますか？")

# 2) 更新は必ず with_update() 経由 — 元は破壊されない
next_state = state.with_update(
    retry_count=state.retry_count + 1,
    fail_reason=FailReason.NO_SOURCE,
)
assert state.retry_count == 0
assert next_state.retry_count == 1

# 3) RetryDirective を組み立てる（プロンプト注入の唯一の入口）
directive = RetryHintBuilder.build(next_state)
print(directive.fix_instruction)
# → "主張ごとに出典URLを必ず添付してください"
print(directive.forbidden_claims)
# → []
```

CRITIC が前回否定した主張は `fail_history` に
`"critic_rejected:<claim>"` の形式で追記する規約です。

```python
s = GraphState(
    user_query="...",
    fail_reason=FailReason.CRITIC_REJECTED,
    fail_history=["critic_rejected:緑茶は癌を治す"],
)
RetryHintBuilder.build(s).forbidden_claims
# → ['緑茶は癌を治す']
```

---

## 設計ルール

ここを変えると元の保証が成り立たなくなる項目。

### 1. State は必ずイミュータブル更新

```python
# good
new_state = state.with_update(retry_count=state.retry_count + 1)

# bad: 直接ミューテーション禁止
state.retry_count += 1
```

### 2. プロンプトに `fail_history` の生文字列を埋めない

`fail_history` にはユーザー入力や LLM の前回出力に由来する文字列が含まれるため、
**プロンプトインジェクションの侵入経路** になります。
プロンプトに注入できるのは `RetryDirective` 型だけ、と決まっています。

```python
# good
directive = RetryHintBuilder.build(state)
prompt = build_prompt(directive)

# bad
prompt = f"前回の失敗: {state.fail_history}"
```

`RetryHintBuilder` は `INSTRUCTION_MAPS[locale]` の **固定文言** からのみ
`fix_instruction` を生成します。動的な文字列は混入しません。`locale` は
`DomainConfig.retry_locale()` 経由で `StructuredNode` から渡され、デフォルト
は `"en"` です。

`StructuredNode` から見える retry プロンプトの組み立て窓口は
`DomainConfig.format_retry_directive(base_prompt, directive)` です。
セパレータの文言や禁止クレームの並べ方を変えたいときは、ドメイン側で
このメソッドだけを差し替えれば足ります（`StructuredNode` を継承する
必要はありません）。実装上、`fail_history` の生文字列を組み込んではいけない
という不変条件は引き続きこの境界で保たれます。

ビルトインの `GeneralDomain` / `MedicalDomain` は `locale="en"`（既定）と
`locale="ja"` を受け付け、`system_prompt` / `critic_prompt` /
`format_retry_directive` / `retry_locale` の 4 つを同時に切り替えます。
これにより `RetryHintBuilder` が返す `fix_instruction` も locale に追従し、
英語プロンプトに日本語の指示文が混入しません。`verdict=PASS` /
`verdict=FAIL` などの構造化出力マーカーや出典ブランド名（PubMed, WHO,
CDC, Cochrane, NEJM）は、日本語版でも ASCII のまま保たれ、ホスト
allow-list や `CriticVerdict` のパースを壊さない設計です。`Locale` 型
（`Literal["en", "ja"]`）は `hallucination_guard.domain.base.Locale` から
import できます。

### 3. ドメイン知識をフレームワーク本体に書かない

```python
# good
graph = Graph(domain=MedicalDomain())

# bad: フレームワーク本体の if 文に書かない
if domain == "medical":
    threshold = 0.95
```

閾値・出典バリデーション・Critic プロンプト・出力スキーマは
**すべて `DomainConfig` サブクラスに閉じ込める**。

### 4. 無限ループ防止

ルーティング関数では **最初に** `retry_count >= max_retries` をチェックすること。

```python
def route_after_gate(state: GraphState) -> str:
    if state.retry_count >= state.max_retries:
        return "error_output"   # ← 必ず先頭
    if state.gate_result == "FAIL":
        return "retry"
    return "critic"
```

`max_retries` のデフォルトは `3`。`Graph` 初期化時に上書き可能にする予定。

### 5. LLM 呼び出しは `nodes/` 以下にだけ書く

`state.py` / `domain/` / `retry/` から LLM API を直接叩かない。
テスト容易性と関心分離のため。

---

## `fail_history` のエントリ形式

`get_rejected_claims()` が動くために、
`fail_history` の各エントリは次の形式で書きます:

```
"<FailReason.value>:<本文>"
```

例:

```python
"low_confidence:確信度0.4 を返した"
"no_source:出典なしで断定した"
"critic_rejected:緑茶は癌を治す"
```

このうち `critic_rejected:` プレフィックスのものだけが
`get_rejected_claims()` および `RetryDirective.forbidden_claims` に流れます。
**プロンプトに到達するのは prefix を除いた本文のみ** です（プレフィックス文字列は混入しません）。

---

## トラブルシュート

### `ModuleNotFoundError: No module named 'pydantic'`
仮想環境を有効化し忘れているか、依存をインストールしていません。
[セットアップ](#セットアップ) を参照。

### `pytest` が `tests/` を見つけない
ルートディレクトリ（`pyproject.toml` のある場所）で実行してください。
`pyproject.toml` の `[tool.pytest.ini_options]` で `testpaths = ["tests"]` を指定しています。

### `mypy` が import エラーを出す
`langgraph` のような外部依存を `pip install -e ".[dev]"` で
入れていない可能性があります。`hallucination_guard/graph.py` は
`langgraph` を import するため、未インストールだと `mypy` も失敗します。
extras を絞っている場合（例: `.[openai]` だけで `graph` を含めていない）も
同じ事象が起きるので、フル開発時は `.[dev]` または `.[all]` を使ってください。

---

## リリース手順（PyPI）

すべて `Makefile` のターゲットに集約してあります。事前に `make install-dev`
を済ませて `build` / `twine` を取り込んでおくこと。

```bash
make check          # pytest + mypy + ポリシー grep
make build          # dist/*.tar.gz と *.whl を作る
make release-check  # check + build + twine check
make release-test   # TestPyPI へアップロード（事前に ~/.pypirc を設定）
make release        # 本番 PyPI へアップロード
```

リリースを切るときの手順:

1. `pyproject.toml` の `version` を bump
2. `CHANGELOG.md` 先頭に新エントリを追記（Keep a Changelog 形式）
3. `make release-check` で artifacts を検証
4. `make release-test` で TestPyPI に上げて `pip install -i https://test.pypi.org/simple/ hallguard==<ver>` を確認
5. 問題なければ `make release`
6. Git タグを切って push

認証は `~/.pypirc` か `TWINE_USERNAME` / `TWINE_PASSWORD` 環境変数で渡します
（API トークンを推奨）。

---

## ライセンス

MIT。詳細は [LICENSE](LICENSE) を参照。
