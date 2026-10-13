# EdgeSLM_Offload

モデルの一部の層をGPUにオフロードして実行したとき、推論性能がどう振る舞うかを検証するプロジェクト。

## これは何か

Qwen2.5-3B（36層）の各層を CPU / GPU のどちらに置くかを任意に制御し、次の2点を定量的に測る。

1. **オフロード率と速度の関係** — GPUに載せる層数を 0〜36 で振ったときの TTFT / tokens per sec の曲線
2. **デバイス境界を跨ぐコスト** — オフロード率を50%に固定したまま境界数だけを 1 / 5 / 17 / 35 と振り、境界1回あたりの単位コストを回帰で求める

2 が本プロジェクトの中核。オフロード率が同一なら計算量は同じなので、水準間の差は境界コストのみに由来する。

## EdgeSLM との関係

[EdgeSLM](../EdgeSLM) とは**独立した別プロジェクト**。

- EdgeSLM のコード・ドキュメントは変更しない。必要な部品はコピーして取り込む
- EdgeSLM の Phase B（Qualcomm SoC / QNN）への正式なインプットではなく、**参考**という位置づけ
- 本検証の数値は PC環境（NVIDIA GPU + x86 CPU）のものであり、エッジ端末の NPU/HTP の性能予測には使わない

## 技術選定

| 区分 | 選定 | 理由 |
|---|---|---|
| 主軸 | PyTorch + accelerate | 任意の層を任意のデバイスに置ける唯一の選択肢。ONNX Runtime には層単位の割り当てAPIがなく、llama.cpp の `-ngl` は連続分割しかできない |
| 対照 | llama.cpp（Vulkan、ビルド済みバイナリ） | 成熟した実装の `-ngl` 曲線と突き合わせる。Vulkan SDK の導入は不要 |
| 精度 | float32 統一 | 本機の CPU は AVX-512 / bf16命令を持たず、bf16行列積が fp32 の 1/620 しか出ないため（実測）。詳細は要件定義書 FR-2 |

## 結果

**[検証結果まとめ](DoC/検証結果まとめ.md) ← まずこれを読む**

三行で言うと:

1. decode時間はGPUに載せた層数に対して直線的に減る（R²=0.995）
2. その逆数なので速度は双曲線。**50%載せても全体の伸びの14%**、半分の効果に89%の層が必要
3. 同じ層数でも**配置が断片的だと遅い**。1層おき交互は連続配置より4.23%遅く、境界1回あたり365 µs

llama.cpp（Vulkan、同一モデル）でも同じ形状が再現した（相関 r = 0.9889）。

## ドキュメント

- [検証結果まとめ](DoC/検証結果まとめ.md) — 結論のみ
- [実験記録](DoC/実験記録.md) — 時系列の詳細（失敗と切り分けを含む）
- [要件定義書](DoC/要件定義書.md) / [基本設計](DoC/基本設計_オフロード検証設計.md) — 設計文書は `DoC/` 配下
- [StudyMemo_CPU.md](StudyMemo_CPU.md) — 学習メモ（数値形式とCPU命令セット）。テーマごとに `StudyMemo_<テーマ>.md` で直下に置く

## 使い方

```bash
uv sync
```

| コマンド | 内容 |
|---|---|
| `uv run python -X utf8 -m app.cli check` | 環境・配置・境界数の整合を確認（計測なし） |
| `uv run python -X utf8 -m app.cli offload` | オフロード率スイープ（Step 1） |
| `uv run python -X utf8 -m app.cli noise` | ノイズフロア確認（Step 2 / FR-7） |
| `uv run python -X utf8 -m app.cli boundary` | 境界数スイープ（Step 3・中核実験） |
| `uv run python -X utf8 -m app.cli analyze results/xxx.csv` | 集計・回帰・プロット |

検証用スクリプト:

| スクリプト | 内容 |
|---|---|
| `scripts/smoke_test.py` | 配置・境界数・トークン列一致の確認 |
| `scripts/probe_memory.py` | ロード／配置変更のピークRAM・VRAM |
| `scripts/probe_hook_overhead.py` | 計測装置（フック）自身のコスト |

テスト: `uv run python -X utf8 -m pytest -q`

## 実行前の注意

**他プロセスがRAM/VRAMを占有していると計測が成立しません。** 実測では `llama-server`（9Bモデルを `-ngl 999` で常駐）がVRAM 9.4 GiB・RAM 13.3 GBを保持しており、スワップにより速度が140倍劣化しました。計測前に空きRAM 15GB以上・空きVRAM 13GB以上を確保してください。経緯は [実験記録](DoC/実験記録.md) を参照。

Python 3.12 / uv 管理。Git管理は行わない。
