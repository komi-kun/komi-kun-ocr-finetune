#!/usr/bin/env python3
"""Build the text corpus used by the synthetic manga renderer.

Sources (all freely licensed):
  * Japanese Wikipedia (CC BY-SA 4.0) - sentence-like fragments
  * a hand-written onomatopoeia / manga-interjection list (common words, no copyright)

Output:
  data/synth/corpus.txt          one line per training string
  data/synth/onomatopoeia.txt    SFX subset (also mixed into corpus.txt)
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synth"

# Common Japanese onomatopoeia / manga interjections.  Individual words are not
# copyrightable; the *rendering* (font, colour, distortion) is what we train on.
ONOMATOPOEIA = """
ドドド ゴゴゴ ズズズ バババ ガガガ ズドド ドドン バァン ドーン ズバッ バキッ バキバキ ゴキッ
グキッ メキッ ビキッ ピキッ ミシッ ギシギシ ミシミシ パキパキ バリバリ ビリビリ ゴリゴリ ガリガリ
ザラザラ ゴワゴワ ガサガサ バサバサ パサパサ ヒラヒラ ヒラリ フワリ フワフワ ユラユラ グラグラ
グルグル クルクル コロコロ ゴロゴロ ドタドタ ドタバタ バタバタ スタスタ サッサッ スッ ススス
シュッ シュルル シュパッ ビュン ビューン ヒュン ヒューン ビシッ ビシッと ピシッ バシッ ベシッ
ゴツン コツン コンコン トントン ドンドン ガンガン バンバン パンパン チクチク チクリ ズキズキ
ズキン キリキリ シクシク ジンジン ヒリヒリ ムズムズ モゾモゾ ソワソワ ワクワク ドキドキ ドキッ
キュン キュンキュン ズキュン ギュッ ギュー ぎゅっ ハグッ ムギュ ニギニギ ワシャッ ガシッ
ガッチリ グッ グイッ ググッ クイッ ニョキッ ニュッ ニョロニョロ ヌルヌル ツルツル ベタベタ
ネバネバ サラサラ ツヤツヤ ピカピカ キラキラ ギラギラ チカチカ パチパチ バチバチ ピチピチ
プクプク ブクブク モクモク ケムケム ホワホワ モワモワ ムワッ フワッ パァ パアッ ポワッ
ハァ ハァハァ フー フゥ ホッ ふぅ ため息 ゴクリ コクリ クチャ クチャクチャ モグモグ パクパク
ムシャムシャ ガブッ ガブガブ ペロッ ペロペロ チュッ チュッチュ ビクッ ビクビク ブルブル
ガタガタ プルプル ワナワナ フルフル ゾクッ ゾクゾク ヒヤッ ヒヤヒヤ ハラハラ ドキドキ
ギクッ ギョッ ギョッと ハッ ハッと ふと じっ ジッ ジーッ じーっ じろじろ チラッ チラチラ
クスッ クスクス フフフ ふふっ ニヤリ ニヤッ ニコッ ニコニコ ニタニタ ニマニマ ゲラゲラ
ガハハ ワハハ アハハ ハハハ ウフフ ンフフ クックッ ホホホ オホホ ケラケラ ギャハハ
うわっ うわぁ わっ わあっ きゃっ きゃあ ぎゃあ ああっ えっ ええっ はっ はあ へえ ほお
おおっ おおお うおお うぉお ぬおお ぐおお ぐぬぬ ぬぬぬ むむっ むう ふむ ふうむ うむ
おや おやおや あら あらまあ まあ なんと これは やあ よう おい こら こらっ おっと
いてっ いたっ いたた あいた ぐはっ がはっ うげっ げっ げえっ うえっ おえっ うっぷ
ぷはっ ぷはぁ ふぅっ すぅ すー はぁ はあぁ んっ んんっ ふんっ ふーん へっ へへっ
しん しーん シーン ざわ ざわざわ ザワザワ がやがや わいわい どよどよ ひそひそ こそこそ
パチン カチッ カチカチ チッチッ コチコチ ジジジ ブーン ブンブン ヴヴヴ ゴォォ ゴオオ
ボォ ボワン ドワッ ダダダ ダッ ダッシュ ザザッ ザッ ザザー ジャー ジャジャジャ チャッ
チャチャ シャシャ シャー シュー シュワッ シュワシュワ プシュッ プシュー ポン ポンッ
ポンポン パチパチ パンッ パーン ピン ピーン ピピッ ピーッ ブブー ブー ブブッ
カーン ガーン ガガーン ズガーン ドカーン バコーン バコッ ゴッ ゴスッ ガツッ ガツン
ドスッ ドスドス ズシッ ズシン ドシン ドシッ バタン バタッ ガチャ ガチャッ ガチャガチャ
ギィ ギイィ キィ キーッ キュッ キュルル クルッ グルッ バッ パッ パァッ サァッ スーッ
ヒューヒュー ピューピュー ゴォー ゴーッ ザァー ザアア ドドッ ドドー ゴゴー ズズー
""".split()


def clean(s: str) -> str:
    s = s.replace("　", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # keep only characters the model dictionary knows + ascii space
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wiki", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seen = set()
    corpus = []

    def add(s: str):
        s = clean(s)
        if not (2 <= len(s) <= 24):
            return
        if any(c in s for c in "{}[]<>|\\"):
            return
        h = hashlib.md5(s.encode()).hexdigest()
        if h in seen:
            return
        seen.add(h)
        corpus.append(s)

    for w in ONOMATOPOEIA:
        add(w)

    if args.n_wiki:
        from datasets import load_dataset

        print(f"streaming wikipedia ({args.n_wiki} lines) ...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
        for ex in ds:
            if len(corpus) >= len(ONOMATOPOEIA) + args.n_wiki:
                break
            for sent in re.split(r"(?<=[。！？!?…])", ex["text"]):
                sent = sent.strip()
                if 3 <= len(sent) <= 24 and not sent.startswith(("「", "『", "*", "|", "=")):
                    add(sent)

    (args.out / "onomatopoeia.txt").write_text("\n".join(ONOMATOPOEIA) + "\n", encoding="utf-8")
    (args.out / "corpus.txt").write_text("\n".join(corpus) + "\n", encoding="utf-8")
    print(f"wrote {len(corpus)} lines -> {args.out/'corpus.txt'}")


if __name__ == "__main__":
    main()
