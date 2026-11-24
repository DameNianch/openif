"""Semantic categories and ordered lexical evidence for Nasubi."""

import json
from functools import lru_cache
from importlib.resources import files

from .morphology import Token, words

CONTENT = {"名詞", "動詞", "形容詞", "形状詞", "副詞", "代名詞"}
STOP = {
    "する",
    "ある",
    "いる",
    "くださる",
    "くれる",
    "もらう",
    "なる",
    "ほしい",
    "欲しい",
    "こと",
    "もの",
    "ん",
}
CONCEPTS = {
    "remove": "除く 除去する 外す 消す 削除する 取り消す 解除する 取り除く",
    "add": "加える 追加する 入れる 登録する 書き加える 載せる 記す",
    "seek": "探す 検索する 調べる 捜す 見つける",
    "know": "知る 教える 知らせる 尋ねる 問う 聞く 確かめる 確認する",
    "start": "始める 開く 始動する 起動する 開始する つける 再開する",
    "stop": "止める やめる 止まる 中止する 停止する 終える",
    "move": "行く 向かう 来る 到着する 出発する 通る 戻る 帰る",
    "increase": "上げる 増やす 強める 大きい 大きく 高い",
    "decrease": "下げる 減らす 弱める 小さい 小さく 低い",
    "transfer": "渡す 送る 送信する 届ける 伝える 提出する",
    "make": "作る 造る 建てる 用意する 準備する 作成する",
    "want": "たい 欲しい ほしい 望む 希望する 頼む お願い",
    "question": "何 いつ どこ 誰 どう どの いくつ いくら なぜ どれ",
    "time": "今 今日 明日 昨日 来週 来月 朝 昼 夜 午前 午後 週末 翌日",
    "negative": "ない ぬ ず いいえ 嫌い 不要 無い",
    "preference": "好き 好む 気に入る 気にいる お気に入り 苦手 好み",
    "person": "私 あなた 彼 彼女 君 父 母 友人 友達 家族",
}

LEX = {word: category for category, terms in CONCEPTS.items() for word in terms.split()}
LEXICON = json.loads(
    files(__package__).joinpath("lexicon.json").read_text(encoding="utf-8")
)


def category(token: Token) -> str:
    if token.lemma in LEX:
        return LEX[token.lemma]
    if token.lemma + "する" in LEX:
        return LEX[token.lemma + "する"]
    if token.kind == "数詞":
        return "number"
    if token.kind == "固有名詞" or token.subkind in ("助数詞可能", "副詞可能"):
        return token.subkind
    return token.pos


@lru_cache(maxsize=256)
def entry(term: str):
    tokens = words(term)
    return [t.lemma for t in tokens], [t.surface for t in tokens]


def lookup(terms, lemmas, surfaces):
    for term in terms:
        lemma, surface = entry(term)
        size = len(lemma)
        if size:
            for start in range(len(lemmas)):
                if (
                    lemmas[start] == lemma[0] and lemmas[start : start + size] == lemma
                ) or (
                    surfaces[start] == surface[0]
                    and surfaces[start : start + size] == surface
                ):
                    return term
    return None


def evidence(spec, lemmas, surfaces) -> list[str]:
    topic = lookup(spec["topic"], lemmas, surfaces)
    if topic is None:
        return []
    cue = lookup(spec["cue"], lemmas, surfaces)
    if cue and not lookup(spec["veto"], lemmas, surfaces):
        return ["topic", "match", "by_topic/" + topic, "by_cue/" + cue]
    return ["topic"]
