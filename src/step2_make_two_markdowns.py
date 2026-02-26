# -*- coding: utf-8 -*-
import os
import re
import json
import time
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from volcenginesdkarkruntime import Ark

load_dotenv()

BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
API_KEY = os.getenv("ARK_API_KEY")
TEXT_MODEL = os.getenv("ARK_TEXT_MODEL")

if not API_KEY:
    raise RuntimeError("未检测到 ARK_API_KEY，请检查 .env")
if not TEXT_MODEL:
    raise RuntimeError("未检测到 ARK_TEXT_MODEL，请检查 .env")

# KP 输入过长时截断（不需要你调参）
MAX_KP_INPUT_CHARS = 12000


# ---------------------------
# JSON 解析兜底
# ---------------------------
def safe_json_loads(s: str) -> dict:
    """
    尝试解析 JSON。
    允许 JSON 前后夹杂少量文字：截取最外层 {} 再解析。
    """
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(s[l : r + 1])
        raise


def ark_chat_json(
    client: Ark,
    prompt: str,
    temperature: float = 0.4,
    max_retries: int = 3,
    stage: str = "model_call",
) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=TEXT_MODEL,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},  # 强制输出 JSON
            )
            content = resp.choices[0].message.content
            return safe_json_loads(content)
        except Exception as e:
            last_err = e
            msg = str(e)
            if attempt < max_retries:
                wait = 1.5 * attempt
                print(f"[警告] {stage} 第{attempt}次返回非JSON/解析失败，{wait:.1f}s后重试：{msg}", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{stage} 连续{max_retries}次失败：{msg}") from e
    raise RuntimeError(f"{stage} failed") from last_err


# ---------------------------
# 文本清洗：避免控制字符 + 恢复 JSON 吃掉的 LaTeX 反斜杠
# ---------------------------
def recover_latex_from_json_escapes(s: str) -> str:
    """
    把 JSON 解析后产生的控制字符（\b \f \t 等）恢复为 LaTeX 反斜杠形式：
    - \x08(backspace) + 'egin' => '\\begin'
    - \x0c(formfeed) + 'rac'  => '\\frac'
    - \t(tab) + 'imes'        => '\\times'
    """
    if not s:
        return s

    out = []
    n = len(s)
    i = 0

    def is_letter(ch: str) -> bool:
        return ("a" <= ch <= "z") or ("A" <= ch <= "Z")

    while i < n:
        c = s[i]
        nxt = s[i + 1] if i + 1 < n else ""

        if c == "\x08":  # backspace, from \b
            out.append("\\b")
            i += 1
            continue
        if c == "\x0c":  # formfeed, from \f
            out.append("\\f")
            i += 1
            continue
        if c == "\t":  # tab, from \t
            # 如果后面像 LaTeX 命令（imes / ext / an ...），恢复成 \t
            if nxt and is_letter(nxt):
                out.append("\\t")
            else:
                out.append("    ")
            i += 1
            continue
        if c == "\x0b":  # vertical tab
            out.append("\\v")
            i += 1
            continue

        out.append(c)
        i += 1

    return "".join(out)


def clean_md_text(s: str) -> str:
    """
    Markdown 输出也尽量不要包含 NULL 等控制字符。
    保留：\n
    """
    if s is None:
        return ""
    s = recover_latex_from_json_escapes(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 去掉除了 \n 之外的大多数控制字符（0x00-0x1F）
    cleaned = []
    for ch in s:
        code = ord(ch)
        if ch == "\n":
            cleaned.append(ch)
        elif code == 0:
            # NULL 直接丢弃
            continue
        elif 0x01 <= code <= 0x1F:
            # 其他控制字符丢弃
            continue
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def md_escape_leading_numbered_list(line: str) -> str:
    """
    防止 '1.' '12.' 在 Markdown 里被当作有序列表导致格式变化：
    把 '1.' 变成 '1\.'
    """
    m = re.match(r"^(\s*)(\d+)([\.、\)])\s*", line)
    if not m:
        return line
    prefix, num, sep = m.group(1), m.group(2), m.group(3)
    rest = line[m.end() :]
    if sep == ".":
        return f"{prefix}{num}\\.{rest}"
    return f"{prefix}{num}{sep}{rest}"


def md_hardwrap_block(text: str) -> str:
    """
    把多行文本写成“硬换行”（每行末尾加两个空格），
    Pandoc 转 docx 时能保留 A/B/C/D 分行，同时不影响 $...$ 数学公式解析。
    """
    text = clean_md_text(text)
    lines = text.splitlines() if text else [""]
    out_lines = []
    for ln in lines:
        ln = md_escape_leading_numbered_list(ln)
        out_lines.append(ln + "  ")
    return "\n".join(out_lines).rstrip() + "\n"


def section_heading(sec_idx: Optional[int], sec_title: Optional[str]) -> str:
    if sec_idx is not None and sec_title:
        return f"大题{sec_idx}：{sec_title}"
    if sec_idx is not None:
        return f"大题{sec_idx}"
    if sec_title:
        return f"{sec_title}"
    return "未命名分组"


# ---------------------------
# Prompt：核心知识点 + 改编题
# ---------------------------
def build_kp_prompt(chapter_name: str, questions_flat: List[Dict[str, Any]]) -> str:
    """
    核心知识点：复习笔记式概括（不逐题对应），样式仿你给的示例。
    """
    lines: List[str] = []
    for q in questions_flat:
        uid = q.get("uid")
        sec_idx = q.get("section_index")
        sec_title = q.get("section_title")
        qno = q.get("question_number")
        text = (q.get("raw_text") or "").replace("\n", " ")
        text = text[:180]
        lines.append(f"{uid} | {sec_idx}/{sec_title} | 小题{qno}：{text}")

    joined = "\n".join(lines)
    if len(joined) > MAX_KP_INPUT_CHARS:
        joined = joined[:MAX_KP_INPUT_CHARS] + "\n（后续题目摘要已截断）"

    return f"""
你将收到某章节的一批试题摘要。请输出“核心知识点总结（复习笔记风格）”。
不需要逐题对应，只需概括本章主线与高频考点。

必须只输出 JSON（不要输出解释、不要 markdown、不要任何多余字符）。

写作风格（严格仿照示例）：
- 用“一、二、三 …”作为一级标题
- 一级标题下可用“1. 2. 3.”小点
- 强调高频考点/易错点/对比点

JSON 结构（必须完全一致）：
{{
  "kp_title": "{chapter_name}核心知识点",
  "kp_text": "string"
}}

题目摘要：
{joined}
""".strip()


def build_rewrite_prompt(uid: str, kp_text: str, raw: str, fig_note: Optional[str]) -> str:
    """
    改编题：保持题型风格（例如选择题仍要 A/B/C/D），不输出图片内容。
    输出要包含答案与解析；解析尽量中文。
    """
    fig_tip = ""
    if fig_note:
        fig_tip = f"\n注意：原题含图，图备注：{fig_note}\n本题不允许改编，请直接返回 JSON，rewritten_text/answer/explanation 全部置空字符串。\n"

    return f"""
你将收到一道原题（识别稿）以及本章核心知识点笔记。
请基于知识点对该题进行改编（保持题型一致；若为单项选择题必须有 A/B/C/D 四个选项）。
改编后的题目要“风格相近但不能与原题相同”，可以参考原题的设问方式与数据规模，但不要照抄。

非常重要：
- 改编题中如出现数学/化学/物理公式，请用 LaTeX 且用 $...$ 包裹（行内即可）。
- 只输出 JSON（不要输出解释、不要 markdown、不要任何多余字符）。
{fig_tip}

输出 JSON 结构（必须完全一致）：
{{
  "uid": "{uid}",
  "rewritten_text": "string",
  "answer": "string",
  "explanation": "string"
}}

本章核心知识点笔记：
{kp_text}

原题（识别稿）：
{raw}
""".strip()


# ---------------------------
# 抽样改编目标
# ---------------------------
def choose_rewrite_targets(questions_flat: List[Dict[str, Any]], ratio: float, seed: int) -> List[Dict[str, Any]]:
    eligible = [
        q for q in questions_flat
        if (q.get("uid") and (q.get("raw_text") or "").strip() and (not q.get("has_figure", False)))
    ]
    n = len(eligible)
    if n == 0:
        return []
    k = int(math.floor(n * ratio))
    if ratio > 0 and k == 0:
        k = 1
    k = min(k, n)
    rng = random.Random(seed)
    return rng.sample(eligible, k=k)


# ---------------------------
# Markdown 导出（替代 docx）
# ---------------------------
def export_md_raw(chapter_name: str, chapter_json: dict, out_path: Path) -> None:
    pages: List[dict] = chapter_json.get("pages", [])
    pages_sorted = sorted(pages, key=lambda p: p.get("_meta", {}).get("page_no", 10**9))

    buf: List[str] = []
    buf.append(f"# {chapter_name}（原文整理版）\n")

    for page in pages_sorted:
        meta = page.get("_meta", {})
        page_no = meta.get("page_no")
        image = meta.get("image")
        buf.append(f"## 第 {page_no} 页（{image}）\n")

        for sec in page.get("sections", []):
            sec_idx = sec.get("section_index")
            sec_title = sec.get("section_title")
            buf.append(f"### {section_heading(sec_idx, sec_title)}\n")

            for q in sec.get("questions", []):
                qno = q.get("question_number")
                title = f"#### 小题 {qno}" if qno is not None else "#### 小题（题号缺失）"
                buf.append(title + "\n")

                if q.get("has_figure", False):
                    note = q.get("figure_note") or ""
                    buf.append(md_hardwrap_block(f"[含图] {note}".strip()) + "\n")

                raw_text = q.get("raw_text", "")
                buf.append(md_hardwrap_block(raw_text) + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(buf), encoding="utf-8")


def export_md_kp_and_rewrites(chapter_name: str, kp: dict, rewrites: List[dict], out_path: Path) -> None:
    buf: List[str] = []
    buf.append(f"# {chapter_name}（核心知识点 + 改编题）\n")

    kp_title = clean_md_text(kp.get("kp_title", f"{chapter_name}核心知识点"))
    kp_text = kp.get("kp_text", "")

    buf.append(f"## {kp_title}\n")
    buf.append(md_hardwrap_block(kp_text) + "\n")

    buf.append("## 改编题（含答案与解析）\n")
    if not rewrites:
        buf.append("本章无可改编题（可能因为题目均含图或比例为0）。\n")
    else:
        for item in rewrites:
            uid = item.get("uid")
            sec_idx = item.get("source_section_index")
            sec_title = item.get("source_section_title")
            qno = item.get("source_question_number")
            src_uid = item.get("source_uid")

            buf.append(f"### 改编映射：大题{sec_idx}-小题{qno}（原uid={src_uid}） -> 新uid={uid}\n")
            if sec_title:
                buf.append(f"- 原分组：{clean_md_text(str(sec_title))}\n")

            # 原题（便于校验）
            raw = item.get("source_raw_text", "")
            buf.append("#### 原题（识别稿）\n")
            buf.append(md_hardwrap_block(raw) + "\n")

            # 改编题
            buf.append("#### 改编题\n")
            buf.append(md_hardwrap_block(item.get("rewritten_text", "")) + "\n")

            ans = clean_md_text(item.get("answer", ""))
            exp = item.get("explanation", "")

            buf.append(f"**答案：** {ans}\n\n")
            buf.append("**解析：**\n\n")
            buf.append(md_hardwrap_block(exp) + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(buf), encoding="utf-8")


# ---------------------------
# 主流程
# ---------------------------
def main(chapter: str, ratio: float = 0.3, seed: int = 42) -> None:
    chapter_name = chapter
    out_dir = Path("out") / chapter_name
    in_json_path = out_dir / f"{chapter_name}.json"

    if not in_json_path.exists():
        raise FileNotFoundError(f"未找到章节 JSON：{in_json_path.resolve()}（请先跑 step1）")

    chapter_json = json.loads(in_json_path.read_text(encoding="utf-8"))
    questions_flat: List[Dict[str, Any]] = chapter_json.get("questions", [])
    pages: List[Dict[str, Any]] = chapter_json.get("pages", [])

    if not questions_flat and pages:
        # 兜底：如果没扁平 questions，就从 pages 拉平
        seq = 0
        for page in pages:
            meta = page.get("_meta", {})
            page_no = meta.get("page_no")
            image = meta.get("image")
            for sec in page.get("sections", []):
                sec_idx = sec.get("section_index")
                sec_title = sec.get("section_title")
                sec_key = f"IDX:{sec_idx}|{sec_title}"
                for q in sec.get("questions", []):
                    seq += 1
                    questions_flat.append({
                        "uid": f"Q{seq:05d}",
                        "chapter": chapter_name,
                        "page_no": page_no,
                        "image": image,
                        "section_index": sec_idx,
                        "section_title": sec_title,
                        "section_key": sec_key,
                        "question_number": q.get("question_number"),
                        "raw_text": q.get("raw_text", ""),
                        "has_figure": bool(q.get("has_figure", False)),
                        "figure_note": q.get("figure_note", None),
                        "errors": q.get("errors", []),
                        "confidence": float(q.get("confidence", 0.85)),
                    })

    print("[阶段] 1/4 生成“原文整理版.md”...", flush=True)
    md_raw_path = out_dir / f"{chapter_name}_原文整理版.md"
    export_md_raw(chapter_name, chapter_json, md_raw_path)
    print(f"[完成] 输出：{md_raw_path.resolve()}", flush=True)

    print("[阶段] 2/4 调用模型生成核心知识点（JSON）...", flush=True)
    client = Ark(base_url=BASE_URL, api_key=API_KEY)
    kp_prompt = build_kp_prompt(chapter_name, questions_flat)
    kp = ark_chat_json(client, kp_prompt, temperature=0.4, max_retries=3, stage="kp")
    kp["kp_title"] = clean_md_text(kp.get("kp_title", f"{chapter_name}核心知识点"))
    kp["kp_text"] = clean_md_text(kp.get("kp_text", ""))

    print("[阶段] 3/4 随机抽题并逐题改编（JSON）...", flush=True)
    targets = choose_rewrite_targets(questions_flat, ratio=ratio, seed=seed)
    rewrites: List[dict] = []

    for q in tqdm(targets, desc="改编中", ncols=80):
        uid = q.get("uid")
        raw = clean_md_text(q.get("raw_text", ""))
        fig_note = q.get("figure_note") if q.get("has_figure", False) else None

        prompt = build_rewrite_prompt(uid=uid, kp_text=kp["kp_text"], raw=raw, fig_note=fig_note)

        try:
            out = ark_chat_json(client, prompt, temperature=0.5, max_retries=3, stage=f"rewrite:{uid}")
        except Exception as e:
            # 连续失败：跳过该题
            print(f"[跳过] {uid} 改编失败：{e}", flush=True)
            continue

        item = {
            "uid": out.get("uid", uid),
            "rewritten_text": clean_md_text(out.get("rewritten_text", "")),
            "answer": clean_md_text(out.get("answer", "")),
            "explanation": clean_md_text(out.get("explanation", "")),

            # 映射信息（便于校验）
            "source_uid": uid,
            "source_section_index": q.get("section_index"),
            "source_section_title": q.get("section_title"),
            "source_question_number": q.get("question_number"),
            "source_raw_text": clean_md_text(q.get("raw_text", "")),
        }
        rewrites.append(item)

    print("[阶段] 4/4 生成“核心知识点+改编题.md”...", flush=True)
    md_new_path = out_dir / f"{chapter_name}_核心知识点与改编题.md"
    export_md_kp_and_rewrites(chapter_name, kp, rewrites, md_new_path)
    print(f"[完成] 输出：{md_new_path.resolve()}", flush=True)

    # 同时把 step2 结果保存成 JSON，方便你后续调试/复用
    out_json = out_dir / f"{chapter_name}_step2_output.json"
    out_json.write_text(json.dumps({"kp": kp, "rewrites": rewrites}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 输出：{out_json.resolve()}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("chapter", help="章节名，例如 ch01")
    parser.add_argument("--ratio", type=float, default=0.3, help="改编比例（0~1）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    main(args.chapter, ratio=args.ratio, seed=args.seed)
