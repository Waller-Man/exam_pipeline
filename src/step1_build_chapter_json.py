# -*- coding: utf-8 -*-
import os
import re
import json
import time
import base64
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm
from volcenginesdkarkruntime import Ark

load_dotenv()

BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
API_KEY = os.getenv("ARK_API_KEY")
VISION_MODEL = os.getenv("ARK_VISION_MODEL")


# ---------- 工具函数 ----------

def safe_json_loads(s: str) -> dict:
    """兜底解析：允许 JSON 前后混入少量文字，截取最外层 {} 再 loads。"""
    s = (s or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def image_to_data_url(img_path: Path) -> str:
    suffix = img_path.suffix.lower().lstrip(".")
    if suffix in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif suffix == "png":
        mime = "image/png"
    elif suffix == "webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_page_no_from_name(name: str) -> int:
    """
    强制要求：文件名（不含扩展名）必须是纯数字页码，例如：
    1.jpg / 001.png / 12.webp
    否则抛异常。
    """
    stem = Path(name).stem
    if not re.fullmatch(r"\d+", stem):
        raise ValueError(f"页码格式错误：{name}（要求文件名必须为纯数字页码，如 001.jpg）")
    return int(stem)


def list_images_sorted(chapter_path: Path) -> List[Path]:
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        imgs.extend(chapter_path.glob(ext))
    imgs = sorted(imgs, key=lambda p: (extract_page_no_from_name(p.name), p.name))
    return imgs


def section_key(sec_idx: Optional[int], sec_title: Optional[str]) -> str:
    if sec_idx is not None:
        return f"IDX:{sec_idx}|{sec_title or ''}"
    if sec_title:
        return f"TTL:{sec_title}"
    return "UNKNOWN"


# ---------- 模型调用 ----------

def build_prompt_sections() -> str:
    """
    输出结构：按大题/题型分组（sections），每个 section 内有 questions。
    小题题号允许在不同大题里重复（例如大题二重新从1开始）。
    """
    return (
        "你是一名“试题图片转文本”助手。该图片是一页试卷/练习题扫描页，可能包含多个大题/题型，"
        "且每个大题内部小题题号可能从1重新开始。请按“大题 -> 题目”的层级结构输出。\n"
        "必须只输出 JSON（不要输出解释、不要 markdown、不要多余字符）。\n\n"

        "【关键一致性规则（必须严格遵守）】\n"
        "A) 对于“材料题/英语阅读理解/完形填空/语篇填空/阅读材料题/篇章类题目”：\n"
        "   - 只要该页出现一段较长的文章/材料/对话/说明文字，并且后面跟着多个小题（通常题号连续），\n"
        "     你必须把“文章/材料 + 后续所有相关小题”合并成 questions 数组里的【一个】对象（当作一道大题题目）。\n"
        "   - 严禁把文章本身当作一条小题；也严禁把同一篇文章拆成多条 question。\n"
        "   - 合并后的 raw_text 必须包含完整结构：\n"
        "     【材料/原文】\\n(文章全文)\\n\\n【小题1】\\n(题干+选项)\\n\\n【小题2】... 直到该材料对应的小题结束。\n"
        "   - 合并后的 question_number：填该材料题组的“起始小题号”。若材料对应 11~15，则 question_number=11。\n"
        "   - 如果你能明确材料对应的小题范围（例如“回答11-15题”“Questions 11-15”），请在 raw_text 开头额外加一行：\n"
        "     【题组范围】11-15\n"
        "B) 对于非材料题：仍按“一小题一个 question”输出。\n\n"
        "C) 所有数学公式只允许用 $...$ 或 $$...$$。在数学公式内，所有 LaTeX 控制序列一律使用单个反斜杠（例如 \geq \leq \frac \sqrt），禁止输出 \\geq、\\leq、\\frac 这种双反斜杠形式。只有在 \begin{aligned} / \begin{array} 等对齐环境里需要换行时，才允许使用 \\ 作为“换行”，且 \\ 后必须跟空格或换行，而不是字母。"

        "识别要求：\n"
        "1) 将页面按大题/题型分组输出到 sections 数组。若同一页出现多个大题（如“一、… 二、…”），必须拆分为多个 section。\n"
        "2) section 字段：\n"
        "   - section_index：大题编号（从“一、二、三/第1部分/第2部分”等识别；能确认填整数，否则填 null）\n"
        "   - section_title：大题标题或题型名称（如“单项选择题/填空题/判断题/简答题/阅读理解/完形填空/语篇填空”等；无法确认填 null）\n"
        "   - questions：该大题内的题目数组。\n"
        "3) question 字段：\n"
        "   - question_number：\n"
        "       * 普通题：小题题号（整数；无法确认填 null 并在 errors 加入 \"MISSING_QNO\"）\n"
        "       * 材料题组：填该题组“第一道小题号”（例如 11-15 填 11）。\n"
        "   - raw_text：题干+选项，尽量保留换行与符号。材料题组必须按【材料/原文】与【小题x】格式输出。\n"
        "   - has_figure：该题是否含图/表/坐标图等非纯文字内容（材料题组中任一小题含图则为 true）\n"
        "   - figure_note：若 has_figure=true，用一句话描述图类型/关键元素；否则 null\n"
        "   - errors：数组（可包含：\"MISSING_QNO\"、\"TRUNCATED\"、\"MATERIAL_GROUP_UNCERTAIN\" 等）\n"
        "   - confidence：0~1\n"
        "4) 排序：普通题按 question_number 升序；材料题组按其首题号排序。\n"
        "5) 如果你判断存在材料题组，但无法确定哪些小题属于同一材料，请仍然合并为一个 question，\n"
        "   并在该 question 的 errors 中加入 \"MATERIAL_GROUP_UNCERTAIN\"。\n"
        "6) 如果该页明显包含多个大题但你无法可靠分割，请在顶层 errors 中加入 \"SPLIT_UNCERTAIN\"。\n\n"

        "JSON 结构（必须完全一致）：\n"
        "{\n"
        '  "sections": [\n'
        "    {\n"
        '      "section_index": int|null,\n'
        '      "section_title": "string|null",\n'
        '      "questions": [\n'
        "        {\n"
        '          "question_number": int|null,\n'
        '          "raw_text": "string",\n'
        '          "has_figure": true|false,\n'
        '          "figure_note": "string|null",\n'
        '          "errors": ["..."],\n'
        '          "confidence": 0.0\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "errors": ["..."],\n'
        '  "confidence": 0.0\n'
        "}\n"
    )


def call_vision_page(client: Ark, img_path: Path, max_retries: int = 3) -> dict:
    prompt = build_prompt_sections()
    data_url = image_to_data_url(img_path)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=VISION_MODEL,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                response_format={"type": "json_object"},
            )
            return safe_json_loads(resp.choices[0].message.content)
        except Exception as e:
            msg = str(e)
            if attempt < max_retries:
                wait = 1.5 * attempt
                print(f"[警告] {img_path.name} 第{attempt}次识别失败，将在{wait:.1f}s后重试：{msg}", flush=True)
                time.sleep(wait)
                continue
            raise


# ---------- 合并与输出 ----------

def normalize_page_schema(page: Any) -> dict:
    """
    归一化为：
    {
      "sections": [ { "section_index": int|null, "section_title": str|null, "questions":[{...}] } ],
      "errors": [...],
      "confidence": float
    }
    并尽量把“字符串形式的 section / question”兜底成对象，避免后续 .get() 崩溃。
    """
    if not isinstance(page, dict):
        return {"sections": [], "errors": ["BAD_SCHEMA"], "confidence": 0.0}

    errors = page.get("errors") or []
    if not isinstance(errors, list):
        errors = ["BAD_SCHEMA_ERRORS"]

    sections = page.get("sections", [])
    if isinstance(sections, dict):
        sections = [sections]
    elif isinstance(sections, str):
        sections = [{"section_index": None, "section_title": sections, "questions": []}]
        errors.append("SECTION_WAS_STRING")
    elif not isinstance(sections, list):
        sections = []
        errors.append("BAD_SCHEMA_SECTIONS")

    norm_sections: List[dict] = []

    for sec in sections:
        if isinstance(sec, str):
            sec = {"section_index": None, "section_title": sec, "questions": []}
            errors.append("SECTION_WAS_STRING")

        if not isinstance(sec, dict):
            errors.append("SECTION_BAD_TYPE")
            continue

        sec.setdefault("section_index", None)
        sec.setdefault("section_title", None)

        qs = sec.get("questions", [])
        if isinstance(qs, dict):
            qs = [qs]
        elif isinstance(qs, str):
            qs = [{
                "question_number": None,
                "raw_text": qs,
                "has_figure": False,
                "figure_note": None,
                "errors": ["Q_WAS_STRING"],
                "confidence": 0.0,
            }]
            errors.append("QUESTIONS_WAS_STRING")
        elif not isinstance(qs, list):
            qs = []
            errors.append("BAD_SCHEMA_QUESTIONS")

        norm_qs: List[dict] = []
        for q in qs:
            if isinstance(q, str):
                q = {
                    "question_number": None,
                    "raw_text": q,
                    "has_figure": False,
                    "figure_note": None,
                    "errors": ["Q_WAS_STRING"],
                    "confidence": 0.0,
                }
                errors.append("Q_WAS_STRING")

            if not isinstance(q, dict):
                errors.append("Q_BAD_TYPE")
                continue

            q.setdefault("question_number", None)
            q.setdefault("raw_text", "")
            q.setdefault("has_figure", False)
            q.setdefault("figure_note", None)
            q.setdefault("errors", [])
            q.setdefault("confidence", None)

            if not isinstance(q["errors"], list):
                q["errors"] = ["BAD_Q_ERRORS"]

            norm_qs.append(q)

        sec["questions"] = norm_qs
        norm_sections.append(sec)

    page["sections"] = norm_sections
    page["errors"] = list(dict.fromkeys(errors))

    conf = page.get("confidence", 0.0)
    if not isinstance(conf, (int, float)):
        page["confidence"] = 0.0

    return page


def infer_section_if_missing(
    page: dict,
    last_section: Optional[Tuple[Optional[int], Optional[str]]]
) -> Tuple[dict, Optional[Tuple[Optional[int], Optional[str]]]]:
    """
    若本页只有一个 section 且 section_index/title 都为空，
    并且上一页存在 last_section，则继承上一页的大题信息。
    """
    sections = page.get("sections", [])
    if not isinstance(sections, list):
        page["errors"] = list(set(page.get("errors", []) + ["BAD_SCHEMA"]))
        return page, last_section

    if len(sections) == 1:
        sec = sections[0]
        idx = sec.get("section_index")
        title = sec.get("section_title")
        if (idx is None and (title is None or str(title).strip() == "")) and last_section is not None:
            sec["section_index"] = last_section[0]
            sec["section_title"] = last_section[1]
            sec["_meta"] = sec.get("_meta", {})
            sec["_meta"]["section_inferred"] = True

    new_last = last_section
    for sec in sections:
        idx = sec.get("section_index")
        title = sec.get("section_title")
        if idx is not None or (title is not None and str(title).strip() != ""):
            new_last = (idx, title)
    return page, new_last


def flatten_questions(chapter_name: str, pages: List[dict]) -> List[dict]:
    """
    拉平成题目列表，并生成不冲突的 uid：
    - 优先：S<section_index>-Q<question_number>
    - 若 section_index 缺失：用 section_title
    - 若仍缺：用 P<page>-Q<question_number>
    若遇到同 uid 重复，加 -2/-3 后缀。
    """
    flat = []
    uid_count: Dict[str, int] = {}

    for page in pages:
        page_no = page["_meta"]["page_no"]
        image = page["_meta"]["image"]

        for sec in page.get("sections", []):
            sec_idx = sec.get("section_index")
            sec_title = sec.get("section_title")
            sk = section_key(sec_idx, sec_title)

            for q in sec.get("questions", []):
                qno = q.get("question_number")
                if qno is None:
                    base_uid = f"P{page_no}-UNK"
                else:
                    if sec_idx is not None:
                        base_uid = f"S{sec_idx}-Q{qno}"
                    elif sec_title:
                        base_uid = f"{sec_title}-Q{qno}"
                    else:
                        base_uid = f"P{page_no}-Q{qno}"

                uid_count[base_uid] = uid_count.get(base_uid, 0) + 1
                uid = base_uid if uid_count[base_uid] == 1 else f"{base_uid}-{uid_count[base_uid]}"

                flat.append({
                    "uid": uid,
                    "chapter": chapter_name,
                    "page_no": page_no,
                    "image": image,
                    "section_index": sec_idx,
                    "section_title": sec_title,
                    "section_key": sk,
                    "question_number": qno,
                    "raw_text": q.get("raw_text", ""),
                    "has_figure": bool(q.get("has_figure", False)),
                    "figure_note": q.get("figure_note"),
                    "errors": q.get("errors", []),
                    "confidence": q.get("confidence", None),
                    "section_inferred": bool(sec.get("_meta", {}).get("section_inferred", False)),
                })

    return flat


# ---------- 并发识别 ----------

def _worker_call_vision(img_path: Path, page_no: int, max_retries: int) -> Tuple[int, str, dict]:
    """
    每个线程一个 Ark client（更稳，避免共享 client 的线程安全问题）。
    返回：(page_no, image_name, page_json)
    """
    client = Ark(base_url=BASE_URL, api_key=API_KEY)
    page = call_vision_page(client, img_path, max_retries=max_retries)
    return page_no, img_path.name, page


def recognize_pages_concurrent(
    imgs: List[Path],
    max_workers: int = 5,
    max_retries: int = 3,
) -> Dict[str, dict]:
    """
    并发识别所有图片。返回 dict：{image_name: page_json}
    """
    max_workers = max(1, min(int(max_workers), len(imgs)))
    results: Dict[str, dict] = {}

    tasks: List[Tuple[int, Path]] = []
    for img in imgs:
        page_no = extract_page_no_from_name(img.name)
        tasks.append((page_no, img))

    print(f"[阶段] 2/5 并发识别中... workers={max_workers}，任务数={len(tasks)}", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(_worker_call_vision, img, page_no, max_retries): (page_no, img)
            for (page_no, img) in tasks
        }

        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="识别页(并发)", ncols=90):
            page_no, img = future_map[fut]
            try:
                got_page_no, img_name, page = fut.result()
                results[img_name] = page
            except Exception as e:
                raise RuntimeError(f"识别失败：page_no={page_no} file={img.name} err={e}") from e

    return results


# ---------- 主流程 ----------

def main(chapter_dir: str, workers: int = 5, max_retries: int = 3):
    if not API_KEY or not VISION_MODEL:
        raise RuntimeError("缺少环境变量：ARK_API_KEY 或 ARK_VISION_MODEL（请检查 .env）")

    chapter_path = Path(chapter_dir)
    if not chapter_path.exists():
        raise FileNotFoundError(f"章节目录不存在：{chapter_path.resolve()}")

    chapter_name = chapter_path.name

    # 扫描图片并检查页码格式
    imgs_raw = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        imgs_raw.extend(chapter_path.glob(ext))

    if not imgs_raw:
        raise RuntimeError(f"章节目录没有图片：{chapter_path.resolve()}")

    bad = []
    ok: List[Tuple[int, Path]] = []
    for p in imgs_raw:
        try:
            page_no = extract_page_no_from_name(p.name)
            ok.append((page_no, p))
        except Exception as e:
            bad.append(str(e))

    if bad:
        out_dir = Path("out") / chapter_name
        out_dir.mkdir(parents=True, exist_ok=True)
        err_path = out_dir / "errors_pages.txt"
        err_path.write_text("\n".join(bad), encoding="utf-8")
        raise RuntimeError(f"发现页码格式异常，已写入：{err_path.resolve()}（请修正文件名后重跑）")

    # 页码合格后按页码排序
    imgs = [p for _, p in sorted(ok, key=lambda x: (x[0], x[1].name))]

    out_dir = Path("out") / chapter_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = out_dir / f"{chapter_name}.json"

    print(f"[阶段] 1/5 开始识别章节：{chapter_name}，页数={len(imgs)}", flush=True)

    # 并发识别：得到每张图的 page 输出
    pages_by_image = recognize_pages_concurrent(imgs, max_workers=workers, max_retries=max_retries)

    # 按页码顺序做归一化、挂 meta、跨页继承
    pages: List[dict] = []
    last_section: Optional[Tuple[Optional[int], Optional[str]]] = None

    print("[阶段] 3/5 归一化/跨页继承/组装 pages...", flush=True)

    for img in imgs:
        page_no = extract_page_no_from_name(img.name)
        page = pages_by_image.get(img.name)
        if page is None:
            raise RuntimeError(f"缺少识别结果：{img.name}")

        page = normalize_page_schema(page)

        page["_meta"] = {"page_no": page_no, "image": img.name}

        page, last_section = infer_section_if_missing(page, last_section)
        pages.append(page)

    print("[阶段] 4/5 拉平题目并生成 uid...", flush=True)
    flat_questions = flatten_questions(chapter_name, pages)

    chapter_errors = []
    if any("SPLIT_UNCERTAIN" in (p.get("errors") or []) for p in pages):
        chapter_errors.append("SPLIT_UNCERTAIN")
    if any(q.get("question_number") is None for q in flat_questions):
        chapter_errors.append("HAS_MISSING_QNO")

    out = {
        "chapter_name": chapter_name,
        "source_dir": str(chapter_path.as_posix()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": VISION_MODEL,
        "pages": pages,
        "questions": flat_questions,
        "errors": chapter_errors,
    }

    print(f"[阶段] 5/5 写入总 JSON：{out_json_path}", flush=True)
    out_json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("完成", flush=True)
    print(f"输出文件：{out_json_path.resolve()}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("chapter_dir", help="图片文件夹路径，例如 data\\h21")
    parser.add_argument("--workers", type=int, default=5, help="并发数（建议 3~8，默认 5）")
    parser.add_argument("--retries", type=int, default=3, help="单页最大重试次数（默认 3）")
    args = parser.parse_args()

    main(args.chapter_dir, workers=args.workers, max_retries=args.retries)