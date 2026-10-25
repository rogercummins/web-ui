"""Gradio tab builders for the two Agentic AI add-ons.

* :func:`build_api_config_tab` — a Cursor-style page to manage API provider
  profiles (keys, endpoints, models) backed by :class:`ProviderStore`.
* :func:`build_memory_tab` — a Codex-style local memory browser backed by the
  Hindsight :class:`HindsightStore` (projects + memories + search).

Both are built *inside* an existing ``gr.Blocks`` context and own their event
wiring. Each store is a module-level singleton so the tabs share one backend
across the app's lifetime. Every handler funnels backend errors through
``_err`` so a failure shows up as a red status line instead of a stack trace.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import gradio as gr

from src.config import ProviderStore, ProviderConfigError, SUPPORTED_PROVIDERS
from src.memory import HindsightStore, HindsightError
from src.memory.models import MEMORY_TYPES

logger = logging.getLogger(__name__)

# Shared singletons (local, file-backed).
_provider_store = ProviderStore()
_memory_store = HindsightStore()


def _ok(msg: str) -> str:
    return f"✅ {msg}"


def _err(msg: str) -> str:
    return f"❌ {msg}"


# ====================================================================== #
# API Configuration tab
# ====================================================================== #
def _provider_rows() -> Tuple[List[list], List[str]]:
    """Return (dataframe rows, parallel id list) for the provider table."""
    rows, ids = [], []
    for p in _provider_store.list_profiles(public=True):
        rows.append([
            p["label"], p["provider"],
            "on" if p["enabled"] else "off",
            p["default_model"] or (p["models"][0] if p["models"] else ""),
            len(p["models"]),
            p["api_key"] if p["has_key"] else "—",
        ])
        ids.append(p["id"])
    return rows, ids


def _split_models(text: str) -> List[str]:
    """Parse a models textbox: split on newlines and commas."""
    if not text:
        return []
    parts: List[str] = []
    for line in text.replace(",", "\n").splitlines():
        line = line.strip()
        if line:
            parts.append(line)
    return parts


def build_api_config_tab() -> None:
    gr.Markdown(
        "### 🔌 API Configuration\n"
        "Manage LLM provider profiles — keys, endpoints and model lists — "
        "like Cursor's model settings. Keys are stored locally and shown masked."
    )
    ids_state = gr.State([])
    selected_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=5):
            provider_table = gr.Dataframe(
                headers=["Label", "Provider", "Enabled", "Default model", "# Models", "API key"],
                datatype=["str", "str", "str", "str", "number", "str"],
                interactive=False,
                label="Configured providers",
                wrap=True,
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh")
                test_btn = gr.Button("🧪 Test selected")
                delete_btn = gr.Button("🗑️ Delete selected", variant="stop")
        with gr.Column(scale=4):
            gr.Markdown("#### Profile editor")
            f_label = gr.Textbox(label="Label", placeholder="e.g. My OpenAI")
            f_provider = gr.Dropdown(
                choices=list(SUPPORTED_PROVIDERS), value="openai", label="Provider type")
            f_key = gr.Textbox(label="API key", type="password",
                               placeholder="leave blank on update to keep existing key")
            f_base = gr.Textbox(label="Base URL (optional)",
                                placeholder="https://api.openai.com/v1")
            f_models = gr.Textbox(label="Models (one per line, or comma-separated)",
                                  lines=4, placeholder="gpt-4o\ngpt-4")
            f_default = gr.Textbox(label="Default model", placeholder="gpt-4o")
            f_enabled = gr.Checkbox(label="Enabled", value=True)
            with gr.Row():
                add_btn = gr.Button("➕ Add profile", variant="primary")
                update_btn = gr.Button("💾 Update selected")
                clear_btn = gr.Button("✨ Clear form")

    status = gr.Textbox(label="Status", interactive=False, lines=2)

    # ---- handlers ---------------------------------------------------- #
    def do_refresh():
        rows, ids = _provider_rows()
        return rows, ids, _ok(f"{len(ids)} provider profile(s).")

    def do_add(label, provider, key, base, models, default, enabled):
        try:
            _provider_store.add_profile(
                label=label, provider=provider, api_key=key or "",
                base_url=base or "", models=_split_models(models),
                default_model=default or "", enabled=bool(enabled))
            rows, ids = _provider_rows()
            return rows, ids, _ok(f"Added profile '{label}'.")
        except ProviderConfigError as exc:
            rows, ids = _provider_rows()
            return rows, ids, _err(str(exc))

    def do_update(selected, label, provider, key, base, models, default, enabled):
        if not selected:
            rows, ids = _provider_rows()
            return rows, ids, _err("Select a profile row first.")
        try:
            _provider_store.update_profile(
                selected, label=label, provider=provider, api_key=key or "",
                base_url=base or "", models=_split_models(models),
                default_model=default or "", enabled=bool(enabled))
            rows, ids = _provider_rows()
            return rows, ids, _ok("Profile updated.")
        except ProviderConfigError as exc:
            rows, ids = _provider_rows()
            return rows, ids, _err(str(exc))

    def do_delete(selected):
        if not selected:
            rows, ids = _provider_rows()
            return rows, ids, _err("Select a profile row first.")
        try:
            _provider_store.delete_profile(selected)
            rows, ids = _provider_rows()
            return rows, ids, _ok("Profile deleted.")
        except ProviderConfigError as exc:
            rows, ids = _provider_rows()
            return rows, ids, _err(str(exc))

    def do_test(selected):
        if not selected:
            return _err("Select a profile row first.")
        result = _provider_store.test_profile(selected, live=False)
        if result.get("ok"):
            return _ok(result.get("message", "client built successfully."))
        return _err(f"[{result.get('stage')}] {result.get('error')}")

    def on_select(ids, evt: gr.SelectData):
        # evt.index is [row, col]; map the row to a profile id.
        if not ids or evt.index is None:
            return (gr.update(),) * 7 + (None,)
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if row >= len(ids):
            return (gr.update(),) * 7 + (None,)
        pid = ids[row]
        try:
            p = _provider_store.get_profile(pid)
        except ProviderConfigError:
            return (gr.update(),) * 7 + (None,)
        return (
            gr.update(value=p.label),
            gr.update(value=p.provider),
            gr.update(value=""),  # never populate the key field
            gr.update(value=p.base_url),
            gr.update(value="\n".join(p.models)),
            gr.update(value=p.default_model),
            gr.update(value=p.enabled),
            pid,
        )

    def do_clear():
        return ("", "openai", "", "", "", "", True, None)

    form = [f_label, f_provider, f_key, f_base, f_models, f_default, f_enabled]

    refresh_btn.click(do_refresh, outputs=[provider_table, ids_state, status])
    add_btn.click(do_add, inputs=form, outputs=[provider_table, ids_state, status])
    update_btn.click(do_update, inputs=[selected_state] + form,
                     outputs=[provider_table, ids_state, status])
    delete_btn.click(do_delete, inputs=[selected_state],
                     outputs=[provider_table, ids_state, status])
    test_btn.click(do_test, inputs=[selected_state], outputs=[status])
    clear_btn.click(do_clear, outputs=form + [selected_state])
    provider_table.select(on_select, inputs=[ids_state], outputs=form + [selected_state])


# ====================================================================== #
# Hindsight memory tab
# ====================================================================== #
def _project_choices() -> List[Tuple[str, str]]:
    out = []
    try:
        for p in _memory_store.list_projects():
            out.append((f"{p.name}  ·  {p.id[-6:]}", p.id))
    except HindsightError as exc:
        logger.warning("listing projects failed: %s", exc)
    return out


def _memory_rows(project_id: str) -> Tuple[List[list], List[str]]:
    rows, ids = [], []
    if not project_id:
        return rows, ids
    try:
        for m in _memory_store.list_memories(project_id):
            rows.append([m.title, m.type, ", ".join(m.tags),
                         (m.content[:60] + "…") if len(m.content) > 60 else m.content])
            ids.append(m.id)
    except HindsightError as exc:
        logger.warning("listing memories failed: %s", exc)
    return rows, ids


def build_memory_tab() -> None:
    gr.Markdown(
        "### 🧠 Hindsight Memory\n"
        "A local, file-based memory system (Codex-style): organize **memories** "
        "into **projects**, search across them, all stored under `./tmp/hindsight`."
    )
    mem_ids_state = gr.State([])
    sel_mem_state = gr.State(None)

    with gr.Row():
        project_dd = gr.Dropdown(choices=_project_choices(), label="Project",
                                 interactive=True)
        new_proj_name = gr.Textbox(label="New project name", scale=2)
        new_proj_desc = gr.Textbox(label="Description", scale=2)
        create_proj_btn = gr.Button("➕ Create project", variant="primary")
        del_proj_btn = gr.Button("🗑️ Delete project", variant="stop")

    with gr.Row():
        with gr.Column(scale=5):
            mem_table = gr.Dataframe(
                headers=["Title", "Type", "Tags", "Preview"],
                datatype=["str", "str", "str", "str"],
                interactive=False, label="Memories", wrap=True)
            with gr.Row():
                search_box = gr.Textbox(label="Search (title / content / tags)", scale=3)
                search_btn = gr.Button("🔍 Search")
                refresh_mem_btn = gr.Button("🔄 Refresh")
        with gr.Column(scale=4):
            gr.Markdown("#### Memory editor")
            m_title = gr.Textbox(label="Title")
            m_type = gr.Dropdown(choices=list(MEMORY_TYPES), value="note", label="Type")
            m_tags = gr.Textbox(label="Tags (comma-separated)")
            m_content = gr.Textbox(label="Content", lines=6)
            with gr.Row():
                add_mem_btn = gr.Button("➕ Add memory", variant="primary")
                update_mem_btn = gr.Button("💾 Update selected")
                del_mem_btn = gr.Button("🗑️ Delete selected", variant="stop")

    mem_status = gr.Textbox(label="Status", interactive=False, lines=2)

    def _tags(text: str) -> List[str]:
        return [t.strip() for t in (text or "").split(",") if t.strip()]

    # ---- project handlers ------------------------------------------- #
    def do_create_project(name, desc):
        try:
            p = _memory_store.create_project(name=name or "", description=desc or "")
            return (gr.update(choices=_project_choices(), value=p.id),
                    _ok(f"Created project '{p.name}'."), "", "")
        except HindsightError as exc:
            return gr.update(choices=_project_choices()), _err(str(exc)), name, desc

    def do_delete_project(pid):
        if not pid:
            return gr.update(), [], [], _err("Pick a project first.")
        try:
            _memory_store.delete_project(pid)
            choices = _project_choices()
            return gr.update(choices=choices, value=None), [], [], _ok("Project deleted.")
        except HindsightError as exc:
            return gr.update(), [], [], _err(str(exc))

    def on_project_change(pid):
        rows, ids = _memory_rows(pid)
        return rows, ids, _ok(f"{len(ids)} memories.") if pid else ""

    # ---- memory handlers -------------------------------------------- #
    def do_add_memory(pid, title, mtype, tags, content):
        if not pid:
            return gr.update(), [], _err("Pick a project first.")
        try:
            _memory_store.add_memory(pid, title=title or "", content=content or "",
                                     type=mtype, tags=_tags(tags))
            rows, ids = _memory_rows(pid)
            return rows, ids, _ok("Memory added.")
        except HindsightError as exc:
            rows, ids = _memory_rows(pid)
            return rows, ids, _err(str(exc))

    def do_update_memory(pid, mid, title, mtype, tags, content):
        if not (pid and mid):
            return gr.update(), [], _err("Select a memory row first.")
        try:
            _memory_store.update_memory(pid, mid, title=title or "",
                                        content=content or "", type=mtype,
                                        tags=_tags(tags))
            rows, ids = _memory_rows(pid)
            return rows, ids, _ok("Memory updated.")
        except HindsightError as exc:
            rows, ids = _memory_rows(pid)
            return rows, ids, _err(str(exc))

    def do_delete_memory(pid, mid):
        if not (pid and mid):
            return gr.update(), [], _err("Select a memory row first.")
        try:
            _memory_store.delete_memory(pid, mid)
            rows, ids = _memory_rows(pid)
            return rows, ids, _ok("Memory deleted.")
        except HindsightError as exc:
            rows, ids = _memory_rows(pid)
            return rows, ids, _err(str(exc))

    def do_search(pid, query):
        try:
            results = _memory_store.search(query or "", project_id=pid or None)
            rows = [[m.title, m.type, ", ".join(m.tags),
                     (m.content[:60] + "…") if len(m.content) > 60 else m.content]
                    for m in results]
            ids = [m.id for m in results]
            return rows, ids, _ok(f"{len(ids)} match(es).")
        except HindsightError as exc:
            return gr.update(), [], _err(str(exc))

    def on_mem_select(pid, ids, evt: gr.SelectData):
        if not (pid and ids) or evt.index is None:
            return gr.update(), gr.update(), gr.update(), gr.update(), None
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if row >= len(ids):
            return gr.update(), gr.update(), gr.update(), gr.update(), None
        try:
            m = _memory_store.get_memory(pid, ids[row])
        except HindsightError:
            return gr.update(), gr.update(), gr.update(), gr.update(), None
        return (gr.update(value=m.title), gr.update(value=m.type),
                gr.update(value=", ".join(m.tags)), gr.update(value=m.content),
                m.id)

    create_proj_btn.click(do_create_project, inputs=[new_proj_name, new_proj_desc],
                          outputs=[project_dd, mem_status, new_proj_name, new_proj_desc])
    del_proj_btn.click(do_delete_project, inputs=[project_dd],
                       outputs=[project_dd, mem_table, mem_ids_state, mem_status])
    project_dd.change(on_project_change, inputs=[project_dd],
                      outputs=[mem_table, mem_ids_state, mem_status])
    refresh_mem_btn.click(on_project_change, inputs=[project_dd],
                          outputs=[mem_table, mem_ids_state, mem_status])
    add_mem_btn.click(do_add_memory,
                      inputs=[project_dd, m_title, m_type, m_tags, m_content],
                      outputs=[mem_table, mem_ids_state, mem_status])
    update_mem_btn.click(do_update_memory,
                         inputs=[project_dd, sel_mem_state, m_title, m_type, m_tags, m_content],
                         outputs=[mem_table, mem_ids_state, mem_status])
    del_mem_btn.click(do_delete_memory, inputs=[project_dd, sel_mem_state],
                      outputs=[mem_table, mem_ids_state, mem_status])
    search_btn.click(do_search, inputs=[project_dd, search_box],
                     outputs=[mem_table, mem_ids_state, mem_status])
    mem_table.select(on_mem_select, inputs=[project_dd, mem_ids_state],
                     outputs=[m_title, m_type, m_tags, m_content, sel_mem_state])
