const STATE = { files: [], currentIdx: 0, currentText: "", entities: [], editingIdx: -1 };

const $ = id => document.getElementById(id);

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const r = await fetch(path, opts);
  return r.json();
}

async function loadFileList() {
  STATE.files = await api("GET", "/api/files");
  const sel = $("fileSelect");
  sel.innerHTML = STATE.files.map((f, i) =>
    `<option value="${i}">${f}</option>`).join("");
  sel.value = "0";
}

async function loadFile(idx, saveCurrent) {
  if (saveCurrent && STATE.entities.length > 0) await saveGold();
  STATE.currentIdx = idx;
  STATE.entities = [];
  STATE.editingIdx = -1;
  const fid = STATE.files[idx].replace(".txt", "");
  const data = await api("GET", `/api/input/${fid}`);
  STATE.currentText = data.text;
  $("noteContent").textContent = data.text;

  const gold = await api("GET", `/api/gold/${fid}`);
  if (Array.isArray(gold) && gold.length > 0) {
    STATE.entities = gold;
  }
  $("fileSelect").value = String(idx);
  $("fileIndex").textContent = `${idx + 1} / ${STATE.files.length}`;
  clearForm();
  renderEntities();
  renderHighlights();
  status(`Loaded "${data.text.length}" chars, ${STATE.entities.length} entities`);
}

async function saveGold() {
  const fid = STATE.files[STATE.currentIdx].replace(".txt", "");
  const data = STATE.entities.map(e => ({
    text: e.text, type: e.type, position: e.position,
    assertions: e.assertions || [], candidates: e.candidates || [],
  }));
  const r = await api("PUT", `/api/gold/${fid}`, data);
  status(`Saved: ${r.count} entities`);
}

function status(msg) {
  $("status").textContent = msg;
  setTimeout(() => { if ($("status").textContent === msg) $("status").textContent = ""; }, 3000);
}

function getSelectedRange() {
  const sel = window.getSelection();
  if (!sel.rangeCount || sel.isCollapsed) return { text: "", start: -1, end: -1 };
  const range = sel.getRangeAt(0);
  return { text: range.toString().trim(), start: range.startOffset, end: range.endOffset };
}

function highlightText(node, start, end) {
  const range = document.createRange();
  let offset = 0;
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  let textNode, beforeNode;
  while (textNode = walker.nextNode()) {
    const len = textNode.textContent.length;
    if (offset <= start && offset + len >= end) {
      const s = start - offset, e = end - offset;
      beforeNode = textNode.splitText(s);
      beforeNode.splitText(e - s);
      range.setStart(beforeNode, 0);
      range.setEnd(beforeNode, e - s);
      break;
    }
    offset += len;
  }
  return range;
}

// Hack to get global offset from <pre> selection
$("noteContent").addEventListener("mouseup", () => {
  const sel = window.getSelection();
  if (!sel.rangeCount || sel.isCollapsed) return;
  const range = sel.getRangeAt(0);

  let offset = 0;
  const walker = document.createTreeWalker($("noteContent"), NodeFilter.SHOW_TEXT);
  let textNode;
  while (textNode = walker.nextNode()) {
    if (textNode === range.startContainer || textNode === range.startContainer.parentNode) break;
    offset += textNode.textContent.length;
  }
  if (range.startContainer.nodeType === Node.TEXT_NODE) {
    offset += range.startOffset;
  }

  const text = range.toString().trim();
  if (!text) return;

  const end = offset + range.toString().length;
  $("editText").value = text;
  $("editPos").textContent = `[${offset}, ${end}]`;
  STATE._selText = text;
  STATE._selStart = offset;
  STATE._selEnd = end;
});

function buildEntity() {
  const assertions = [];
  if ($("cbNegated").checked) assertions.push("isNegated");
  if ($("cbHistorical").checked) assertions.push("isHistorical");
  if ($("cbFamily").checked) assertions.push("isFamily");
  const candRaw = $("editCandidates").value.trim();
  const candidates = candRaw ? candRaw.split(",").map(s => s.trim()).filter(Boolean) : [];

  return {
    text: $("editText").value.trim(),
    type: $("editType").value,
    position: STATE._selStart >= 0 ? [STATE._selStart, STATE._selEnd] : [],
    assertions,
    candidates,
  };
}

function addEntity() {
  const e = buildEntity();
  if (!e.text || !e.type) { status("Missing text or type!"); return; }
  // check duplicate
  if (STATE.entities.some(x => x.text === e.text && x.type === e.type)) {
    status("Duplicate entity (same text+type)!");
    return;
  }
  STATE.entities.push(e);
  clearForm();
  renderEntities();
  renderHighlights();
}

function updateEntity() {
  if (STATE.editingIdx < 0) return;
  const e = buildEntity();
  if (!e.text || !e.type) { status("Missing text or type!"); return; }
  STATE.entities[STATE.editingIdx] = e;
  STATE.editingIdx = -1;
  clearForm();
  renderEntities();
  renderHighlights();
}

function deleteEntity() {
  if (STATE.editingIdx < 0) return;
  STATE.entities.splice(STATE.editingIdx, 1);
  STATE.editingIdx = -1;
  clearForm();
  renderEntities();
  renderHighlights();
}

function clearForm() {
  STATE.editingIdx = -1;
  STATE._selStart = -1;
  STATE._selEnd = -1;
  $("editText").value = "";
  $("editType").value = "";
  $("editPos").textContent = "-";
  $("cbNegated").checked = false;
  $("cbHistorical").checked = false;
  $("cbFamily").checked = false;
  $("editCandidates").value = "";
}

function editEntity(idx) {
  const e = STATE.entities[idx];
  STATE.editingIdx = idx;
  $("editText").value = e.text;
  $("editType").value = e.type;
  STATE._selStart = e.position ? e.position[0] : -1;
  STATE._selEnd = e.position ? e.position[1] : -1;
  $("editPos").textContent = e.position && e.position.length === 2 ? `[${e.position[0]}, ${e.position[1]}]` : "-";
  $("cbNegated").checked = (e.assertions || []).includes("isNegated");
  $("cbHistorical").checked = (e.assertions || []).includes("isHistorical");
  $("cbFamily").checked = (e.assertions || []).includes("isFamily");
  $("editCandidates").value = (e.candidates || []).join(", ");
  renderEntities();
}

function renderEntities() {
  const el = $("entityList");
  const colors = ["type-0", "type-1", "type-2", "type-3", "type-4"];
  const typeMap = { THUỐC: 0, CHẨN_ĐOÁN: 1, TRIỆU_CHỨNG: 2, TÊN_XÉT_NGHIỆM: 3, KẾT_QUẢ_XÉT_NGHIỆM: 4 };

  el.innerHTML = STATE.entities.map((e, i) => {
    const c = colors[typeMap[e.type] ?? 0];
    const sel = i === STATE.editingIdx ? " selected" : "";
    return `<div class="entity-row${sel}" data-idx="${i}">
      <span class="e-type ${c}">${e.type}</span>
      <span class="e-text" title="${e.text}">${e.text}</span>
      <span class="e-actions">
        <button class="e-edit" title="Edit">&#9998;</button>
        <button class="e-del" title="Delete">&#10005;</button>
      </span>
    </div>`;
  }).join("");

  $("listCount").textContent = STATE.entities.length;

  // bind events
  el.querySelectorAll(".entity-row").forEach(row => {
    row.addEventListener("click", () => {
      const idx = parseInt(row.dataset.idx);
      editEntity(idx);
    });
  });
  el.querySelectorAll(".e-del").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.closest(".entity-row").dataset.idx);
      STATE.entities.splice(idx, 1);
      if (STATE.editingIdx === idx) STATE.editingIdx = -1;
      clearForm();
      renderEntities();
      renderHighlights();
    });
  });
}

function renderHighlights() {
  let text = STATE.currentText;
  const spans = [];
  STATE.entities.forEach((e, i) => {
    if (!e.position || e.position.length !== 2) return;
    spans.push({ start: e.position[0], end: e.position[1], cls: i === STATE.editingIdx ? "active" : "highlight" });
  });
  spans.sort((a, b) => b.start - a.start || b.end - a.end);
  let html = text;
  spans.forEach(s => {
    const pre = html.slice(0, s.start);
    const mid = html.slice(s.start, s.end);
    const post = html.slice(s.end);
    html = pre + `<span class="${s.cls}">` + mid + `</span>` + post;
  });
  $("noteContent").innerHTML = html;
}

// Events
$("fileSelect").addEventListener("change", () => {
  loadFile(parseInt($("fileSelect").value), true);
});
$("btnPrev").addEventListener("click", () => {
  if (STATE.currentIdx > 0) loadFile(STATE.currentIdx - 1, true);
});
$("btnNext").addEventListener("click", () => {
  if (STATE.currentIdx < STATE.files.length - 1) loadFile(STATE.currentIdx + 1, true);
});
$("btnSave").addEventListener("click", () => saveGold());
$("btnExport").addEventListener("click", async () => {
  const data = await api("GET", "/api/export");
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gold_export.json";
  a.click();
  status("Exported all gold labels");
});
$("btnAdd").addEventListener("click", addEntity);
$("btnUpdate").addEventListener("click", updateEntity);
$("btnDelete").addEventListener("click", deleteEntity);
$("btnClear").addEventListener("click", clearForm);

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.ctrlKey && !e.metaKey) {
    e.preventDefault();
    if (STATE.editingIdx >= 0) updateEntity();
    else addEntity();
  }
  if (e.key === "Delete" && STATE.editingIdx >= 0) {
    e.preventDefault();
    deleteEntity();
  }
  if (e.ctrlKey && e.key === "ArrowLeft" && STATE.currentIdx > 0) {
    loadFile(STATE.currentIdx - 1, true);
  }
  if (e.ctrlKey && e.key === "ArrowRight" && STATE.currentIdx < STATE.files.length - 1) {
    loadFile(STATE.currentIdx + 1, true);
  }
});

// Init
(async () => {
  await loadFileList();
  await loadFile(0, false);
})();
