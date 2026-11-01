/* ============================================================
   向量检索服务 · 知识库应用
   原生 JS，无构建步骤。三个页面：检索 / 知识库 / 索引运维。
   ============================================================ */
(function () {
  "use strict";

  var MODES = ["vector", "bm25", "hybrid"];
  var MODE_COLOR = { vector: "#5b9dff", bm25: "#f0a94c", hybrid: "#3fd6a8" };
  var MODE_DESC = {
    vector: "稠密向量检索：语义相近但用词不同也能命中",
    bm25: "BM25 稀疏检索：精确词匹配，且不调用模型服务",
    hybrid: "两路召回 + RRF 融合：无法预知查询类型时更稳"
  };

  var $ = function (id) { return document.getElementById(id); };
  var viewEl = $("view"), titleEl = $("pageTitle"), subEl = $("pageSub"), actionsEl = $("pageActions");

  /* ---------------- 基础工具 ---------------- */

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtMs(v) {
    if (v == null) return "—";
    if (v === 0) return "<1 ms";
    if (v < 10) return v.toFixed(1) + " ms";
    return Math.round(v) + " ms";
  }

  function fmtNum(v) {
    if (v == null) return "—";
    return Number(v).toLocaleString("zh-CN");
  }

  function fmtScore(v) {
    if (v == null) return "—";
    var a = Math.abs(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(4);
  }

  function toast(msg, kind) {
    var host = $("toastHost");
    var node = document.createElement("div");
    node.className = "toast" + (kind ? " " + kind : "");
    node.textContent = msg;
    host.appendChild(node);
    setTimeout(function () {
      node.style.opacity = "0";
      node.style.transition = "opacity .25s";
      setTimeout(function () { node.remove(); }, 260);
    }, kind === "err" ? 4600 : 2400);
  }

  /* ---------------- 管理密钥 ---------------- */
  // 只在「删除」和「管理」接口上用到，存在 localStorage，不会发给检索请求。
  var KEY_STORE = "vs-api-key";

  function adminKey() {
    try { return localStorage.getItem(KEY_STORE) || ""; } catch (e) { return ""; }
  }

  function setAdminKey(v) {
    try {
      if (v) localStorage.setItem(KEY_STORE, v);
      else localStorage.removeItem(KEY_STORE);
    } catch (e) { /* 隐私模式下写不了，忽略即可 */ }
  }

  async function api(path, opts) {
    opts = opts || {};
    var init = { method: opts.method || (opts.body !== undefined || opts.form ? "POST" : "GET") };
    var key = adminKey();
    if (opts.form) {
      init.body = opts.form;                       // FormData：不要手写 Content-Type
      if (key) init.headers = { "X-API-Key": key };
    } else if (opts.body !== undefined) {
      init.headers = { "Content-Type": "application/json" };
      if (key) init.headers["X-API-Key"] = key;
      init.body = JSON.stringify(opts.body);
    } else if (key) {
      init.headers = { "X-API-Key": key };
    }
    var res = await fetch(path, init);
    var text = await res.text();
    var data = {};
    try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { raw: text }; }
    if (res.status === 401) {
      throw new Error("需要管理密钥：请在「索引运维」页底部的「管理密钥」里填写");
    }
    if (!res.ok) {
      throw new Error(data.message || data.error || ("HTTP " + res.status));
    }
    return data;
  }

  function spinner(label) {
    return '<div class="empty"><span class="spin"></span> ' + esc(label || "加载中…") + "</div>";
  }

  /* ---------------- 侧栏统计 ---------------- */

  var statsState = { data: null };

  async function loadStats() {
    try {
      var s = await api("/api/stats");
      statsState.data = s;
      $("sideHealth").innerHTML = '<span class="dot' + (s.modelServiceHealthy ? "" : " bad") + '"></span>' +
        (s.modelServiceHealthy ? "模型服务正常" : "模型服务不可用");
      $("statVectors").textContent = fmtNum(s.indexedCount);
      $("statDocs").textContent = fmtNum(s.storedDocuments);
      $("statTerms").textContent = fmtNum((s.sparse || {}).terms);
      return s;
    } catch (e) {
      $("sideHealth").innerHTML = '<span class="dot bad"></span>服务未就绪';
      return null;
    }
  }

  /* ---------------- 检索页 ---------------- */

  var S = { mode: "hybrid", rerank: false, compare: false, busy: false, sources: [] };

  function renderSearch() {
    titleEl.textContent = "检索";
    subEl.textContent = "双路召回 → RRF 融合 → 精排";
    actionsEl.innerHTML = "";

    // 深链参数要在渲染前读，否则选项高亮和实际状态会对不上
    var params = new URLSearchParams(location.search);
    var m = params.get("mode");
    if (MODES.indexOf(m) >= 0) S.mode = m;
    if (params.get("rerank") === "1") S.rerank = true;
    if (params.get("compare") === "1") S.compare = true;
    // 来源多选也支持深链：?sources=a.md,b.md（逗号分隔，整体做 URL 编码）
    if (params.get("sources")) {
      S.sources = params.get("sources").split(",")
        .map(function (s) { return s.trim(); })
        .filter(function (s) { return s.length > 0; });
    }

    viewEl.innerHTML =
      '<div class="panel">' +
        '<div class="search-box">' +
          '<input id="q" class="input" type="text" placeholder="输入一个问题，或试试下面的示例……" autocomplete="off">' +
          '<button id="go" class="btn primary">检索</button>' +
        "</div>" +
        '<div class="controls">' +
          '<div class="group"><span class="group-label">模式</span>' +
            MODES.map(function (m) {
              return '<span class="chip" data-mode="' + m + '"' +
                (S.mode === m ? ' data-active="1"' : "") + ">" + m + "</span>";
            }).join("") +
          "</div>" +
          '<div class="group"><span class="group-label">选项</span>' +
            '<span class="chip" data-toggle="rerank"' + (S.rerank ? ' data-active="1"' : "") + ">精排</span>" +
            '<span class="chip" data-toggle="compare"' + (S.compare ? ' data-active="1"' : "") + ">三路对比</span>" +
          "</div>" +
          '<span class="hint" id="modeHint">' + esc(MODE_DESC[S.mode]) + "</span>" +
        "</div>" +
        '<div class="controls" id="sourceRow">' +
          '<div class="group"><span class="group-label">来源</span>' +
            '<span class="chip" data-source="" data-active="1">全部</span>' +
          "</div>" +
          '<span class="hint" id="sourceHint">按来源过滤会把条件下推到图遍历里</span>' +
        "</div>" +
        '<div class="examples">' +
          '<span class="hint">试试</span>' +
          [
            "为什么重启一次要等两分钟",
            "多个线程改相邻的变量为什么会互相拖慢",
            "两路搜索结果怎样合并才不会被高分的那路带偏",
            "efSearch 参数是干什么的",
            "长文本要切成多大的块比较合适"
          ].map(function (q) { return '<button class="example">' + esc(q) + "</button>"; }).join("") +
        "</div>" +
      "</div>" +
      '<div id="timingHost"></div>' +
      '<div id="out"></div>';

    var qEl = $("q");

    viewEl.querySelectorAll(".chip[data-mode]").forEach(function (c) {
      c.addEventListener("click", function () {
        S.mode = c.dataset.mode;
        viewEl.querySelectorAll(".chip[data-mode]").forEach(function (x) {
          x.dataset.active = x.dataset.mode === S.mode ? "1" : "0";
        });
        $("modeHint").textContent = MODE_DESC[S.mode];
      });
    });

    viewEl.querySelectorAll(".chip[data-toggle]").forEach(function (c) {
      c.addEventListener("click", function () {
        var name = c.dataset.toggle;
        S[name] = !S[name];
        c.dataset.active = S[name] ? "1" : "0";
      });
    });

    // 来源筛选：来源是动态的（取决于导入了哪些文件），所以从后端取。
    // 支持多选 —— 「全部」清空选择，单个来源点一下加入或移出。
    api("/api/documents/sources").then(function (counts) {
      counts = counts || {};
      var names = Object.keys(counts);
      var row = $("sourceRow");
      if (!row) return;
      if (names.length <= 1) {
        // 只有一种来源时筛不筛都一样，这一行没有存在意义
        row.style.display = "none";
        return;
      }
      var group = row.querySelector(".group");
      var allChip = group.querySelector(".chip");
      // 启动时 location.replace("#/search") 会让视图渲染两次，
      // 两个请求都会落到同一个（新的）节点上，导致胶囊被追加两遍。
      // 用节点上的标记做幂等，只保证「这一个节点只填一次」。
      if (group.dataset.populated === "1") {
        return;
      }
      group.dataset.populated = "1";
      var total = 0;
      names.forEach(function (k) { total += counts[k] || 0; });
      // 深链里写的来源可能已经不存在了（文件被删），丢掉它们，
      // 否则界面上一个都不高亮、请求却带着无效条件，看起来像「搜不到东西」
      S.sources = S.sources.filter(function (s) { return names.indexOf(s) >= 0; });

      function paint() {
        allChip.dataset.active = S.sources.length === 0 ? "1" : "0";
        group.querySelectorAll(".chip[data-source]").forEach(function (x) {
          var name = x.dataset.source;
          if (!name) return;
          x.dataset.active = S.sources.indexOf(name) >= 0 ? "1" : "0";
        });
        if (S.sources.length === 0) {
          $("sourceHint").textContent = "按来源过滤会把条件下推到图遍历里";
          return;
        }
        var matched = 0;
        S.sources.forEach(function (s) { matched += counts[s] || 0; });
        // 说清这一下发生了什么：不是「搜完再筛」，而是把条件下推进图遍历
        $("sourceHint").textContent =
          "只看 " + S.sources.length + " 个来源（" + matched + " 篇 / 共 " + total +
          " 篇），范围之外的候选不会占用名额";
      }

      allChip.addEventListener("click", function () {
        S.sources = [];
        paint();
      });

      names.forEach(function (name) {
        var chip = document.createElement("span");
        chip.className = "chip";
        chip.dataset.source = name;
        chip.textContent = name + " (" + counts[name] + ")";
        chip.addEventListener("click", function () {
          var at = S.sources.indexOf(name);
          if (at >= 0) {
            S.sources.splice(at, 1);
          } else {
            S.sources.push(name);
          }
          paint();
        });
        group.appendChild(chip);
      });
      paint();
    }).catch(function () {
      // 拿不到来源就保持「全部」，不影响检索本身
    });

    viewEl.querySelectorAll(".example").forEach(function (b) {
      b.addEventListener("click", function () { qEl.value = b.textContent; runSearch(); });
    });

    $("go").addEventListener("click", runSearch);
    qEl.addEventListener("keydown", function (e) {
      if (e.key !== "Enter") return;
      if (e.ctrlKey || e.metaKey) {
        S.compare = true;
        var chip = viewEl.querySelector('.chip[data-toggle="compare"]');
        if (chip) chip.dataset.active = "1";
      }
      runSearch();
    });
    qEl.focus();

    // 深链：?q=xxx 直接带查询打开
    if (params.get("q")) {
      qEl.value = params.get("q");
      runSearch();
    }
  }

  function renderTiming(timing) {
    var host = $("timingHost");
    if (!timing) { host.innerHTML = ""; return; }
    var embed = Math.max(0, timing.embeddingMs || 0);
    var search = Math.max(0, timing.searchMs || 0);
    var sum = embed + search || 1;
    var ep = (embed / sum) * 100, sp = (search / sum) * 100;

    var note;
    if (embed === 0) {
      note = "BM25 模式不调用模型服务，所以没有编码阶段 —— 端到端只花本地计算的时间。";
    } else if (embed > search * 3) {
      note = "瓶颈在<b>跨进程编码</b>而不是索引本身。这时候去调 HNSW 参数是南辕北辙。";
    } else if (search > embed) {
      note = "检索段占了主要开销 —— 很可能开着精排，CPU 上每个候选都要跑一次 cross-encoder 前向。";
    } else {
      note = "编码与检索开销相当。";
    }

    host.innerHTML =
      '<div class="panel">' +
        '<div class="timing-head"><div class="panel-title" style="margin:0">阶段耗时</div>' +
          '<span class="timing-total">端到端 ' + fmtMs(timing.totalMs) + "</span></div>" +
        '<div class="bar">' +
          (embed > 0 ? '<div class="seg-embed" style="width:' + ep.toFixed(2) + '%">' +
            (ep > 12 ? "编码 " + fmtMs(embed) : "") + "</div>" : "") +
          (search > 0 ? '<div class="seg-search" style="width:' + sp.toFixed(2) + '%">' +
            (sp > 12 ? "检索 " + fmtMs(search) : "") + "</div>" : "") +
        "</div>" +
        '<div class="timing-note">' + note + "</div>" +
      "</div>";
  }

  function renderHits(hits, color, soloIds) {
    if (!hits || !hits.length) {
      return '<div class="empty"><div class="big">没有命中</div>' +
        "知识库可能是空的 —— 去「知识库」页导入几篇文档。</div>";
    }
    var max = 0;
    hits.forEach(function (h) { max = Math.max(max, Math.abs(h.score || 0)); });
    if (!max) max = 1;

    return hits.map(function (h, i) {
      var w = Math.min(100, Math.max(2, (Math.abs(h.score || 0) / max) * 100));
      var badge = (soloIds && soloIds.has(h.id)) ? '<span class="badge only">唯一命中</span>' : "";
      return '<div class="hit">' +
        '<div class="rank">' + (i + 1) + "</div>" +
        '<div class="hit-body">' +
          '<div class="hit-top">' +
            '<span class="hid">#' + h.id + "</span>" +
            '<span class="score-wrap">' +
              '<span class="score-bar"><span class="score-fill" style="width:' + w.toFixed(1) +
                "%;background:" + color + '"></span></span>' +
              '<span class="score-num">' + fmtScore(h.score) + "</span>" +
            "</span>" + badge +
          "</div>" +
          '<div class="hit-text">' + esc(h.text || "（原文缺失）") + "</div>" +
        "</div>" +
      "</div>";
    }).join("");
  }

  function renderSingleResult(resp) {
    var timing = resp.timing || {};
    renderTiming(timing);
    var mode = resp.mode || S.mode;
    var flags = [];
    if (resp.reranked) flags.push("已精排");
    if (resp.indexType) flags.push(resp.indexType);

    $("out").innerHTML =
      '<div class="results-head"><h2>' + esc(mode) + " 结果</h2>" +
        '<span class="meta">top ' + (resp.hits ? resp.hits.length : 0) + " · " +
        fmtMs(timing.totalMs) + (flags.length ? " · " + esc(flags.join(" · ")) : "") + "</span></div>" +
      renderHits(resp.hits, MODE_COLOR[mode] || "#5b9dff", null);
  }

  function renderCompareResult(list) {
    renderTiming(null);
    var slowest = 0;
    var counts = new Map();
    list.forEach(function (r) {
      slowest = Math.max(slowest, (r.timing || {}).totalMs || 0);
      (r.hits || []).forEach(function (h) { counts.set(h.id, (counts.get(h.id) || 0) + 1); });
    });
    var soloIds = new Set();
    counts.forEach(function (n, id) { if (n === 1) soloIds.add(id); });

    var cols = list.map(function (r) {
      var mode = r.mode || "?";
      var color = MODE_COLOR[mode] || "#5b9dff";
      return '<div class="col">' +
        '<div class="col-head"><span class="name" style="color:' + color + '">' + esc(mode) + "</span>" +
          '<span class="kpi">' + fmtMs((r.timing || {}).totalMs) + " · " +
          (r.hits ? r.hits.length : 0) + " 条</span></div>" +
        renderHits(r.hits, color, soloIds) +
      "</div>";
    }).join("");

    $("out").innerHTML =
      '<div class="results-head"><h2>三路对比</h2>' +
        '<span class="meta">同一查询 · 最慢一路 ' + fmtMs(slowest) +
        " · 紫色标签表示只有这一路能命中</span></div>" +
      '<div class="cols">' + cols + "</div>";
  }

  async function runSearch() {
    var qEl = $("q");
    if (!qEl) return;
    var q = qEl.value.trim();
    if (!q) { qEl.focus(); return; }
    if (S.busy) return;

    S.busy = true;
    var go = $("go");
    go.disabled = true;
    go.textContent = "检索中…";
    $("out").innerHTML = spinner("检索中…");
    $("timingHost").innerHTML = "";

    try {
      if (S.compare) {
        var results = await Promise.all(MODES.map(function (m) {
          return api("/api/search", { body: { query: q, topK: 8, mode: m, rerank: S.rerank, sources: S.sources.length ? S.sources : null } });
        }));
        renderCompareResult(results);
      } else {
        var resp = await api("/api/search", {
          body: { query: q, topK: 10, mode: S.mode, rerank: S.rerank, sources: S.sources.length ? S.sources : null }
        });
        renderSingleResult(resp);
      }
    } catch (e) {
      $("out").innerHTML = '<div class="panel" style="border-color:#40212a">' +
        '<div class="panel-title" style="color:#ff6b6b;margin:0">检索失败</div>' +
        '<div class="hint" style="margin-top:8px">' + esc(e.message) + "</div></div>";
    } finally {
      S.busy = false;
      go.disabled = false;
      go.textContent = "检索";
    }
  }

  /* ---------------- 知识库页 ---------------- */

  var L = { offset: 0, limit: 15, q: "", total: 0, loading: false };

  function renderLibrary() {
    titleEl.textContent = "知识库";
    subEl.textContent = "导入文档，自动切分与向量化";
    actionsEl.innerHTML = "";

    viewEl.innerHTML =
      '<div class="panel">' +
        '<div class="drop" id="drop">' +
          '<div class="big">拖入文件，或点击选择</div>' +
          '<div class="small">支持 .md / .txt，自动按段落切块后灌入索引</div>' +
          '<input type="file" id="fileInput" accept=".md,.txt,.markdown,text/plain" style="display:none">' +
        "</div>" +
      "</div>" +
      '<div class="toolbar">' +
        '<input id="filter" class="input" placeholder="按字面过滤文档（不是语义检索）">' +
        '<span class="count" id="docCount">—</span>' +
        '<button class="btn small" id="refreshBtn">刷新</button>' +
      "</div>" +
      '<div id="docList"></div>' +
      '<div class="pager" id="pager"></div>';

    var drop = $("drop"), fileInput = $("fileInput");

    drop.addEventListener("click", function () { fileInput.click(); });
    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files[0]) uploadFile(fileInput.files[0]);
      fileInput.value = "";
    });
    ["dragenter", "dragover"].forEach(function (ev) {
      drop.addEventListener(ev, function (e) { e.preventDefault(); drop.dataset.over = "1"; });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      drop.addEventListener(ev, function (e) { e.preventDefault(); drop.dataset.over = "0"; });
    });
    drop.addEventListener("drop", function (e) {
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) uploadFile(f);
    });

    var filterTimer = null;
    $("filter").addEventListener("input", function (e) {
      clearTimeout(filterTimer);
      filterTimer = setTimeout(function () {
        L.q = e.target.value.trim();
        L.offset = 0;
        loadDocs();
      }, 220);
    });
    $("refreshBtn").addEventListener("click", function () { loadDocs(); });
    $("filter").value = L.q;

    loadDocs();
  }

  async function uploadFile(file) {
    var drop = $("drop");
    if (drop) {
      drop.dataset.over = "0";
      drop.innerHTML = '<div class="big"><span class="spin"></span> 导入中…</div>' +
        '<div class="small">' + esc(file.name) + " · " + fmtNum(file.size) + " 字节</div>";
    }

    var form = new FormData();
    form.append("file", file);

    try {
      var r = await api("/api/documents/upload", { form: form });
      toast("已导入 " + r.chunks + " 个分块（共 " + fmtNum(r.characters) + " 字符）", "ok");
      await loadStats();
      renderLibrary();
    } catch (e) {
      toast("导入失败：" + e.message, "err");
      renderLibrary();
    }
  }

  async function loadDocs() {
    if (L.loading) return;
    L.loading = true;
    var list = $("docList");
    if (list) list.innerHTML = spinner("加载文档…");

    try {
      var params = new URLSearchParams({ offset: L.offset, limit: L.limit });
      if (L.q) params.set("q", L.q);
      var page = await api("/api/documents?" + params.toString());
      L.total = page.total;
      renderDocs(page);
    } catch (e) {
      if (list) list.innerHTML = '<div class="empty">加载失败：' + esc(e.message) + "</div>";
    } finally {
      L.loading = false;
    }
  }

  function renderDocs(page) {
    var list = $("docList");
    if (!list) return;

    $("docCount").textContent = fmtNum(page.total) + " 篇" + (L.q ? "（已过滤）" : "");

    if (!page.items.length) {
      list.innerHTML = '<div class="empty"><div class="big">' +
        (L.q ? "没有匹配的文档" : "知识库还是空的") + "</div>" +
        (L.q ? "换个关键词，或清空过滤条件" : "拖动一个 .md 或 .txt 文件到上面，几秒钟就能搜它") +
        "</div>";
      $("pager").innerHTML = "";
      return;
    }

    list.innerHTML = page.items.map(function (d) {
      return '<div class="doc">' +
        '<div class="doc-id">#' + d.id + "</div>" +
        '<div class="doc-body">' +
          '<div class="doc-text">' + esc(d.text) + "</div>" +
          '<div class="doc-meta">' +
            (d.source ? '<span class="src">' + esc(d.source) + "</span> · " : "") +
            fmtNum((d.text || "").length) + " 字符</div>" +
        "</div>" +
        '<div class="doc-actions">' +
          '<button class="btn small danger" data-del="' + d.id + '">删除</button>' +
        "</div>" +
      "</div>";
    }).join("");

    list.querySelectorAll("[data-del]").forEach(function (btn) {
      btn.addEventListener("click", function () { deleteDoc(Number(btn.dataset.del)); });
    });

    var from = page.total === 0 ? 0 : page.offset + 1;
    var to = page.offset + page.items.length;
    var hasPrev = page.offset > 0;
    var hasNext = to < page.total;
    $("pager").innerHTML =
      '<button class="btn small" id="prevBtn"' + (hasPrev ? "" : " disabled") + ">上一页</button>" +
      '<span class="info">' + from + "–" + to + " / " + fmtNum(page.total) + "</span>" +
      '<button class="btn small" id="nextBtn"' + (hasNext ? "" : " disabled") + ">下一页</button>";

    if (hasPrev) $("prevBtn").addEventListener("click", function () {
      L.offset = Math.max(0, L.offset - L.limit); loadDocs();
    });
    if (hasNext) $("nextBtn").addEventListener("click", function () {
      L.offset += L.limit; loadDocs();
    });
  }

  async function deleteDoc(id) {
    if (!confirm("删除文档 #" + id + "？\n\n这是逻辑删除：检索时立刻不再返回它，" +
      "但索引里的向量要等重建才会真正回收。")) return;
    try {
      var r = await api("/api/documents/" + id, { method: "DELETE" });
      toast(r.affected ? "已删除 #" + id : "#" + id + " 不存在", r.affected ? "ok" : "err");
      await loadStats();
      loadDocs();
    } catch (e) {
      toast("删除失败：" + e.message, "err");
    }
  }

  /* ---------------- 运维页 ---------------- */

  function renderOps() {
    titleEl.textContent = "索引运维";
    subEl.textContent = "状态、落盘与重建";
    actionsEl.innerHTML = "";

    viewEl.innerHTML = spinner("读取索引状态…");
    refreshOps();
  }

  async function refreshOps() {
    var s = await loadStats();
    if (!s) {
      viewEl.innerHTML = '<div class="panel"><div class="panel-title" style="margin:0">服务未就绪</div>' +
        '<div class="hint" style="margin-top:8px">请先启动 Java 检索服务与 Python 模型服务。</div></div>';
      return;
    }

    var sparse = s.sparse || {};
    var hnsw = s.hnsw || {};
    var tomb = s.tombstones || 0;

    viewEl.innerHTML =
      '<div class="cards">' +
        card("向量数", fmtNum(s.indexedCount), "条") +
        card("存活文档", fmtNum(s.storedDocuments), "篇") +
        card("逻辑删除", fmtNum(tomb), "篇") +
        card("BM25 词表", fmtNum(sparse.terms), "项") +
        card("平均文档长度", sparse.averageDocumentLength || "—", "token") +
        card("图最大层", hnsw.maxLevel != null ? hnsw.maxLevel : "—", "层") +
        card("总边数", fmtNum(hnsw.totalLinks), "条") +
        card("快照恢复", s.restoredInMillis != null ? s.restoredInMillis : "—", "ms") +
      "</div>" +

      (tomb > 0 ? '<div class="panel" style="margin-top:16px;border-color:#4a3a1f">' +
        '<div class="panel-title" style="color:#f0a94c;margin:0">有 ' + tomb + ' 篇文档是逻辑删除</div>' +
        '<div class="hint" style="margin-top:8px">它们仍然留在索引里，检索时只是被过滤掉了，' +
        "还会把 BM25 的语料统计拉偏。点下面的「压实索引」可以真正清掉 —— " +
        "不需要重新导入，也不需要重新调用模型。</div></div>" : "") +

      '<div class="panel" style="margin-top:16px">' +
        '<div class="panel-title">操作</div>' +
        '<div class="actions-grid">' +
          '<button class="btn" id="snapshotBtn">保存快照</button>' +
          '<button class="btn" id="compactBtn">压实索引</button>' +
          '<button class="btn" id="switchBtn">' +
            (s.indexType === "brute-force" ? "切回 HNSW" : "切到暴力检索") + "</button>" +
          '<button class="btn" id="reloadStatsBtn">刷新状态</button>' +
        "</div>" +
        '<div class="hint" style="margin-top:12px">' +
          "保存快照会把向量索引与文档原文一起落盘，下次启动直接从快照恢复，不必重建。" +
        "</div>" +
      "</div>" +

      '<div class="panel danger-zone" style="margin-top:16px">' +
        '<div class="panel-title">危险操作</div>' +
        '<div class="actions-grid">' +
          '<button class="btn danger" id="clearBtn">清空知识库</button>' +
        "</div>" +
        '<div class="hint" style="margin-top:12px">' +
          "会删除全部向量与文档原文，不可恢复。演示数据可以随时重新导入。" +
        "</div>" +
      "</div>" +

      '<div class="panel" style="margin-top:16px">' +
        '<div class="panel-title">管理密钥</div>' +
        '<div class="hint">服务端配了 VECTOR_SEARCH_API_KEY 时，删除与管理操作需要它。' +
          "密钥只存在这台浏览器里，检索请求不会带上它。</div>" +
        '<div class="actions-grid" style="margin-top:12px">' +
          '<input class="input" id="apiKeyInput" type="password" autocomplete="off" ' +
            'placeholder="留空表示不使用密钥" value="' + esc(adminKey()) + '">' +
          '<button class="btn" id="saveKeyBtn">保存密钥</button>' +
          '<button class="btn ghost" id="clearKeyBtn">清除</button>' +
        "</div>" +
      "</div>" +

      '<div class="panel" style="margin-top:16px">' +
        '<div class="panel-title">运行信息</div>' +
        '<div class="kv">' +
          '<div class="k">索引实现</div><div class="v">' + esc(s.indexType) + "</div>" +
          '<div class="k">向量维度</div><div class="v">' + s.dimension + "</div>" +
          '<div class="k">HNSW 参数</div><div class="v">m=' + (hnsw.m != null ? hnsw.m : "—") +
            "  efConstruction=" + (hnsw.efConstruction != null ? hnsw.efConstruction : "—") + "</div>" +
          '<div class="k">原文总字符</div><div class="v">' + fmtNum(s.characterCount) + "</div>" +
          '<div class="k">模型服务</div><div class="v">' + (s.modelServiceHealthy ? "正常" : "不可用") + "</div>" +
          '<div class="k">启动恢复耗时</div><div class="v">' +
            (s.restoredInMillis != null ? s.restoredInMillis + " ms（未重建图）" : "—（本次未从快照恢复）") + "</div>" +
        "</div>" +
      "</div>";

    $("snapshotBtn").addEventListener("click", async function () {
      var btn = $("snapshotBtn");
      btn.disabled = true; btn.textContent = "保存中…";
      try {
        var r = await api("/api/admin/snapshot", { method: "POST" });
        toast("已落盘 " + r.vectorCount + " 条向量 · " +
          (r.indexBytes / 1024 / 1024).toFixed(2) + " MB · " + r.elapsedMillis + " ms", "ok");
        await refreshOps();
      } catch (e) {
        toast("保存失败：" + e.message, "err");
        btn.disabled = false; btn.textContent = "保存快照";
      }
    });

    $("compactBtn").addEventListener("click", async function () {
      var btn = $("compactBtn");
      btn.disabled = true; btn.textContent = "压实中…";
      try {
        var r = await api("/api/admin/compact", { method: "POST" });
        if (r.tombstonesPurged === 0) {
          toast("没有墓碑需要清理", "ok");
        } else {
          toast("已压实：向量 " + r.vectorsBefore + " → " + r.vectorsAfter +
            "（清掉 " + r.tombstonesPurged + " 个墓碑）· BM25 词表 " +
            r.termsBefore + " → " + r.termsAfter +
            " · 平均长度 " + r.averageDocumentLengthBefore.toFixed(1) + " → " +
            r.averageDocumentLengthAfter.toFixed(1) + " token · " + r.elapsedMillis + " ms", "ok");
        }
        await refreshOps();
      } catch (e) {
        toast("压实失败：" + e.message, "err");
        btn.disabled = false; btn.textContent = "压实索引";
      }
    });

    $("switchBtn").addEventListener("click", async function () {
      var target = s.indexType === "brute-force" ? "hnsw" : "brute-force";
      if (!confirm("切换到 " + target + "？\n\n索引会被清空并重建，需要重新导入文档。\n" +
        "（这个功能是给性能对照实验用的）")) return;
      try {
        await api("/api/admin/reset?type=" + target, { method: "POST" });
        toast("已切换到 " + target, "ok");
        await loadStats();
        refreshOps();
      } catch (e) {
        toast("切换失败：" + e.message, "err");
      }
    });

    $("reloadStatsBtn").addEventListener("click", function () { refreshOps(); });

    $("saveKeyBtn").addEventListener("click", function () {
      setAdminKey($("apiKeyInput").value.trim());
      toast(adminKey() ? "密钥已保存到本浏览器" : "已清空密钥", "ok");
      refreshOps();
    });

    $("clearKeyBtn").addEventListener("click", function () {
      setAdminKey("");
      toast("已清除本浏览器保存的密钥", "ok");
      refreshOps();
    });

    $("clearBtn").addEventListener("click", async function () {
      if (!confirm("清空全部 " + s.storedDocuments + " 篇文档？此操作不可恢复。")) return;
      try {
        await api("/api/documents", { method: "DELETE" });
        toast("知识库已清空", "ok");
        await loadStats();
        refreshOps();
      } catch (e) {
        toast("清空失败：" + e.message, "err");
      }
    });
  }

  function card(k, v, unit) {
    return '<div class="card"><div class="k">' + esc(k) + '</div><div class="v">' + esc(v) +
      (unit ? '<span class="u">' + esc(unit) + "</span>" : "") + "</div></div>";
  }

  /* ---------------- 路由 ---------------- */

  var ROUTES = {
    search: { title: "检索", render: renderSearch },
    library: { title: "知识库", render: renderLibrary },
    ops: { title: "索引运维", render: renderOps }
  };

  function currentRoute() {
    var hash = (location.hash || "").replace(/^#\/?/, "");
    var name = hash.split("?")[0];
    return ROUTES[name] ? name : "search";
  }

  function navigate() {
    var name = currentRoute();
    document.querySelectorAll(".nav a").forEach(function (a) {
      a.dataset.active = a.dataset.route === name ? "1" : "0";
    });
    ROUTES[name].render();
  }

  window.addEventListener("hashchange", navigate);

  /* ---------------- 启动 ---------------- */

  if (!location.hash) {
    location.replace("#/search");
  }
  loadStats();
  navigate();
})();
