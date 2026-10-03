$(function () {
    "use strict";

    const state = {
        apiKey: "",
        user: null,           
        currentSessionId: "",
        sessions: [],          
        isLoading: false,
        apiBase: "",           
    };

    const $apiKey = $("#apiKey");
    const $btnAuth = $("#btnAuth");
    const $connStatus = $("#connStatus");
    const $btnNewSession = $("#btnNewSession");
    const $sessionList = $("#sessionList");
    const $kbTotal = $("#kbTotal");
    const $kbCollection = $("#kbCollection");
    const $btnRefreshStats = $("#btnRefreshStats");
    const $fileInput = $("#fileInput");
    const $uploadStatus = $("#uploadStatus");
    const $userInfo = $("#userInfo");
    const $currentSession = $("#currentSession");
    const $statIterations = $("#statIterations");
    const $statToolCalls = $("#statToolCalls");
    const $statCompressed = $("#statCompressed");
    const $messageList = $("#messageList");
    const $queryInput = $("#queryInput");
    const $btnSend = $("#btnSend");
    const $loadingMask = $("#loadingMask");
    const $loadingText = $("#loadingText");
    const $toast = $("#toast");


    function showToast(msg, type) {
        type = type || "info";
        $toast.removeClass("success error warning").addClass(type).text(msg).removeClass("hidden");
        setTimeout(() => $toast.addClass("hidden"), 3000);
    }

    function setConnStatus(text, cls) {
        $connStatus.text("● " + text).removeClass("disconnected connecting connected").addClass(cls);
    }

    function setLoading(text) {
        if (text) {
            $loadingText.text(text);
            $loadingMask.removeClass("hidden");
            state.isLoading = true;
        } else {
            $loadingMask.addClass("hidden");
            state.isLoading = false;
        }
    }

    function escapeHtml(s) {
        if (!s) return "";
        return String(s)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    function renderMarkdown(text) {
        if (!text) return "";
        try {
            if (window.marked) {
                return marked.parse(text);
            }
        } catch (e) {  }
        return "<p>" + escapeHtml(text).replace(/\n/g, "<br>") + "</p>";
    }

    function scrollToBottom() {
        $messageList.stop().animate({ scrollTop: $messageList[0].scrollHeight }, 300);
    }


    function appendMessage(role, content, meta) {
        meta = meta || {};
        const avatar = role === "user" ? "👤" : "🤖";
        let metaHtml = "";
        if (meta.iterations || meta.tool_calls) {
            metaHtml = '<div class="message-meta">' +
                (meta.iterations !== undefined ? `<span class="meta-item">轮次: ${meta.iterations}</span>` : "") +
                (meta.tool_calls !== undefined ? `<span class="meta-item">工具: ${meta.tool_calls}</span>` : "") +
                (meta.summary ? `<span class="meta-item">摘要: ${escapeHtml(meta.summary)}</span>` : "") +
                "</div>";
        }

        let toolHtml = "";
        if (meta.tool_call_logs && meta.tool_call_logs.length) {
            toolHtml = meta.tool_call_logs.map(tc => {
                const statusIcon = tc.success ? "✓" : "✗";
                return `<div class="tool-call-box">
                    <div><span class="tool-name">${statusIcon} ${escapeHtml(tc.name)}</span>
                    <span class="tool-args">${escapeHtml(JSON.stringify(tc.arguments || {}))}</span></div>
                    <div class="tool-result">${escapeHtml(String(tc.result || "").slice(0, 200))}</div>
                </div>`;
            }).join("");
        }

        const contentHtml = role === "user"
            ? `<p>${escapeHtml(content)}</p>`
            : toolHtml + renderMarkdown(content);

        const html = `
        <div class="message ${role}">
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                ${contentHtml}
                ${metaHtml}
            </div>
        </div>`;

        $messageList.find(".welcome-banner").remove();
        $messageList.append(html);
        scrollToBottom();
    }


    function apiRequest(method, path, data, isFormData) {
        const opts = {
            url: state.apiBase + path,
            method: method,
            headers: { "X-API-Key": state.apiKey },
        };
        if (isFormData) {
            opts.data = data;
            opts.processData = false;
            opts.contentType = false;
        } else if (data) {
            opts.data = JSON.stringify(data);
            opts.contentType = "application/json";
        }
        return $.ajax(opts).catch(err => {
            const msg = (err.responseJSON && err.responseJSON.detail) || err.statusText || "请求失败";
            return $.Deferred().reject({ status: err.status, message: msg });
        });
    }


    $btnAuth.on("click", authenticate);
    $apiKey.on("keypress", e => { if (e.which === 13) authenticate(); });

    function authenticate() {
        const key = $apiKey.val().trim();
        if (!key) { showToast("请输入 API Key", "warning"); return; }
        state.apiKey = key;
        setConnStatus("连接中...", "connecting");

        $.ajax({ url: state.apiBase + "/health", method: "GET" })
            .then(() => apiRequest("GET", "/knowledge/stats"))
            .then(stats => {
                state.user = { user_id: "已认证", name: "用户", level: "normal" };
                if (key.includes("admin")) state.user.level = "admin";
                else if (key.includes("advanced")) state.user.level = "advanced";
                if (key.includes("normal")) state.user.name = "张三";
                else if (key.includes("advanced")) state.user.name = "李四";
                else if (key.includes("admin")) state.user.name = "管理员";

                setConnStatus("已连接", "connected");
                updateUserInfo();
                enableControls();
                refreshKbStats(stats);
                loadSessionList();          // 连上后恢复历史会话侧边栏（失败只提示，不影响主流程）
                showToast("连接成功", "success");
            })
            .catch(err => {
                setConnStatus("连接失败", "disconnected");
                showToast("连接失败: " + (err.message || "未知错误"), "error");
            });
    }

    function updateUserInfo() {
        if (!state.user) return;
        $userInfo.html(
            `<div class="user-name">${escapeHtml(state.user.name)}</div>
             <div class="user-level">等级: ${escapeHtml(state.user.level)}</div>`
        );
    }

    function enableControls() {
        $btnNewSession.prop("disabled", false);
        $btnRefreshStats.prop("disabled", false);
        $queryInput.prop("disabled", false);
        $btnSend.prop("disabled", false);
    }


    $btnRefreshStats.on("click", () => refreshKbStats());

    function refreshKbStats(stats) {
        if (!stats) {
            apiRequest("GET", "/knowledge/stats")
                .then(s => refreshKbStats(s))
                .catch(err => showToast("知识库统计失败: " + err.message, "error"));
            return;
        }
        $kbTotal.text(stats.total_chunks || stats.total || 0);
        $kbCollection.text(stats.collection || "-");
    }


    $fileInput.on("change", function () {
        const file = this.files[0];
        if (!file) return;
        $uploadStatus.removeClass("success error").text("上传中...").addClass("");

        const fd = new FormData();
        fd.append("file", file);

        apiRequest("POST", "/knowledge/ingest", fd, true)
            .then(resp => {
                $uploadStatus.addClass("success").text(`✓ 入库 ${resp.chunks} 个分片 (${escapeHtml(resp.file)})`);
                refreshKbStats();
                showToast("文档入库成功", "success");
            })
            .catch(err => {
                $uploadStatus.addClass("error").text("✗ " + (err.message || "上传失败"));
                showToast("入库失败: " + err.message, "error");
            });
        $(this).val("");
    });


    $btnNewSession.on("click", () => {
        state.currentSessionId = "";
        $currentSession.text("新会话（待创建）");
        $messageList.empty();
        appendMessage("ai", "已开启新会话，请输入你的问题。");
        $queryInput.focus();
        $sessionList.find("li").removeClass("active");
    });

    function addSessionToList(sessionId, title) {
        state.sessions.push({ session_id: sessionId, title: title });
        const $li = $(`<li data-id="${escapeHtml(sessionId)}" title="${escapeHtml(sessionId)}">${escapeHtml(title)}</li>`);
        $li.on("click", () => loadSession(sessionId));
        $sessionList.find(".empty-tip").remove();
        $sessionList.prepend($li);
    }

    // 【什么】从后端拉取当前用户的历史会话，重建侧边栏
    // 【作用】页面刷新/重新连接后会话列表不再丢失；只拉封面信息，点条目时才懒加载明细
    function loadSessionList() {
        return apiRequest("GET", "/sessions?limit=50")
            .then(resp => {
                state.sessions = [];
                $sessionList.empty();
                (resp.sessions || []).forEach(s => {
                    state.sessions.push({ session_id: s.session_id, title: s.title || "" });
                    // 标题为空（老数据）时兜底显示会话号前 12 位，总不能渲染一行空白
                    const title = s.title || s.session_id.slice(0, 12);
                    const $li = $(`<li data-id="${escapeHtml(s.session_id)}" title="${escapeHtml(s.session_id)}">${escapeHtml(title)}</li>`);
                    $li.on("click", () => loadSession(s.session_id));
                    $sessionList.append($li);
                });
                if (!state.sessions.length) {
                    $sessionList.html('<li class="empty-tip">暂无会话</li>');
                }
                // 若当前正停在某个会话上，恢复它的高亮
                if (state.currentSessionId) {
                    $sessionList.find(`li[data-id="${$.escapeSelector(state.currentSessionId)}"]`).addClass("active");
                }
            })
            .catch(err => {
                // 列表是辅助功能：加载失败不打断鉴权/对话主流程，只给提示
                showToast("会话列表加载失败: " + err.message, "error");
            });
    }

    function loadSession(sessionId) {
        state.currentSessionId = sessionId;
        $sessionList.find("li").removeClass("active");
        $sessionList.find(`li[data-id="${sessionId}"]`).addClass("active");
        $currentSession.text(sessionId.slice(0, 12) + "...");
        apiRequest("GET", `/history/${sessionId}?limit=20`)
            .then(resp => {
                $messageList.empty();
                if (resp.history && resp.history.length) {
                    resp.history.forEach(h => {
                        appendMessage("user", h.query);
                        appendMessage("ai", h.answer || "（无回答）", { summary: h.summary });
                    });
                } else {
                    appendMessage("ai", "该会话暂无历史记录，请开始对话。");
                }
            })
            .catch(err => showToast("加载历史失败: " + err.message, "error"));
    }


    $btnSend.on("click", sendMessage);
    $queryInput.on("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    $queryInput.on("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 120) + "px";
    });

    function sendMessage() {
        if (state.isLoading) { showToast("请等待当前请求完成", "warning"); return; }
        const query = $queryInput.val().trim();
        if (!query) return;
        if (!state.apiKey) { showToast("请先连接 API", "warning"); return; }

        appendMessage("user", query);
        $queryInput.val("").css("height", "auto");

        const aiMsg = createStreamingMessage();
        const $content = aiMsg.$content;
        const $meta = aiMsg.$meta;

        let fullAnswer = "";
        let toolCalls = [];
        let iterations = 0;
        let summary = "";

        setLoading("Agent 推理中...");

        const controller = new AbortController();
        fetch(state.apiBase + "/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-API-Key": state.apiKey,
            },
            body: JSON.stringify({
                query: query,
                session_id: state.currentSessionId || null,
            }),
            signal: controller.signal,
        })
        .then(async response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const events = buffer.split("\n\n");
                buffer = events.pop(); 

                for (const evt of events) {
                    const parsed = parseSSE(evt);
                    if (!parsed) continue;
                    handleSSEEvent(parsed.event, parsed.data);
                }
            }
        })
        .catch(err => {
            $content.html('<span style="color:var(--danger)">⚠️ 流式请求失败: ' + escapeHtml(err.message) + "</span>");
            showToast("对话失败: " + err.message, "error");
        })
        .finally(() => {
            setLoading(null);
        });

        function handleSSEEvent(event, data) {
            switch (event) {
                case "status":
                    $content.html('<span class="streaming-status">' + escapeHtml(data.message || "思考中...") + '<span class="cursor"></span></span>');
                    break;

                case "tool_start":
                    toolCalls.push({ name: data.name, arguments: data.arguments, success: true, result: "" });
                    renderToolCallsBox($content, toolCalls, true);
                    break;

                case "tool_end":
                    const idx = toolCalls.length - 1;
                    if (idx >= 0) {
                        toolCalls[idx].result = data.result;
                        toolCalls[idx].success = data.success;
                    }
                    renderToolCallsBox($content, toolCalls, false);
                    break;

                case "answer_start":
                    iterations = data.iterations || 0;
                    $content.html("");
                    fullAnswer = "";
                    break;

                case "answer_chunk":
                    fullAnswer += data.content || "";
                    $content.html(renderMarkdown(fullAnswer) + '<span class="cursor"></span>');
                    scrollToBottom();
                    break;

                case "answer_end":
                    summary = data.summary || "";
                    $content.html(renderMarkdown(fullAnswer));
                    $statIterations.text(iterations);
                    $statToolCalls.text(toolCalls.length);
                    $meta.html(
                        (iterations !== undefined ? `<span class="meta-item">轮次: ${iterations}</span>` : "") +
                        (toolCalls.length ? `<span class="meta-item">工具: ${toolCalls.length}</span>` : "") +
                        (summary ? `<span class="meta-item">摘要: ${escapeHtml(summary)}</span>` : "")
                    );
                    break;

                case "done":
                    if (data.session_id && data.session_id !== state.currentSessionId) {
                        const oldId = state.currentSessionId;
                        state.currentSessionId = data.session_id;
                        const sess = state.sessions.find(s => s.session_id === oldId);
                        if (sess) sess.session_id = data.session_id;
                        const $li = $sessionList.find(`li[data-id="${$.escapeSelector(oldId)}"]`);
                        if ($li.length) {
                            $li.attr("data-id", data.session_id).attr("title", data.session_id);
                            $li.off("click").on("click", () => loadSession(data.session_id));
                        }
                        $currentSession.text(data.session_id.slice(0, 12) + "...");
                    }
                    break;
            }
        }

        if (!state.currentSessionId) {
            const tmpId = "tmp-" + Date.now();
            state.currentSessionId = tmpId;
            $currentSession.text(tmpId.slice(0, 12) + "...");
            addSessionToList(tmpId, query.slice(0, 30));
        }
    }

    function createStreamingMessage() {
        $messageList.find(".welcome-banner").remove();
        const $msg = $(`
        <div class="message ai">
            <div class="message-avatar">🤖</div>
            <div class="message-content streaming">
                <div class="streaming-body"></div>
                <div class="message-meta"></div>
            </div>
        </div>`);
        $messageList.append($msg);
        scrollToBottom();
        return {
            $content: $msg.find(".streaming-body"),
            $meta: $msg.find(".message-meta"),
            $el: $msg,
        };
    }

    function parseSSE(raw) {
        if (!raw || !raw.trim()) return null;
        let event = "message";
        let dataStr = "";
        const lines = raw.split("\n");
        for (const line of lines) {
            if (line.startsWith("event:")) {
                event = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
                dataStr += line.slice(5).trim();
            }
        }
        if (!dataStr) return null;
        try {
            return { event, data: JSON.parse(dataStr) };
        } catch {
            return { event, data: { raw: dataStr } };
        }
    }

    function renderToolCallsBox($container, toolCalls, inProgress) {
        let html = "";
        for (const tc of toolCalls) {
            const statusIcon = tc.result === "" ? "⟳" : (tc.success ? "✓" : "✗");
            const spin = tc.result === "" ? " class=\"spinning\"" : "";
            html += `<div class="tool-call-box">
                <div><span class="tool-name"${spin}>${statusIcon} ${escapeHtml(tc.name)}</span>
                <span class="tool-args">${escapeHtml(JSON.stringify(tc.arguments || {}))}</span></div>
                ${tc.result !== "" ? `<div class="tool-result">${escapeHtml(String(tc.result).slice(0, 200))}</div>` : ""}
            </div>`;
        }
        $container.html(html);
        scrollToBottom();
    }


    function init() {
        if (!$apiKey.val()) {
            $apiKey.attr("placeholder", "演示: sk-user-normal-001");
        }
        setConnStatus("未连接", "disconnected");
        console.log("%c企业级 AI Agent 前端已加载", "color:#3b82f6;font-weight:bold;font-size:14px");
    }

    init();
});
