"""
多 Agent 协调模块
==================
在「主管派活」之外补上三种协作能力：横向对话、一致性与投票、冲突仲裁与重规划。

【为什么不能只靠 Supervisor-Worker】
    中心化派活解决了"谁做什么"，但没解决三件事：
        1. 执行者之间对不上口径   —— 都要等主管汇总时才发现，返工成本已付出
        2. 产出互相矛盾时听谁的   —— 缺少一个明确的收敛规则
        3. 一批人失败了怎么办     —— 现状是标注"部分失败"就收工
    这三层分别由 MessageBus / Consensus / Arbitrator + ReplanController 承担。

【收敛纪律：协作不能变成群聊】
    放开自由对话会退化成谁都在说话、没人收口。这里的三条硬约束：
        1. 消息只在同一次运行内有效：harness 每次 run 开始调用 bus.clear()
        2. 总线有容量上限（默认 200 条），超出丢最旧的
        3. objection 只作为仲裁的输入，不触发新的执行轮次
    这样协作是"有向、有界、有终"的，token 与耗时都可预期。
"""
from __future__ import annotations

import re
from typing import Optional

from modules.common.logger import get_logger
from modules.multiagent.models import (
    AgentMessage,
    AgentOutput,
    ConsensusResult,
    MessageKind,
    PlanResult,
    SubTask,
)

_logger = get_logger("coordination")

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 【什么】把一段文本切成字符 2-gram 集合
# 【作用】相似度计算的最小单位。用字符而不是词，中文无需分词器。
#
#   【输入 → 输出对照】
#       输入: "营收1200万"
#       输出: {"营收", "收1", "12", "20", "00", "0万"}
#   【为什么用 2-gram 而不是整串比对】
#       整串相等判断太脆：多一个"同比+8%"就判成完全不同。
#       2-gram 能容忍语序与附加信息，只要主干一致就给高分。
# ---------------------------------------------------------------------------
def _grams(text: str) -> set[str]:
    t = _WS.sub("", text or "")
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def text_overlap(a: str, b: str) -> float:
    """两段文本的重叠度（Jaccard），0~1。

    【输入 → 输出对照】
        输入: a="营收1200万，同比+8%", b="营收1200万"   → 0.5000
              （B 的 6 个 2-gram 全在 A 里，交集 6 / 并集 12）
        输入: a="营收1200万",        b="营收800万"      → 0.3750
              （交集 {"营收","00","0万"} 共 3 / 并集 8）
        输入: a="营收1200万",        b=""               → 0.0
    """
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return round(len(ga & gb) / len(ga | gb), 4)


class MessageBus:
    """Agent 之间的消息存储与检索层。

    【存的是什么】
        存的是标准信封 Envelope（比 AgentMessage 多了 trace_id / hop / round /
        delivery 等字段，见 modules.orchestration.protocol）。
        因为 Envelope 继承自 AgentMessage，过去按 AgentMessage 读写的代码
        （`.content` / `.sender` / `.kind`）一行都不用改。

    【为什么在这里延迟导入 Envelope 而不是放文件头】
        Envelope 定义在 orchestration 包，而 orchestration 又依赖本包之外的
        multiagent.models。若放文件头，首次 import 顺序不同会撞上
        "模块部分初始化"的循环导入。函数内导入发生在运行时，
        那时所有模块都已就绪，两个方向 import 都不会失败。
    """

    def __init__(self, max_messages: int = 200):
        self._messages: list = []
        # 【为什么要有容量上限】
        #   协作消息若不限量，多轮 objection 互怼会把上下文撑爆，
        #   最后主管拿到的反而是一堆争吵记录而不是结论。
        self._max = max(10, int(max_messages))

    def post(
        self,
        sender: str,
        content: str,
        receiver: str = "*",
        kind: MessageKind = MessageKind.FINDING,
        ref_task_id: str = "",
        **kw,
    ):
        from modules.orchestration.protocol import Envelope

        env = Envelope(
            sender=sender, receiver=receiver, kind=kind,
            content=content, ref_task_id=ref_task_id, **kw,
        )
        return self.post_envelope(env)

    def post_envelope(self, env):
        """把已经过编排器裁决的信封放进总线。

        【为什么投递前要盖 DELIVERED 章】
            编排器判定放行时会先盖章；直接调用本方法（测试、离线构造）
            的信封还是 PENDING。不补这一下，统计"送到了几条"就会漏掉
            所有绕过编排器的消息，排查时以为消息系统坏了。
        """
        from modules.orchestration.protocol import DeliveryStatus

        if env.delivery == DeliveryStatus.PENDING:
            env.delivery = DeliveryStatus.DELIVERED
        self._messages.append(env)
        if len(self._messages) > self._max:
            self._messages = self._messages[-self._max:]
        _logger.debug(f"[{env.sender}] → {env.receiver}: {env.kind.value} {env.content[:40]}")
        return env

    def inbox(self, name: str, limit: int = 20) -> list:
        """取某人能看到的消息（点对点 + 广播），按时间正序。"""
        items = [m for m in self._messages if m.visible_to(name) and m.sender != name]
        return items[: max(1, int(limit))]

    def inbox_text(self, name: str, limit: int = 20) -> str:
        """把收件箱拼成可塞进 prompt 的文本。没人发消息时返回空串。"""
        msgs = self.inbox(name, limit)
        if not msgs:
            return ""
        return "\n".join(
            f"- [{m.kind.value}] {m.sender}: {m.content[:400]}" for m in msgs
        )

    def objections(self, limit: int = 50) -> list:
        return [m for m in self._messages if m.kind == MessageKind.OBJECTION][: max(1, int(limit))]

    def all(self, limit: int = 50) -> list:
        return self._messages[: max(1, int(limit))]

    def clear(self) -> None:
        """清空全部消息。

        【什么时候被调用】由 harness 在每次 run 开始时调用一次，
           因此消息的实际生命周期是"同一次运行内"，不跨运行。
        【为什么不跨运行保留】上一轮的 objection（谁和多数派不一致）
           留到下一轮，新一批 Agent 会看到与本次任务无关的批评，
           属于纯噪声。需要更细的批次级隔离时，调用方自行在批次间再调一次。
        """
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


class Consensus:
    """一致性与投票：判断多个产出是否打架，以及谁是多数派。"""

    def __init__(self, threshold: float = 0.35):
        # 【阈值 0.35 的来历】两条都陈述同一事实、只是详略不同的产出，
        #   2-gram 重叠通常在 0.5 以上；而数字打架（1200 万 vs 800 万）
        #   会掉到 0.3 附近。取 0.35 作为分界，宁可多判一次冲突去仲裁，
        #   也不放过真正的矛盾 —— 漏判的代价是把错误答案直接给用户。
        self.threshold = max(0.0, min(1.0, float(threshold)))

    # ------------------------------------------------------------------
    # 【一致性检测 + 聚类】
    #   聚类用贪心：取一个未归类的产出开新簇，把与簇代表相似度 >= 阈值的
    #   都收进来。O(n²)，n 是并行任务数（本项目个位数），完全够用。
    # ------------------------------------------------------------------
    def detect(self, outputs: list[AgentOutput]) -> ConsensusResult:
        usable = [o for o in outputs if o.success and (o.content or "").strip()]
        if len(usable) < 2:
            return ConsensusResult(
                has_conflict=False, similarity=1.0,
                clusters=[[o.agent] for o in usable],
                majority=[o.agent for o in usable],
                note="有效产出不足两个，无需一致性判定",
            )

        min_sim = 1.0
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                min_sim = min(min_sim, text_overlap(usable[i].content, usable[j].content))

        clusters: list[list[AgentOutput]] = []
        unassigned = list(usable)
        while unassigned:
            head = unassigned.pop(0)
            group = [head]
            rest = []
            for other in unassigned:
                if text_overlap(head.content, other.content) >= self.threshold:
                    group.append(other)
                else:
                    rest.append(other)
            clusters.append(group)
            unassigned = rest

        clusters.sort(key=len, reverse=True)
        majority = [o.agent for o in clusters[0]]
        has_conflict = min_sim < self.threshold
        # 【什么时候需要仲裁】
        #   两种情况：产出互相矛盾，或没有形成多数派（三个人或三类说法，
        #   最大簇也没过半）。后者同样说明结论没收敛，不能挑一个就用。
        needs_arbitration = has_conflict or len(clusters[0]) * 2 <= len(usable)

        return ConsensusResult(
            has_conflict=has_conflict,
            similarity=round(min_sim, 4),
            clusters=[[o.agent for o in g] for g in clusters],
            majority=majority,
            needs_arbitration=needs_arbitration,
            note=(
                f"最小相似度 {min_sim:.2f} {'<' if has_conflict else '>='} 阈值 "
                f"{self.threshold}；分为 {len(clusters)} 簇，多数派 {majority}"
            ),
        )

    def majority_content(self, outputs: list[AgentOutput], consensus: ConsensusResult) -> str:
        """取多数派的产出内容（用于仲裁失败时的兜底）。"""
        by_name = {o.agent: o for o in outputs}
        for name in consensus.majority:
            if name in by_name:
                return by_name[name].content
        return ""


class Arbitrator:
    """冲突仲裁：让审核者（或主管）在多个矛盾产出之间做裁决。"""

    ARBITRATION_PROMPT = (
        "## 用户问题\n{query}\n\n"
        "## 团队成员的分歧产出\n{outputs}\n\n"
        "## 一致性判定\n{note}\n\n"
        "## 你的任务\n"
        "- 判断哪些产出有依据、哪些没有\n"
        "- 给出一份裁决结论：采纳什么、不采纳什么、理由是什么\n"
        "- 无法判断谁对时，如实说明分歧点并给出保守结论\n"
        "- 不要简单复述各方内容，要给出你的判断\n\n"
        "只输出裁决结论正文，不要写标题。"
    )

    def __init__(self, judge=None):
        # judge 优先用 reviewer（它就是干这个的），没有就退到 supervisor
        self.judge = judge

    def arbitrate(self, query: str, outputs: list[AgentOutput],
                  consensus: ConsensusResult) -> Optional[AgentOutput]:
        if self.judge is None:
            return None
        blocks = [
            f"【{o.agent}】{o.content[:1200]}" if o.success else f"【{o.agent}】（失败）{o.error}"
            for o in outputs
        ]
        prompt = self.ARBITRATION_PROMPT.format(
            query=query, outputs="\n\n".join(blocks), note=consensus.note
        )
        try:
            content = self.judge.llm.invoke(
                prompt, system_prompt=getattr(self.judge, "system_prompt", ""),
                temperature=0.2,
            )
            return AgentOutput(
                agent=self.judge.name,
                role=getattr(self.judge.spec, "role", None) or outputs[0].role,
                task_id="",
                content=(content or "").strip(),
                success=True, steps=1,
            )
        except Exception as e:
            _logger.warning(f"仲裁调用失败（由多数派兜底）: {type(e).__name__}: {e}")
            return None


class ReplanController:
    """失败重规划：一批人没干成时，让主管重新拆一次。"""

    REPLAN_NOTE = (
        "## 执行现状\n"
        "- 已完成: {done}\n"
        "- 未完成: {failed}\n\n"
        "请只针对未完成的部分重新拆解任务，不要重复已完成的工作。"
    )

    def __init__(self, supervisor=None, max_replans: int = 1):
        self.supervisor = supervisor
        # 【为什么默认只重规划 1 次】
        #   重规划要再调一次 LLM，且失败往往是任务本身不可解
        #   （缺数据、工具不可用），重试多少次结果都一样。
        #   给 1 次机会处理"临时抖动"，再多就是烧钱。
        self.max_replans = max(0, int(max_replans))

    def need_replan(self, outputs: list[AgentOutput]) -> bool:
        return bool(outputs) and any(not o.success for o in outputs)

    # ------------------------------------------------------------------
    # 【重规划怎么做到不污染上下文】
    #   把执行现状临时拼进 ctx.query 调 plan，调完立刻还原。
    #   不还原的话，下一批 Agent 拿到的 query 里会带上"上一轮谁失败了"
    #   这类与它无关的信息，既不必要又占 token。
    # ------------------------------------------------------------------
    def build_replan(self, ctx, outputs: list[AgentOutput], roster) -> Optional[PlanResult]:
        if self.supervisor is None:
            return None
        done = [o.agent for o in outputs if o.success]
        failed = [f"{o.agent}: {o.error}" for o in outputs if not o.success]
        original = ctx.query
        try:
            ctx.query = (
                f"{original}\n\n"
                + self.REPLAN_NOTE.format(
                    done=", ".join(done) or "（无）",
                    failed="; ".join(failed) or "（无）",
                )
            )
            return self.supervisor.plan(ctx, roster)
        except Exception as e:
            _logger.warning(f"重规划失败（沿用原计划结果）: {type(e).__name__}: {e}")
            return None
        finally:
            ctx.query = original

    def fallback_task(self, outputs: list[AgentOutput], roster) -> Optional[SubTask]:
        """没有主管可用时的兜底：给失败任务换个执行者重试一次。"""
        failed = [o for o in outputs if not o.success]
        if not failed or not roster:
            return None
        busy = {o.agent for o in outputs if o.success}
        # 【为什么要把失败者自己也排除】
        #   重试的意义就是"换个人"。若还派给刚才失败的那个，
        #   大概率再失败一次（同样的工具、同样的权限、同样的数据缺口），
        #   白白多烧一轮 token。没人可换（全员都在忙或都失败过）时返回 None，
        #   由上层照实标注"该部分未能完成"。
        failed_names = {o.agent for o in failed}
        free = [a.name for a in roster if a.name not in busy and a.name not in failed_names]
        if not free:
            return None
        first = failed[0]
        return SubTask(
            id=f"{first.task_id}_retry" if first.task_id else "retry_1",
            assignee=free[0],
            instruction=f"接手未完成的任务（原执行者 {first.agent} 失败）：{first.error[:200]}",
        )
