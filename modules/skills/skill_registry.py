"""
技能注册器模块
==============
统一管理技能的注册、意图匹配、上下文构建。

匹配策略 —— 三级路由（成本由低到高，逐层兜底，命中即停）：

    L1 关键词路由（零外部依赖、零成本、可复现）
        score >= skill_high_threshold(0.6)  → 判定命中，结束路由
        score <  skill_low_threshold(0.1)   → 判定「完全无关」，直接走通用对话
        卡在两者之间（模糊地带）            → 下沉到 L2

    L2 向量语义路由（1 次 embedding，能识别同义转述）
        把 query 与 L1 候选前 N 个技能的语义画像算余弦相似度
        最高相似度 >= skill_vector_threshold(0.75) → 判定命中，结束路由
        否则                                       → 下沉到 L3

    L3 LLM 兜底路由（1 次 LLM 调用，理解力最强）
        只把 query + L1 候选前 N 个技能清单丢给模型做分类
        模型输出 {skill_name, confidence}，校验「名字必须真实存在」
        confidence >= skill_llm_confidence(0.5) → 判定命中
        否则 → 未命中，走通用助手流程

为什么要分层：高频简单问句由 L1 白嫖搞定（不花一分钱）；换种说法的交给 L2；
真正歧义/复杂的才动用 L3。整体成本与延迟远低于「每次都问 LLM」。

设计原则：
- 技能系统全程可降级：任一层异常都只降级到下一层或放弃，绝不影响 ReAct 主链路；
- 技能只做「软引导」：注入 SOP 与推荐工具，不硬屏蔽其他工具，避免误匹配导致无法作答。
"""
from config import Settings


from typing import Dict, List, Optional, Tuple
from modules.common.logger import get_logger
from modules.common.exceptions import handle_exception
from modules.skills.base_skill import BaseSkill, SkillMatch, ROUTE_LABELS

_logger = get_logger("skill_registry")


# 【什么】两个向量的余弦相似度
# 【作用】L2 向量路由的计分函数；维度不一致或零向量返回 0.0，不抛异常打断路由
def _cosine(vec_a: List[float], vec_b: List[float]) -> float:
    """
    余弦相似度：算两个向量的夹角，返回 -1~1（本项目 embedding 归一化后实际落在 0~1）。

    为什么用余弦而不是欧氏距离：余弦只看「方向」不看「长度」，
    长句和短句只要语义接近就能拿到高分，正是语义匹配需要的性质。

    维度不一致 / 零向量 → 直接返回 0.0（安全降级，绝不抛异常打断路由）。
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    try:
        import numpy as np
        va = np.asarray(vec_a, dtype=float)
        vb = np.asarray(vec_b, dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except Exception:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        na = sum(a * a for a in vec_a) ** 0.5
        nb = sum(b * b for b in vec_b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


# 【什么】技能注册器（本地内存注册表）
# 【作用】统一管理技能注册与三级路由匹配，对上层只暴露 match / build_skill_context
class SkillRegistry:
    """技能注册器（本地内存注册表）。"""

    # 【什么】初始化三级路由的全部阈值
    # 【作用】显式传参优先，未传的从 Settings 读同名配置；阈值在此定型后运行期不再变
    def __init__(
        self,
        match_threshold: float = 0.35,        # 旧版统一匹配阈值，保留给外部直接使用
        high_threshold: Optional[float] = None,   # L1 高阈值：达到即判定命中，默认取 skill_high_threshold(0.6)
        low_threshold: Optional[float] = None,    # L1 低阈值：低于它判定与技能无关，默认 skill_low_threshold(0.1)
        vector_threshold: Optional[float] = None, # L2 判定阈值：余弦相似度达标即命中，默认 skill_vector_threshold(0.75)
        vector_top_n: Optional[int] = None,       # L2 参与比对的候选个数，控制 embedding 次数
        llm_route_enabled: Optional[bool] = None, # L3 总开关，关掉则最多走到 L2
        llm_top_n: Optional[int] = None,          # L3 丢给模型的候选个数，控制 prompt 长度
        llm_confidence: Optional[float] = None,   # L3 置信度门槛，低于它按未命中处理
    ):
        self._skills: Dict[str, BaseSkill] = {}
        self._threshold = match_threshold

        from config import get_settings
        s: Settings = get_settings()
        self._high = high_threshold if high_threshold is not None else s.skill_high_threshold
        self._low = low_threshold if low_threshold is not None else s.skill_low_threshold
        self._vec_threshold = (
            vector_threshold if vector_threshold is not None else s.skill_vector_threshold
        )
        self._vec_top_n = vector_top_n if vector_top_n is not None else s.skill_vector_top_n
        self._llm_enabled = (
            llm_route_enabled if llm_route_enabled is not None else s.skill_llm_route_enabled
        )
        self._llm_top_n = llm_top_n if llm_top_n is not None else s.skill_llm_top_n
        self._llm_confidence = (
            llm_confidence if llm_confidence is not None else s.skill_llm_confidence
        )

        self._vec_cache: Dict[str, List[float]] = {}   # {技能名: 语义向量}，首次参与比对时算一次后复用
        self._vector_available = True                  # L2 可用性标记；一旦失败置 False 并永久降级

        _logger.info(
            f"技能三级路由就绪: L1关键词[低{self._low}, 高{self._high}] "
            f"| L2向量[阈值{self._vec_threshold}, top{self._vec_top_n}] "
            f"| L3 LLM[{'开' if self._llm_enabled else '关'}, top{self._llm_top_n}, "
            f"conf>={self._llm_confidence}]"
        )

    # 【什么】注册一个技能进注册表
    # 【作用】按 name 做字典键；重名时覆盖旧技能并清掉它的语义向量缓存
    def register(self, skill: BaseSkill):
        """注册技能，重名时覆盖并告警。"""
        if not skill.name:
            raise ValueError("技能缺少 name 标识")
        if skill.name in self._skills:
            _logger.warning(f"技能重复注册，覆盖: {skill.name}")
        self._skills[skill.name] = skill
        self._vec_cache.pop(skill.name, None)
        _logger.info(
            f"技能注册: {skill.name} | 触发词{len(skill.trigger_keywords)}个 "
            f"| 绑定工具={skill.bound_tools or '无'}"
        )

    # 【什么】批量注册技能
    # 【作用】逐个调用 register，让重名覆盖与日志在批量装配时同样生效
    def register_batch(self, skills: List[BaseSkill]):
        """批量注册技能。"""
        for skill in skills:
            self.register(skill)

    # 【什么】按名称取回技能实例
    # 【作用】未注册的名字返回 None，由调用方决定是否降级
    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """按名称获取技能。"""
        return self._skills.get(name)

    # 【什么】列出全部已注册技能
    # 【作用】给 CLI 展示、describe_all 与外部遍历使用
    def list_skills(self) -> List[BaseSkill]:
        """列出全部已注册技能。"""
        return list(self._skills.values())

    # 【什么】对所有技能做关键词打分，得到候选列表
    # 【作用】L1 的候选来源，也是 L2/L3 的输入；单个技能打分异常只跳过不中断
    def score_candidates(self, query: str) -> List[SkillMatch]:
        """对所有已注册技能做关键词打分，返回 score>0 的候选（按 score、priority 降序）。"""
        if not query or not self._skills:
            return []
        candidates: List[SkillMatch] = []
        for skill in self._skills.values():
            try:
                m = skill.match_score(query)
            except Exception as e:
                _logger.warning(f"技能[{skill.name}]匹配异常，跳过: {e}")
                continue
            if m.score > 0:
                candidates.append(m)
        candidates.sort(key=lambda m: (m.score, m.skill.priority), reverse=True)
        return candidates

    # 【什么】L1 关键词路由
    # 【作用】按最高分给出三档裁决：hit / miss / ambiguous，决定要不要往下走
    def _route_keyword(self, query: str) -> Tuple[str, List[SkillMatch]]:
        """
        L1 关键词路由：三档裁决，决定「命中 / 放弃 / 继续往下问」。

        Returns:
            (决策, 候选列表)，决策 ∈ {"hit", "miss", "ambiguous"}
              hit       —— 得分 >= 高阈值，证据充分，路由到此结束
              miss      —— 得分 <  低阈值，与所有技能无关，路由到此结束
              ambiguous —— 卡在中间（例如只蹭中一个单字弱词），交给 L2/L3 继续判
        """
        candidates = self.score_candidates(query)
        if not candidates:
            _logger.info(f"[L1关键词路由] 无任何技能命中，走通用助手流程: query='{query[:30]}'")
            return "miss", []
        best = candidates[0]
        if best.score >= self._high:
            _logger.info(
                f"技能命中[L1关键词]: {best.skill.name} | score={best.score} "
                f"| 命中词={best.hit_keywords} | 候选数={len(candidates)}"
            )
            return "hit", candidates
        if best.score < self._low:
            _logger.info(
                f"[L1关键词路由] 最高分{best.score} < 低阈值{self._low}，"
                f"判定与技能无关: {best.skill.name}"
            )
            return "miss", candidates
        _logger.info(
            f"[L1关键词路由] 最高分{best.score} 落在模糊地带"
            f"[{self._low}, {self._high})，下沉L2向量路由: {best.skill.name}"
        )
        return "ambiguous", candidates

    # 【什么】取技能的语义向量（带缓存）
    # 【作用】每个技能只在首次参与比对时编码一次，避免重复 embedding 开销
    def _skill_vectors(self, names: List[str]) -> Dict[str, List[float]]:
        """懒加载 + 缓存技能语义向量：每个技能只在首次参与比对时编码一次。"""
        missing = [n for n in names if n not in self._vec_cache]
        if missing:
            from modules.memory.embedding_service import get_embedding_service
            svc = get_embedding_service()
            texts = [self._skills[n].routing_text for n in missing]
            vectors = svc.embed_batch(texts)
            for name, vec in zip(missing, vectors):
                self._vec_cache[name] = vec
            _logger.info(f"[L2向量路由] 技能语义向量已编码并缓存: {missing}")
        return {n: self._vec_cache[n] for n in names if n in self._vec_cache}

    # 【什么】L2 向量语义路由
    # 【作用】识别换个说法的同义问句；相似度不过阈值返回 None，交给 L3 继续判
    def _route_vector(
        self,
        query: str,
        candidates: List[SkillMatch],
        query_emb: Optional[List[float]] = None,   # 上层已算好的问句向量，传入可省一次 embedding
    ) -> Optional[SkillMatch]:
        """
        L2 向量语义路由：识别「换了个说法」的问句（关键词表里没有，但语义是一回事）。

        坑（见设计文档第四节）：向量分不出「查账号」和「注销账号」这种动作相反的说法，
        所以它只负责「选出候选」，最终仍由阈值把关，宁可漏判也不误判。
        """
        if not self._vector_available or not candidates:
            return None
        cands = candidates[: self._vec_top_n]
        try:
            from modules.memory.embedding_service import get_embedding_service
            qv = query_emb if query_emb is not None else get_embedding_service().embed(query)
            vecs = self._skill_vectors([m.skill.name for m in cands])
            scored = [
                (_cosine(qv, vecs[m.skill.name]), m)
                for m in cands if m.skill.name in vecs
            ]
            if not scored:
                return None
            scored.sort(key=lambda x: x[0], reverse=True)
            _logger.info(
                "[L2向量路由] 相似度: "
                + ", ".join(f"{m.skill.name}={sim:.4f}" for sim, m in scored)
            )
            sim, best = scored[0]
            if sim >= self._vec_threshold:
                _logger.info(
                    f"技能命中[L2向量]: {best.skill.name} | sim={sim:.4f} >= {self._vec_threshold}"
                )
                return SkillMatch(
                    skill=best.skill,
                    score=round(float(sim), 4),
                    hit_keywords=list(best.hit_keywords),
                    route="vector",
                    vector_score=round(float(sim), 4),
                )
            _logger.info(f"[L2向量路由] 最高相似度{sim:.4f} < {self._vec_threshold}，下沉L3 LLM路由")
            return None
        except Exception as e:
            self._vector_available = False
            _logger.warning(f"L2向量路由不可用，已永久降级: {e}")
            return None

    _LLM_ROUTE_SYSTEM = (   # L3 的系统提示词：强制只输出 JSON，并约定 skill_name / confidence / reason 三字段
        "你是技能路由器。根据用户提问，从给定的候选技能中选出唯一最合适的一个。\n"
        "只输出 JSON，不要输出任何解释性文字。\n"
        "若用户提问不属于任何候选技能，就把 skill_name 填为 \"none\"。\n"
        '输出格式: {"skill_name": "<候选技能的name，或 none>", '
        '"confidence": <0到1之间的小数>, "reason": "<一句话理由>"}'
    )

    # 【什么】L3 LLM 兜底路由
    # 【作用】把 query 与候选清单交给模型分类，识别关键词和向量都判不出的复杂意图
    def _route_llm(self, query: str, candidates: List[SkillMatch]) -> Optional[SkillMatch]:
        """
        L3 LLM 兜底路由：把 query + 候选技能清单丢给模型做分类。

        两条硬约束（设计文档第四节）：
        1. 强制结构化 JSON 输出，不让模型自由发挥长文本；
        2. 必须校验返回的 skill_name 真实存在，防幻觉；置信度不够也按未命中处理。
        """
        if not self._llm_enabled or not candidates:
            return None
        cands = candidates[: self._llm_top_n]
        try:
            from modules.agent.llm import LLMClient
            llm = LLMClient()
        except Exception as e:
            self._llm_enabled = False
            _logger.warning(f"L3 LLM路由不可用（客户端初始化失败），已永久降级: {e}")
            return None

        lines = []
        for i, m in enumerate(cands, 1):
            sk = m.skill
            kws = "、".join(sk.trigger_keywords[:8])
            lines.append(
                f'{i}. name="{sk.name}" | 展示名: {sk.display_name or sk.name}\n'
                f"   说明: {sk.description}\n"
                f"   典型触发词: {kws}"
            )
        prompt = (
            f"## 用户提问\n{query}\n\n"
            f"## 候选技能（只能从中选一个，或选 none）\n" + "\n".join(lines)
        )
        try:
            # temperature=0.1：路由分类要的是稳定可复现，不是发散，所以取低温
            result = llm.invoke_structured(
                prompt, system_prompt=self._LLM_ROUTE_SYSTEM, temperature=0.1
            )
        except Exception as e:
            _logger.warning(f"L3 LLM路由调用失败，按未命中处理: {e}")
            return None
        if not isinstance(result, dict) or result.get("error"):
            _logger.warning(f"L3 LLM路由输出解析失败，按未命中处理: {str(result)[:200]}")
            return None

        name = str(result.get("skill_name", "") or "").strip()
        reason = str(result.get("reason", "") or "")
        try:
            confidence = float(result.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            confidence = 0.0

        if not name or name.lower() in ("none", "null", "无"):
            _logger.info(f"[L3 LLM路由] 模型判定无合适技能: reason={reason}")
            return None
        skill = self._skills.get(name)
        if skill is None:
            _logger.warning(
                f"[L3 LLM路由] 模型返回了未注册的技能名 '{name}'（幻觉），按未命中处理"
            )
            return None
        if confidence < self._llm_confidence:
            _logger.info(
                f"[L3 LLM路由] 置信度{confidence} < {self._llm_confidence}，按未命中处理: {name}"
            )
            return None
        _logger.info(f"技能命中[L3 LLM路由]: {name} | confidence={confidence} | reason={reason}")
        return SkillMatch(
            skill=skill,
            score=round(confidence, 4),
            hit_keywords=[],
            route="llm",
            confidence=round(confidence, 4),
            route_reason=reason,
        )

    # 【什么】三级路由主入口
    # 【作用】L1 → L2 → L3 逐层兜底，命中即停；三层都没判定命中返回 None 走通用流程
    def match(self, query: str, query_emb: Optional[List[float]] = None) -> Optional[SkillMatch]:
        """
        三级路由主入口：L1关键词 → L2向量语义 → L3 LLM兜底，命中即停。

        Args:
            query:     用户提问（应已过 sanitize_input 清洗）
            query_emb: 可选的问句向量。上层若已算过可传入，帮 L2 省一次 embedding 调用。

        Returns:
            SkillMatch（route 字段标明命中层级）；三层都未判定命中则返回 None（走通用流程）。
        """
        decision, candidates = self._route_keyword(query)
        if decision == "hit":
            return candidates[0]
        if decision == "miss":
            return None
        m = self._route_vector(query, candidates, query_emb)
        if m:
            return m
        m = self._route_llm(query, candidates)
        if m:
            return m
        _logger.info("技能未命中（L1/L2/L3 三层均未判定命中），走通用助手流程")
        return None

    # 【什么】一步完成「三级路由匹配 + 上下文构建」
    # 【作用】给 ReActAgent.prepare_state 用；异常被装饰器兜成 (None, "")，不打断主链路
    @handle_exception(default=(None, ""), log_msg="技能上下文构建异常")
    def build_skill_context(
        self, query: str, query_emb: Optional[List[float]] = None
    ) -> Tuple[Optional[BaseSkill], str]:
        """
        一步完成「三级路由匹配 + 上下文文本构建」，供 ReActAgent.prepare_state 调用。
        返回 (匹配到的技能, 注入Prompt的文本)；未命中时返回 (None, "")。
        """
        match = self.match(query, query_emb=query_emb)
        if not match:
            return None, ""
        _logger.info(
            f"技能上下文已构建: {match.skill.name} | "
            f"路由层={ROUTE_LABELS.get(match.route, match.route)}"
        )
        return match.skill, match.skill.format_context(match)

    # 【什么】生成全部技能的清单文本
    # 【作用】CLI / 调试查看注册了哪些技能；触发词超过 8 个时截断并用「…」示意
    def describe_all(self) -> str:
        """生成全部技能的清单文本（CLI / 调试用）。"""
        if not self._skills:
            return "（未注册任何技能）"
        lines = []
        for i, skill in enumerate(self._skills.values(), 1):
            tools = "、".join(skill.bound_tools) if skill.bound_tools else "无（走RAG/通用能力）"
            keywords = "、".join(skill.trigger_keywords[:8])
            if len(skill.trigger_keywords) > 8:
                keywords += " …"
            lines.append(
                f"{i}. [{skill.name}] {skill.display_name}\n"
                f"   说明: {skill.description}\n"
                f"   触发词: {keywords}\n"
                f"   绑定工具: {tools} | 优先级: {skill.priority}"
            )
        return "\n".join(lines)
