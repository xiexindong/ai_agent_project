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
from config import Settings  # 配置类：用作类型标注；三级路由各阈值的默认来源也是它


from typing import Dict, List, Optional, Tuple  # 类型注解工具：字典 / 列表 / 可选值 / 元组
from modules.common.logger import get_logger  # 统一日志器工厂：按名字获取模块级日志器
from modules.common.exceptions import handle_exception  # 异常兜底装饰器：把异常吞掉并返回默认返回值
from modules.skills.base_skill import BaseSkill, SkillMatch, ROUTE_LABELS  # 技能基类 / 匹配结果对象 / 路由层中文名映射表

_logger = get_logger("skill_registry")  # 创建本模块专属日志器，日志来源名固定为 skill_registry


# 【什么】两个向量的余弦相似度
# 【作用】L2 向量路由的计分函数；维度不一致或零向量返回 0.0，不抛异常打断路由
# 【为什么】L2 要识别「换个说法」的同义问句；余弦相似度只比方向不比长度，长短句也能公平比较
def _cosine(vec_a: List[float], vec_b: List[float]) -> float:  # 计算 vec_a 与 vec_b 的余弦相似度，返回 -1~1
    """
    余弦相似度：算两个向量的夹角，返回 -1~1（本项目 embedding 归一化后实际落在 0~1）。

    为什么用余弦而不是欧氏距离：余弦只看「方向」不看「长度」，
    长句和短句只要语义接近就能拿到高分，正是语义匹配需要的性质。

    维度不一致 / 零向量 → 直接返回 0.0（安全降级，绝不抛异常打断路由）。
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):  # 入参防御：空向量或维度对不齐无法计算
        return 0.0  # 按「完全不相似」返回，让路由安全降级而不是抛异常
    try:
        import numpy as np  # 延迟导入 numpy：只有真走到 L2 才加载，不拖累进程启动速度
        va = np.asarray(vec_a, dtype=float)  # 把问句向量转成 numpy 浮点数组
        vb = np.asarray(vec_b, dtype=float)  # 把技能向量转成 numpy 浮点数组
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))  # 分母 = 两个向量模长之积
        if denom == 0.0:  # 任一向量为零向量时模长积为 0，除法无意义
            return 0.0  # 同样按不相似处理，安全降级
        return float(np.dot(va, vb) / denom)  # 点积 ÷ 模长积 = 余弦相似度
    except Exception:
        # 【设计意图】numpy 缺失或计算异常时退回纯 Python 实现：宁可慢一点，绝不让路由层因依赖问题崩掉
        dot = sum(a * b for a, b in zip(vec_a, vec_b))  # 纯 Python 求两向量点积
        na = sum(a * a for a in vec_a) ** 0.5  # 纯 Python 求 vec_a 的模长
        nb = sum(b * b for b in vec_b) ** 0.5  # 纯 Python 求 vec_b 的模长
        if na == 0.0 or nb == 0.0:  # 任一向量模长为 0（零向量）
            return 0.0  # 按不相似处理
        return dot / (na * nb)  # 返回余弦相似度


# 【什么】技能注册器（本地内存注册表）
# 【作用】统一管理技能注册与三级路由匹配，对上层只暴露 match / build_skill_context
# 【为什么】集中收口注册与路由逻辑，上层无需关心三级路由的内部细节，只拿「命中结果 + 注入文本」
class SkillRegistry:  # 技能注册器类：本地内存注册表 + 三级路由
    """技能注册器（本地内存注册表）。"""

    # 【什么】初始化三级路由的全部阈值
    # 【作用】显式传参优先，未传的从 Settings 读同名配置；阈值在此定型后运行期不再变
    # 【为什么】阈值集中在构造时定型，保证一次请求内（乃至进程生命周期内）裁决口径一致，便于复现与调参
    def __init__(  # 构造函数：装配三级路由的全部参数
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
        self._skills: Dict[str, BaseSkill] = {}  # 注册表本体：{技能名: 技能实例}
        self._threshold = match_threshold  # 保存旧版统一阈值，兼容外部直接读取的场景

        from config import get_settings  # 延迟导入配置单例工厂，规避模块级循环依赖
        s: Settings = get_settings()  # 取全局配置单例
        self._high = high_threshold if high_threshold is not None else s.skill_high_threshold  # L1 高阈值：显式传参优先，否则读配置（默认0.6）
        self._low = low_threshold if low_threshold is not None else s.skill_low_threshold  # L1 低阈值：显式传参优先，否则读配置（默认0.1）
        self._vec_threshold = (  # L2 余弦相似度判定阈值
            vector_threshold if vector_threshold is not None else s.skill_vector_threshold  # 显式传参优先，否则读配置（默认0.75）
        )
        self._vec_top_n = vector_top_n if vector_top_n is not None else s.skill_vector_top_n  # L2 候选个数：显式传参优先，否则读配置
        self._llm_enabled = (  # L3 总开关
            llm_route_enabled if llm_route_enabled is not None else s.skill_llm_route_enabled  # 显式传参优先，否则读配置
        )
        self._llm_top_n = llm_top_n if llm_top_n is not None else s.skill_llm_top_n  # L3 候选个数：显式传参优先，否则读配置
        self._llm_confidence = (  # L3 置信度门槛
            llm_confidence if llm_confidence is not None else s.skill_llm_confidence  # 显式传参优先，否则读配置（默认0.5）
        )

        self._vec_cache: Dict[str, List[float]] = {}   # {技能名: 语义向量}，首次参与比对时算一次后复用
        self._vector_available = True                  # L2 可用性标记；一旦失败置 False 并永久降级

        _logger.info(  # 启动日志：打印三级路由的全部生效参数，便于排障时核对配置
            f"技能三级路由就绪: L1关键词[低{self._low}, 高{self._high}] "  # L1 两档阈值
            f"| L2向量[阈值{self._vec_threshold}, top{self._vec_top_n}] "  # L2 阈值与候选数
            f"| L3 LLM[{'开' if self._llm_enabled else '关'}, top{self._llm_top_n}, "  # L3 开关与候选数
            f"conf>={self._llm_confidence}]"  # L3 置信度门槛
        )

    # 【什么】注册一个技能进注册表
    # 【作用】按 name 做字典键；重名时覆盖旧技能并清掉它的语义向量缓存
    # 【为什么】name 是技能唯一身份；覆盖式注册让「后注册生效」语义简单可预期，清缓存防止旧向量污染 L2
    def register(self, skill: BaseSkill):  # 注册单个技能实例
        """注册技能，重名时覆盖并告警。"""
        if not skill.name:  # name 是注册表字典键，缺失则无法注册
            raise ValueError("技能缺少 name 标识")  # 直接抛错，让装配阶段就暴露配置问题
        if skill.name in self._skills:  # 检测到重名注册
            _logger.warning(f"技能重复注册，覆盖: {skill.name}")  # 告警提示覆盖行为，便于排查装配错误
        self._skills[skill.name] = skill  # 写入注册表（重名时新技能覆盖旧技能）
        self._vec_cache.pop(skill.name, None)  # 清掉同名技能的旧语义向量缓存，防止 L2 用到过期向量
        _logger.info(  # 注册成功日志
            f"技能注册: {skill.name} | 触发词{len(skill.trigger_keywords)}个 "  # 技能名与触发词数量
            f"| 绑定工具={skill.bound_tools or '无'}"  # 绑定工具列表，为空则显示「无」
        )

    # 【什么】批量注册技能
    # 【作用】逐个调用 register，让重名覆盖与日志在批量装配时同样生效
    # 【为什么】复用 register 的校验/覆盖/清缓存逻辑，避免批量路径与单个路径行为不一致
    def register_batch(self, skills: List[BaseSkill]):  # 批量注册一组技能实例
        """批量注册技能。"""
        for skill in skills:  # 遍历待注册的技能列表
            self.register(skill)  # 逐个走标准注册流程

    # 【什么】按名称取回技能实例
    # 【作用】未注册的名字返回 None，由调用方决定是否降级
    # 【为什么】用「返回 None」而不是抛异常表达「不存在」，让调用方自由处理未命中场景
    def get_skill(self, name: str) -> Optional[BaseSkill]:  # 按技能名查询
        """按名称获取技能。"""
        return self._skills.get(name)  # 字典 get：查不到返回 None

    # 【什么】列出全部已注册技能
    # 【作用】给 CLI 展示、describe_all 与外部遍历使用
    # 【为什么】注册表内部是字典，对外提供列表视图更安全，也符合「遍历」的使用习惯
    def list_skills(self) -> List[BaseSkill]:  # 返回全部技能实例的列表
        """列出全部已注册技能。"""
        return list(self._skills.values())  # 取字典 values 转列表返回

    # 【什么】对所有技能做关键词打分，得到候选列表
    # 【作用】L1 的候选来源，也是 L2/L3 的输入；单个技能打分异常只跳过不中断
    # 【为什么】候选统一在这里产出并排序，保证三层路由看到的都是同一份「按得分+优先级降序」的候选
    def score_candidates(self, query: str) -> List[SkillMatch]:  # 入参为用户问句，返回候选列表
        """对所有已注册技能做关键词打分，返回 score>0 的候选（按 score、priority 降序）。"""
        if not query or not self._skills:  # 空问句或未注册任何技能时没必要打分
            return []  # 返回空候选，后续路由直接按 miss 处理
        candidates: List[SkillMatch] = []  # 收集得分 > 0 的候选
        for skill in self._skills.values():  # 遍历全部已注册技能
            try:
                m = skill.match_score(query)  # 调用技能自身的关键词打分逻辑
            except Exception as e:
                # 【设计意图】单个技能打分异常只跳过该技能：绝不让一个坏技能拖垮整个路由
                _logger.warning(f"技能[{skill.name}]匹配异常，跳过: {e}")  # 记录异常后继续
                continue  # 跳过这个技能，继续打分其余技能
            if m.score > 0:  # 只保留有命中的候选，0 分技能不参与后续裁决
                candidates.append(m)  # 加入候选列表
        candidates.sort(key=lambda m: (m.score, m.skill.priority), reverse=True)  # 先按得分降序，同分再按优先级降序，保证 best 永远是 candidates[0]
        return candidates  # 返回排序后的候选列表

    # 【什么】L1 关键词路由
    # 【作用】按最高分给出三档裁决：hit / miss / ambiguous，决定要不要往下走
    # 【为什么】关键词匹配零成本且可复现，能当场解决的高频问句就在这里终结，不为它们多花一分钱
    def _route_keyword(self, query: str) -> Tuple[str, List[SkillMatch]]:  # 返回 (决策, 候选列表)
        """
        L1 关键词路由：三档裁决，决定「命中 / 放弃 / 继续往下问」。

        Returns:
            (决策, 候选列表)，决策 ∈ {"hit", "miss", "ambiguous"}
              hit       —— 得分 >= 高阈值，证据充分，路由到此结束
              miss      —— 得分 <  低阈值，与所有技能无关，路由到此结束
              ambiguous —— 卡在中间（例如只蹭中一个单字弱词），交给 L2/L3 继续判
        """
        candidates = self.score_candidates(query)  # 先拿到按得分+优先级降序的候选列表
        if not candidates:  # 一个候选都没有
            _logger.info(f"[L1关键词路由] 无任何技能命中，走通用助手流程: query='{query[:30]}'")  # 记录截断后的问句，防日志过长
            return "miss", []  # 判定 miss，路由结束
        best = candidates[0]  # 最高分候选（列表已排序）
        if best.score >= self._high:  # 达到高阈值，证据充分
            _logger.info(  # 命中日志
                f"技能命中[L1关键词]: {best.skill.name} | score={best.score} "  # 技能名与得分
                f"| 命中词={best.hit_keywords} | 候选数={len(candidates)}"  # 命中词明细与候选总数
            )
            return "hit", candidates  # 判定 hit，路由结束
        if best.score < self._low:  # 连低阈值都没够到，视为与技能无关
            _logger.info(  # 记录裁决依据
                f"[L1关键词路由] 最高分{best.score} < 低阈值{self._low}，"
                f"判定与技能无关: {best.skill.name}"
            )
            return "miss", candidates  # 判定 miss（但保留候选返回，便于上层排查）
        _logger.info(  # 落在模糊地带，记录后下沉
            f"[L1关键词路由] 最高分{best.score} 落在模糊地带"
            f"[{self._low}, {self._high})，下沉L2向量路由: {best.skill.name}"
        )
        return "ambiguous", candidates  # 判定 ambiguous，交给 L2/L3 继续判

    # 【什么】取技能的语义向量（带缓存）
    # 【作用】每个技能只在首次参与比对时编码一次，避免重复 embedding 开销
    # 【为什么】embedding 调用有成本（时间/费用），技能画像文本运行期不变，算一次缓存复用是最优解
    def _skill_vectors(self, names: List[str]) -> Dict[str, List[float]]:  # 入参为候选技能名列表
        """懒加载 + 缓存技能语义向量：每个技能只在首次参与比对时编码一次。"""
        missing = [n for n in names if n not in self._vec_cache]  # 筛出还没缓存向量的技能名
        if missing:  # 有缺口才需要调用 embedding 服务
            from modules.memory.embedding_service import get_embedding_service  # 延迟导入 embedding 服务工厂，避免循环依赖与无谓初始化
            svc = get_embedding_service()  # 获取 embedding 服务实例
            texts = [self._skills[n].routing_text for n in missing]  # 取每个技能的「语义画像」文本
            vectors = svc.embed_batch(texts)  # 批量编码：一次调用搞定全部缺口，比逐个编码省开销
            for name, vec in zip(missing, vectors):  # 把技能名与向量一一对应
                self._vec_cache[name] = vec  # 写入缓存，后续比对直接复用
            _logger.info(f"[L2向量路由] 技能语义向量已编码并缓存: {missing}")  # 记录本次编码了哪些技能
        return {n: self._vec_cache[n] for n in names if n in self._vec_cache}  # 只返回成功拿到向量的技能（编码失败的自动缺席）

    # 【什么】L2 向量语义路由
    # 【作用】识别换个说法的同义问句；相似度不过阈值返回 None，交给 L3 继续判
    # 【为什么】关键词表覆盖不全所有说法，向量语义能补上「词不同但意思相同」的盲区，且成本远低于 LLM
    def _route_vector(  # L2 向量语义路由
        self,
        query: str,
        candidates: List[SkillMatch],
        query_emb: Optional[List[float]] = None,   # 上层已算好的问句向量，传入可省一次 embedding
    ) -> Optional[SkillMatch]:  # 命中返回 SkillMatch，否则返回 None
        """
        L2 向量语义路由：识别「换了个说法」的问句（关键词表里没有，但语义是一回事）。

        坑（见设计文档第四节）：向量分不出「查账号」和「注销账号」这种动作相反的说法，
        所以它只负责「选出候选」，最终仍由阈值把关，宁可漏判也不误判。
        """
        if not self._vector_available or not candidates:  # L2 已永久降级或无候选时不走这层
            return None  # 直接交给后续层处理
        cands = candidates[: self._vec_top_n]  # 只取前 N 个候选参与比对，控制 embedding 次数
        try:
            from modules.memory.embedding_service import get_embedding_service  # 延迟导入，避免循环依赖
            qv = query_emb if query_emb is not None else get_embedding_service().embed(query)  # 问句向量：上层给了就复用，否则现场编码
            vecs = self._skill_vectors([m.skill.name for m in cands])  # 取候选技能的语义向量（带缓存）
            scored = [  # 逐个候选算余弦相似度
                (_cosine(qv, vecs[m.skill.name]), m)  # 组成 (相似度, 候选) 二元组
                for m in cands if m.skill.name in vecs  # 只算成功拿到向量的候选
            ]
            if not scored:  # 一个都没算出来（向量全缺席）
                return None  # 交给 L3 继续判
            scored.sort(key=lambda x: x[0], reverse=True)  # 按相似度降序，取最优候选
            _logger.info(  # 打印全部候选的相似度，便于调阈值时回看分布
                "[L2向量路由] 相似度: "
                + ", ".join(f"{m.skill.name}={sim:.4f}" for sim, m in scored)  # 每个技能保留4位小数
            )
            sim, best = scored[0]  # 拆出最高相似度与对应候选
            if sim >= self._vec_threshold:  # 相似度过阈值，判定命中
                _logger.info(  # 命中日志
                    f"技能命中[L2向量]: {best.skill.name} | sim={sim:.4f} >= {self._vec_threshold}"
                )
                return SkillMatch(  # 构造命中结果
                    skill=best.skill,  # 命中的技能实例
                    score=round(float(sim), 4),  # 综合得分填相似度（保留4位小数）
                    hit_keywords=list(best.hit_keywords),  # 带上 L1 阶段命中的关键词（仅作展示）
                    route="vector",  # 标记判定层级为 L2
                    vector_score=round(float(sim), 4),  # 记录原始余弦相似度
                )
            _logger.info(f"[L2向量路由] 最高相似度{sim:.4f} < {self._vec_threshold}，下沉L3 LLM路由")  # 不过阈值则下沉
            return None  # 交给 L3 继续判
        except Exception as e:
            # 【设计意图】embedding 服务挂了不等于系统挂了：置永久降级标记，本次及以后请求都跳过 L2
            self._vector_available = False  # 永久关闭 L2，避免每次请求都重试失败路径
            _logger.warning(f"L2向量路由不可用，已永久降级: {e}")  # 记录降级原因
            return None  # 交给 L3 继续判

    _LLM_ROUTE_SYSTEM = (   # L3 的系统提示词：强制只输出 JSON，并约定 skill_name / confidence / reason 三字段
        "你是技能路由器。根据用户提问，从给定的候选技能中选出唯一最合适的一个。\n"  # 角色设定与任务说明
        "只输出 JSON，不要输出任何解释性文字。\n"  # 输出纪律：禁止自由文本，方便程序解析
        "若用户提问不属于任何候选技能，就把 skill_name 填为 \"none\"。\n"  # 约定「无合适技能」的兜底填法
        '输出格式: {"skill_name": "<候选技能的name，或 none>", '  # 输出格式约定：技能名字段
        '"confidence": <0到1之间的小数>, "reason": "<一句话理由>"}'  # 输出格式约定：置信度与理由字段
    )

    # 【什么】L3 LLM 兜底路由
    # 【作用】把 query 与候选清单交给模型分类，识别关键词和向量都判不出的复杂意图
    # 【为什么】前两层是「死规则」，真正歧义/复杂的表达只有 LLM 的理解力能兜住；但它最贵，所以放最后
    def _route_llm(self, query: str, candidates: List[SkillMatch]) -> Optional[SkillMatch]:  # 命中返回 SkillMatch，否则 None
        """
        L3 LLM 兜底路由：把 query + 候选技能清单丢给模型做分类。

        两条硬约束（设计文档第四节）：
        1. 强制结构化 JSON 输出，不让模型自由发挥长文本；
        2. 必须校验返回的 skill_name 真实存在，防幻觉；置信度不够也按未命中处理。
        """
        if not self._llm_enabled or not candidates:  # 开关关闭或无候选时不走 L3
            return None  # 按未命中返回
        cands = candidates[: self._llm_top_n]  # 只取前 N 个候选，控制 prompt 长度与调用成本
        try:
            from modules.agent.llm import LLMClient  # 延迟导入 LLM 客户端，避免循环依赖
            llm = LLMClient()  # 初始化 LLM 客户端
        except Exception as e:
            # 【设计意图】客户端都建不起来说明配置/依赖有根本问题，重试无意义，直接永久关闭 L3
            self._llm_enabled = False  # 永久关闭 L3 路由
            _logger.warning(f"L3 LLM路由不可用（客户端初始化失败），已永久降级: {e}")  # 记录降级原因
            return None  # 按未命中处理

        lines = []  # 逐行拼装候选技能清单文本
        for i, m in enumerate(cands, 1):  # 序号从 1 开始，方便模型按序号引用
            sk = m.skill  # 取出技能实例
            kws = "、".join(sk.trigger_keywords[:8])  # 触发词最多展示 8 个，防止 prompt 过长
            lines.append(  # 追加一个候选的展示块
                f'{i}. name="{sk.name}" | 展示名: {sk.display_name or sk.name}\n'  # 序号、内部名、展示名
                f"   说明: {sk.description}\n"  # 技能职责说明
                f"   典型触发词: {kws}"  # 典型触发词
            )
        prompt = (  # 拼装最终的用户提示词
            f"## 用户提问\n{query}\n\n"  # 第一段：用户原始问句
            f"## 候选技能（只能从中选一个，或选 none）\n" + "\n".join(lines)  # 第二段：候选清单
        )
        try:
            # temperature=0.1：路由分类要的是稳定可复现，不是发散，所以取低温
            result = llm.invoke_structured(  # 调用结构化输出接口做分类
                prompt, system_prompt=self._LLM_ROUTE_SYSTEM, temperature=0.1  # 传入提示词、系统提示与低温参数
            )
        except Exception as e:
            _logger.warning(f"L3 LLM路由调用失败，按未命中处理: {e}")  # 网络/服务异常只降级，不抛出
            return None  # 按未命中处理
        if not isinstance(result, dict) or result.get("error"):  # 返回不是 dict 或带 error 字段都视为解析失败
            _logger.warning(f"L3 LLM路由输出解析失败，按未命中处理: {str(result)[:200]}")  # 截断记录原始返回，防日志爆炸
            return None  # 按未命中处理

        name = str(result.get("skill_name", "") or "").strip()  # 取技能名并转字符串去空白
        reason = str(result.get("reason", "") or "")  # 取模型的判定理由（仅用于日志展示）
        try:
            confidence = float(result.get("confidence", 0) or 0)  # 置信度转 float
        except (TypeError, ValueError):
            confidence = 0.0  # 模型给了非数字时按 0 置信度处理

        if not name or name.lower() in ("none", "null", "无"):  # 模型明确表示无合适技能
            _logger.info(f"[L3 LLM路由] 模型判定无合适技能: reason={reason}")  # 记录模型理由
            return None  # 按未命中处理
            # 这一步是容错处理 要和本地的 skills的名字进行匹配
        skill = self._skills.get(name)  # 用模型给的名字在本地注册表反查技能实例
        if skill is None:  # 名字在本地注册表查不到
            _logger.warning(
                f"[L3 LLM路由] 模型返回了未注册的技能名 '{name}'（幻觉），按未命中处理"  # 防幻觉：模型编造的名字一律不认
            )
            return None  # 按未命中处理
        if confidence < self._llm_confidence:  # 置信度低于门槛
            _logger.info(
                f"[L3 LLM路由] 置信度{confidence} < {self._llm_confidence}，按未命中处理: {name}"  # 宁可漏判不误判
            )
            return None  # 按未命中处理
        _logger.info(f"技能命中[L3 LLM路由]: {name} | confidence={confidence} | reason={reason}")  # 命中日志
        return SkillMatch(  # 构造命中结果
            skill=skill,  # 命中的技能实例
            score=round(confidence, 4),  # 综合得分填置信度（保留4位小数）
            hit_keywords=[],  # L3 无关键词命中，按约定填空列表
            route="llm",  # 标记判定层级为 L3
            confidence=round(confidence, 4),  # 记录原始置信度
            route_reason=reason,  # 记录模型给出的判定理由
        )

    # 【什么】三级路由主入口
    # 【作用】L1 → L2 → L3 逐层兜底，命中即停；三层都没判定命中返回 None 走通用流程
    # 【为什么】「命中即停」让低成本层拦截尽可能多的请求，只有真复杂的才一路打到 LLM，整体成本最优
    def match(self, query: str, query_emb: Optional[List[float]] = None) -> Optional[SkillMatch]:  # 入参为问句与可选的问句向量
        """
        三级路由主入口：L1关键词 → L2向量语义 → L3 LLM兜底，命中即停。

        Args:
            query:     用户提问（应已过 sanitize_input 清洗）
            query_emb: 可选的问句向量。上层若已算过可传入，帮 L2 省一次 embedding 调用。

        Returns:
            SkillMatch（route 字段标明命中层级）；三层都未判定命中则返回 None（走通用流程）。
        """
        decision, candidates = self._route_keyword(query)  # 先走 L1 关键词路由，拿到裁决与候选
        if decision == "hit":  # L1 高分命中
            return candidates[0]  # 直接返回最高分候选，路由结束
        if decision == "miss":  # L1 判定与技能无关
            return None  # 走通用助手流程
        m = self._route_vector(query, candidates, query_emb)  # 模糊地带下沉 L2 向量语义路由
        if m:  # L2 命中
            return m  # 返回 L2 结果，路由结束
        m = self._route_llm(query, candidates)  # L2 也没把握，下沉 L3 LLM 兜底
        if m:  # L3 命中
            return m  # 返回 L3 结果，路由结束
        _logger.info("技能未命中（L1/L2/L3 三层均未判定命中），走通用助手流程")  # 三层全落空，记录日志
        return None  # 返回 None，由上层走通用流程

    # 【什么】一步完成「三级路由匹配 + 上下文构建」
    # 【作用】给 ReActAgent.prepare_state 用；异常被装饰器兜成 (None, "")，不打断主链路
    # 【为什么】主链路只想要「技能 + 注入文本」一个结果；把匹配与格式化合并并用装饰器兜底，技能系统出问题也绝不影响对话
    @handle_exception(default=(None, ""), log_msg="技能上下文构建异常")  # 任何异常都降级为 (None, "")
    def build_skill_context(  # 匹配 + 上下文构建一体化入口
        self, query: str, query_emb: Optional[List[float]] = None
    ) -> Tuple[Optional[BaseSkill], str]:  # 返回 (技能实例 或 None, 注入文本)
        """
        一步完成「三级路由匹配 + 上下文文本构建」，供 ReActAgent.prepare_state 调用。
        返回 (匹配到的技能, 注入Prompt的文本)；未命中时返回 (None, "")。
        """
        match = self.match(query, query_emb=query_emb)  # 先走三级路由匹配
        if not match:  # 未命中
            return None, ""  # 返回空技能与空文本
        _logger.info(  # 记录上下文构建结果
            f"技能上下文已构建: {match.skill.name} | "
            f"路由层={ROUTE_LABELS.get(match.route, match.route)}"  # 把路由层代号翻成中文展示名
        )
        return match.skill, match.skill.format_context(match)  # 返回技能实例与渲染好的注入文本

    # 【什么】生成全部技能的清单文本
    # 【作用】CLI / 调试查看注册了哪些技能；触发词超过 8 个时截断并用「…」示意
    # 【为什么】排障时需要一眼看清技能全貌；截断长触发词列表保证清单紧凑可读
    def describe_all(self) -> str:  # 返回多行清单文本
        """生成全部技能的清单文本（CLI / 调试用）。"""
        if not self._skills:  # 注册表为空
            return "（未注册任何技能）"  # 返回占位文案
        lines = []  # 逐行拼装清单
        for i, skill in enumerate(self._skills.values(), 1):  # 序号从 1 开始遍历全部技能
            tools = "、".join(skill.bound_tools) if skill.bound_tools else "无（走RAG/通用能力）"  # 绑定工具列表，空则给说明文案
            keywords = "、".join(skill.trigger_keywords[:8])  # 触发词最多展示 8 个
            if len(skill.trigger_keywords) > 8:  # 实际触发词超过 8 个
                keywords += " …"  # 追加省略号示意还有剩余
            lines.append(  # 追加一个技能的展示块
                f"{i}. [{skill.name}] {skill.display_name}\n"  # 序号、内部名、展示名
                f"   说明: {skill.description}\n"  # 技能职责说明
                f"   触发词: {keywords}\n"  # 触发词（可能截断）
                f"   绑定工具: {tools} | 优先级: {skill.priority}"  # 绑定工具与优先级
            )
        return "\n".join(lines)  # 用换行拼接成完整清单返回
