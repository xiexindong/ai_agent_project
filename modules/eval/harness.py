"""
被测对象装配（System Under Test）
================================
把「要评测的那套东西」按模式装配出来，让 runner 只管跑、不管怎么造。

【为什么要分 offline / online 两档】
    同一套评测在两种场合都要能跑：
        CI / 本地没网没密钥  → 要跑得动、跑得快、结果可复现
        发版前人工验收       → 要跑完整链路（含 embedding 与 LLM）
    两档的差异只在于「外部依赖开不开」，被测的业务逻辑完全一样，
    否则评测的就不是同一件东西了。

【offline 档为什么强制关掉向量与 LLM 路由】
    可复现性。向量要调 embedding 服务、L3 要调 LLM，两者都受网络、
    配额、模型版本影响 —— 同一份代码今天 0.82 明天 0.79，没法做回归门禁。
    offline 档只跑确定性部分（L1 关键词 + 关键词召回 + 规则风控 + 规则意图），
    它测的是「我们的编排逻辑」，不是「外部模型的表现」。

【offline 档测不到什么，必须写清楚】
    测不到语义召回的真实效果、测不到答案质量。这两项归 online 档，
    报告里会标注 mode，避免拿 offline 的分数冒充端到端效果。
"""
from __future__ import annotations  # 启用注解延迟求值：类型标注可引用后方才定义的名字（如方法内注解）

import json  # 标准库 JSON 解析：读取工具目录文件用
import time  # 标准库时间函数：给每次被测调用计时（换算成 latency_ms）
from dataclasses import dataclass, field  # dataclass 装饰器与字段默认值工厂：定义轻量数据类 Sut
from pathlib import Path  # 跨平台路径对象：读取本地工具目录文件用
from typing import Any, Dict, List, Optional  # 泛型类型注解工具：提升可读性并支持静态检查


# 模拟 RAG 召回的长上下文：约上千 token，用来稳定触发预算裁剪
# 【为什么用固定文本而不是随机生成】随机文本会让 token 数每次不同，
#   报告里的 avg_tokens 就没法跨次比较，回归判定也失去意义。
_MOCK_RAG = (  # 开始拼接模拟长上下文：一段固定制度文本重复多次，凑出约上千 token 的规模
    "《差旅费用报销管理办法》第三章 报销标准："  # 段首：制度名与章节，模拟真实制度文档的开头
    "员工因公出差产生的交通费、住宿费、市内交通费按本办法执行。"  # 总则句：说明制度适用范围
    "一、交通费：高铁二等座、飞机经济舱据实报销，超标准部分自理。"  # 条款一：交通费标准
    "二、住宿费：一线城市每晚不超过 600 元，二线城市 450 元，其他城市 350 元。"  # 条款二：住宿费标准
    "三、市内交通：凭票据实报销，单日上限 100 元。"  # 条款三：市内交通标准
    "四、审批流程：提交后由直属主管初审，财务复核，超过 5000 元需分管副总审批。"  # 条款四：审批流程
    "五、报销时限：出差结束后 30 个工作日内提交，逾期不予受理。"  # 条款五：报销时限
    "六、票据要求：发票抬头须为公司全称，税号正确，否则退回重开。"  # 条款六：票据要求
) * 6  # 整段重复 6 遍：使 token 规模稳定达到触发预算裁剪的量级，且每次运行完全一致

_MOCK_PERMISSION = "可访问：本人报销单、本人订单；禁止：他人数据、薪资、人事档案"  # 模拟权限上下文：固定文本，保证每次组装的提示词逐字节一致
_MOCK_MEMORY = "用户上月咨询过差旅报销标准；偏好分点作答"  # 模拟记忆上下文：固定文本，避免真实记忆服务引入结果波动
_MOCK_TOOLS = "calculator / data_query / web_search / get_current_time / knowledge_stats"  # 模拟工具描述串：与真实工具目录的量级相当


# 【什么】离线网关替身类：冒充真 MCP 网关，只读目录文件、不起子进程
# 【作用】为 router 提供「目录里有哪些工具、描述是什么」的数据，同时保持离线可复现
# 【为什么】真网关每次调用要 Popen 一个 MCP 子进程，40 条用例起 40 次进程又慢又不稳定；
#           而路由决策只需要目录数据，数据已落在 data/mcp_catalog.json 里，直接读它即可
class StubGateway:
    """离线网关替身：只提供工具目录，不起子进程。

    【为什么要这个替身】
        真网关会 Popen 一个 MCP 子进程，评测跑 40 条就要起 40 次进程，
        既慢又引入"子进程启动失败"这种与被测逻辑无关的不稳定因素。
        而路由决策真正需要的只是「目录里有哪些工具、描述是什么」，
        这份数据已经落在 data/mcp_catalog.json 里 —— 直接读它即可，
        既离线可复现，又是项目的真实数据。

    【ensure_registered 为什么是空实现】
        评测只测"该选谁"，不测"选完之后能不能真的调用"。
        装工具要连 server，属于集成测试范畴，留给用户自己跑 demo 验证。
    """

    # 【什么】构造替身并加载目录
    # 【作用】记录目录文件路径，立即尝试读出工具清单缓存起来
    # 【为什么】读文件失败不抛异常而是降级为空目录：目录缺失不该让整个评测跑不起来
    def __init__(self, catalog_path: str = "data/mcp_catalog.json"):  # 入参：目录文件路径，默认指向项目真实目录
        self.catalog_path = catalog_path  # 保存目录文件路径，排查问题时能知道数据从哪来
        self._catalog: List[Dict] = []  # 目录缓存初始化为空列表：读失败时候选池为空而不是 None，下游不用判空
        try:  # 读取并解析目录文件；任何失败都走降级分支而不是炸掉
            data = json.loads(Path(catalog_path).read_text(encoding="utf-8"))  # 按 UTF-8 读出整个文件并解析为 dict
            self._catalog = list(data.get("tools", []) or [])  # 取 tools 字段；字段缺失或为 null 时都退化为空列表
        except Exception as e:  # 文件不存在 / JSON 非法 / 编码错误等一切异常
            print(f"[warn] 工具目录读取失败，候选池只剩本地工具: {e}")  # 打警告让使用者知道发生了降级，但不中断评测

    # 【什么】返回远程工具目录
    # 【作用】与真网关同名同签名，router 无需区分
    # 【为什么】router 只认接口不认实现（鸭子类型），替身与真网关可以无缝互换
    def catalog(self) -> List[Dict]:  # 对外暴露目录查询接口，签名与真网关保持一致
        return list(self._catalog)  # 返回浅拷贝：防止调用方意外改到内部缓存，污染后续用例

    # 【什么】按需注册（空实现）
    # 【作用】满足 router 的调用契约；评测不测可执行性
    # 【为什么】评测只关心"该选谁"，不关心"能不能调通"；真去连 server 属于集成测试范畴
    def ensure_registered(self, registry, names) -> int:  # 保持与真网关相同的调用签名，router 调用时不报错
        return 0  # 声称"新注册了 0 个工具"：替身语义下目录里的工具视为已就绪


# 【什么】被测系统句柄：一个数据类，打包装配好的全套组件
# 【作用】runner 只面对这个统一句柄，不需要关心组件是怎么造出来的
# 【为什么】装配逻辑复杂且分档（offline/online），集中在 build_sut 一处；
#           句柄用数据类承载，让「装配」与「使用」彻底解耦
@dataclass  # 用 dataclass 自动生成 __init__/__repr__ 等样板代码
class Sut:
    """被测系统句柄：持有装配好的一套组件。

    Attributes:
        mode:         "offline" 或 "online"
        notes:        装配过程中的降级说明（如"embedding 不可用，已降级"）
    """

    mode: str = "offline"  # 运行档位：offline=确定性无外部依赖；online=完整链路
    notes: List[str] = field(default_factory=list)  # 装配降级说明列表：逐条记录"什么被关了/换成了什么"，随报告输出
    skill_registry: Any = None  # 技能注册表：L1/L2/L3 三级技能路由的数据源
    tool_registry: Any = None  # 本地工具注册表：工具名单的静态目录
    tool_index: Any = None  # 工具召回索引：Tool-RAG（关键词或向量）召回器
    router: Any = None  # 路由编排器：把上面三样组装成一次 route() 调用

    # ------------------------------------------------------------------
    # 【四个被测面】技能路由 / 工具路由 / 意图 / 风控，加一个提示词组装
    #   每个方法都返回「标准化 dict + 耗时」，异常时返回 error 而不是抛出，
    #   保证一条用例炸了不影响其余 39 条。
    # ------------------------------------------------------------------
    # 【什么】跑一次路由（技能 + 工具名单）
    # 【作用】返回标准化结果；异常时 ok=False 并带 error
    # 【为什么】runner 只认固定字段（ok/skill/tools/...），异常也必须落成同构 dict，
    #           否则一条异常用例会让整批评测中断
    def route(self, query: str) -> Dict:  # 被测面①：对一句提问做完整路由决策
        started = time.time()  # 记录起始时间戳，用于计算本次路由耗时
        try:  # 路由内部任何异常（含组件 bug）都兜底成标准化失败结果
            decision = self.router.route(query, registry=self.tool_registry)  # 调编排器做决策，注入工具注册表供其查本地工具
            skill = getattr(decision, "skill", None)  # 取决策命中的技能对象；防御性 getattr 兼容不同返回类型
            return {  # 成功分支：把 decision 拍平成标准化 dict
                "ok": True,  # 标记本次执行成功
                "skill": getattr(skill, "name", "") if skill else "",  # 命中技能名；未命中则为空串（= 通用流程）
                "skill_route": getattr(decision, "skill_route", ""),  # 命中的路由层级（L1/L2/L3），供归因分析
                "tools": list(getattr(decision, "tool_names", []) or []),  # 工具名单；None 兜底成空列表
                "source": getattr(decision, "source", ""),  # 决策来源（哪一级路由拍的板），供排查
                "reason": getattr(decision, "reason", ""),  # 决策理由文本，供人读报告时理解
                "latency_ms": round((time.time() - started) * 1000, 1),  # 耗时换算毫秒，保留 1 位小数
                "error": "",  # 成功时错误串为空，与失败分支同构
            }
        except Exception as e:  # 失败分支：同样返回标准化 dict，只是 ok=False
            return {"ok": False, "skill": "", "skill_route": "", "tools": [],  # 各结果字段置空
                    "source": "", "reason": "", "latency_ms": 0.0,  # 失败不记耗时（数值无意义）
                    "error": f"{type(e).__name__}: {e}"}  # 记录异常类型与消息：runner 据此区分执行错误与效果不达标

    # 【什么】跑一次意图识别
    # 【作用】offline 档强制 use_llm=False 走规则，保证确定性
    # 【为什么】意图若走 LLM，结果受模型版本与网络影响；离线档必须锁死规则路径才可复现
    def intent(self, query: str) -> Dict:  # 被测面②：对一句提问做意图分类
        from modules.pipeline.intent import recognize_intent  # 延迟导入：避免模块级循环依赖，也让本文件可被独立阅读

        started = time.time()  # 记录起始时间戳
        try:  # 意图识别异常兜底
            use_llm = self.mode == "online"  # 仅在线档允许走 LLM；离线档强制规则路径
            r = recognize_intent(query, llm=None, use_llm=use_llm)  # 调用意图识别；llm=None 表示不注入真实 LLM 客户端
            return {"ok": True, "intent": str(getattr(r, "intent_type", "")),  # 成功：返回意图大类字符串
                    "confidence": float(getattr(r, "confidence", 0.0) or 0.0),  # 置信度；None 兜底成 0.0
                    "latency_ms": round((time.time() - started) * 1000, 1), "error": ""}  # 耗时与空错误串
        except Exception as e:  # 失败分支
            return {"ok": False, "intent": "", "confidence": 0.0,  # 结果字段置空/置零
                    "latency_ms": 0.0, "error": f"{type(e).__name__}: {e}"}  # 记录异常信息

    # 【什么】跑一次前置风控
    # 【作用】纯规则，两档行为一致
    # 【为什么】风控是纯规则实现、不依赖外部服务，天然确定，无需随档位切换行为
    def risk(self, query: str) -> Dict:  # 被测面③：对一句提问做风险判定
        from modules.pipeline.risk_guard import check_risk  # 延迟导入风控检查函数

        try:  # 风控异常兜底
            r = check_risk(query)  # 调纯规则风控检查
            return {"ok": True, "passed": bool(getattr(r, "passed", True)),  # 是否放行；缺字段时按放行处理
                    "level": str(getattr(r, "level", "")),  # 风险等级文本
                    "action": str(getattr(r, "action", "")), "error": ""}  # 处置动作（拦截/放行/改写）与空错误串
        except Exception as e:  # 失败分支
            return {"ok": False, "passed": True, "level": "", "action": "",  # 兜底放行：评测框架自身故障不应误报为拦截
                    "error": f"{type(e).__name__}: {e}"}  # 记录异常信息

    # 【什么】跑一次提示词组装（主链路模板 react.reason）
    # 【作用】评测 prompt 这一层：能否渲染成功、是否守住 token 预算、裁剪是否记录
    # 【为什么】prompt 层是成本与效果的闸门：上下文超预算会推高费用并挤掉关键信息，
    #           必须用固定长上下文稳定复现裁剪行为，才能验证预算逻辑真的生效
    def build_prompt(self, query: str) -> Dict:  # 被测面④：对一句提问组装完整提示词
        from modules.prompt import PromptBuilder  # 延迟导入提示词组装器

        started = time.time()  # 记录起始时间戳
        try:  # 组装异常兜底
            b = PromptBuilder(  # 构造组装器：主链路 ReAct 推理模板
                template_name="react.reason",  # 使用 react.reason 模板（与生产主链路一致）
                strategy="react",  # 推理策略为 ReAct（思考-行动-观察循环）
                role="你是一个企业级智能AI助手。",  # 角色设定，与生产默认一致
                max_iter=10,  # 最大推理轮数，与生产配置对齐
                max_context_tokens=3000,  # 上下文 token 预算上限：评测判定"是否守住预算"的红线
            )
            rec = b.build_record(  # 执行组装并返回记录（含 token 统计与裁剪明细）
                query,  # 用户提问原文，组装的核心输入
                permission_context=_MOCK_PERMISSION,  # 注入固定权限上下文
                skill_context="（未匹配到专门技能，按通用企业助手流程处理）",  # 固定技能上下文：本面评测的是模板本身而非技能命中
                rag_context=_MOCK_RAG,  # 注入固定长 RAG 上下文，稳定触发裁剪
                memory_context=_MOCK_MEMORY,  # 注入固定记忆上下文
                tool_history="（未提供）",  # 固定工具历史占位
                observation="（初始状态，暂无工具观察）",  # 固定观察占位
                tools_desc=_MOCK_TOOLS,  # 注入固定工具描述串
            )
            return {  # 成功分支：标准化组装结果
                "ok": True,  # 渲染成功
                "prompt_tokens": int(rec.get("prompt_tokens", 0)),  # 完整提示词 token 数
                "context_tokens": int(rec.get("context_tokens", 0)),  # 上下文部分 token 数（预算判定用）
                "trimmed": list(rec.get("trimmed", []) or []),  # 被裁剪掉的区块名单：验证裁剪是否被如实记录
                "template": str(rec.get("template", "")),  # 实际使用的模板名
                "latency_ms": round((time.time() - started) * 1000, 1),  # 组装耗时
                "error": "",  # 空错误串
            }
        except Exception as e:  # 失败分支
            return {"ok": False, "prompt_tokens": 0, "context_tokens": 0,  # 统计字段置零
                    "trimmed": [], "template": "",  # 明细字段置空
                    "latency_ms": 0.0, "error": f"{type(e).__name__}: {e}"}  # 记录异常信息


# 【什么】装配一套被测对象
# 【作用】按 mode 决定外部依赖开不开，返回 Sut
# 【为什么】装配顺序与降级开关是评测可信度的根基，必须集中在唯一的工厂函数里，
#           避免每个调用方各自拼装出口径不一的"被测对象"
def build_sut(mode: str = "offline", max_tools: int = 8, top_k: int = 5,  # 工厂入口：mode 选档；max_tools/top_k 对齐生产参数
              catalog_path: str = "data/mcp_catalog.json") -> Sut:  # catalog_path 指向离线替身的目录数据源
    """装配被测系统。

    Args:
        mode:         "offline"（确定性，无外部依赖）/ "online"（完整链路）
        max_tools:    单轮工具名单上限，对齐生产配置
        top_k:        语义召回保留数
        catalog_path: 远程工具目录（离线替身的数据源），给空串则只用本地工具

    【为什么要显式关掉 L2/L3 而不是等它自己失败降级】
        等它失败要付出一次真实的网络超时（可能几十秒 × 每条用例），
        而且"失败后永久降级"意味着前几条用例的耗时和后面的不一样，
        测出来的延迟没有可比性。显式关闭 = 行为确定 + 跑得快。
    """
    from modules.skills import SkillRegistry, get_default_skills  # 延迟导入技能注册表与默认技能集
    from modules.tools.tool_registry import ToolRegistry  # 延迟导入本地工具注册表
    from modules.tools.tool_rag import ToolRAGIndex  # 延迟导入工具召回索引
    from modules.agent.router import AgentRouter  # 延迟导入路由编排器

    notes: List[str] = []  # 降级说明列表：把装配过程中每个"非生产行为"记下来，随报告输出
    offline = mode != "online"  # 只要不是显式的 online 一律按离线处理：防止拼写错误意外触网

    skill_registry = SkillRegistry(  # 创建技能注册表
        llm_route_enabled=False if offline else None,  # 离线显式关 L3；在线传 None 走组件默认配置
    )
    skill_registry.register_batch(get_default_skills())  # 批量注册项目默认技能：技能候选池与生产一致

    tool_registry = ToolRegistry()  # 创建空的本地工具注册表
    tool_registry.register_batch(tool_registry.get_default_tools())  # 注册默认本地工具集：与生产启动流程一致

    tool_index = ToolRAGIndex(top_k=top_k)  # 创建工具召回索引：召回条数与生产一致

    # 【设计意图】离线档的确定性来自"显式关闭"而非"等失败再降级"：
    #   直接打掉 _vector_available 标志，让向量召回从第一毫秒起就不存在，
    #   避免"前几条用例先超时再降级"造成的耗时前后不可比。
    if offline:  # 离线档：关闭全部非确定性来源
        # 强制关掉两个非确定性来源，把被测面收敛到"我们的编排逻辑"
        skill_registry._vector_available = False  # 关技能 L2 向量路由（不连 embedding 服务）
        tool_index._vector_available = False  # 关工具向量召回（退化为关键词召回）
        notes.append("L2 向量路由已关闭（离线档）")  # 记录降级说明①
        notes.append("L3 LLM 路由已关闭（离线档）")  # 记录降级说明②
        notes.append("Tool-RAG 走关键词召回（离线档）")  # 记录降级说明③
    else:  # 在线档：完整链路
        notes.append("完整三级路由 + embedding 语义召回")  # 记录当前是完整链路

    gateway = StubGateway(catalog_path) if catalog_path else None  # 有目录路径就用替身读目录；空串则完全不用远程工具
    if gateway is not None:  # 用了替身就把目录规模记进说明
        notes.append(f"远程工具目录：{catalog_path}（{len(gateway.catalog())} 个，离线替身）")  # 记录目录来源与候选数量

    router = AgentRouter(  # 装配路由编排器：把前面所有组件串成一次决策
        skill_registry=skill_registry,  # 注入技能注册表
        tool_index=tool_index,  # 注入工具召回索引
        gateway=gateway,          # 不起子进程，只提供目录
        top_k=top_k,  # 语义召回保留数，与生产一致
        max_tools=max_tools,  # 单轮工具名单上限，与生产一致
    )

    return Sut(  # 把装配成果打包成句柄返回
        mode=mode,  # 记录档位：报告里要标注
        notes=notes,  # 带上全部降级说明
        skill_registry=skill_registry,  # 技能注册表句柄
        tool_registry=tool_registry,  # 工具注册表句柄
        tool_index=tool_index,  # 工具索引句柄
        router=router,  # 路由器句柄（runner 主要靠它）
    )
