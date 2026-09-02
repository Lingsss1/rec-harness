"use client";

import { FormEvent, PointerEvent as ReactPointerEvent, useEffect, useMemo, useRef, useState } from "react";

type Difficulty = "简单" | "中等" | "困难" | "待标注";
type Tab = "hot" | "projects";
type ProblemProgress = { times: number; repeat: boolean; notes: string; lastPracticed?: string };
type KnowledgeProgress = { level: number; interval: number; due?: string; reviews: number; notes: string };
type Problem = { id: number; title: string; slug: string; difficulty: Difficulty; focus: string; category: string; order: number; imported?: boolean; href?: string };
type Knowledge = { id: string; title: string; category: string; subtitle: string; summary: string; details: string[]; metrics: string[]; recall: string; lessonIds?: string[] };
type Lesson = { id: string; title: string; category: string; description: string; href: string; depth: string; source: string };
type ParsedQuestion = { link: string; id?: number; title?: string; difficulty?: "Easy" | "Medium" | "Hard"; slug?: string; href?: string; error?: string };

const rawCategories: Array<{ name: string; logic: string; items: Array<[number, string, string, Difficulty, string]> }> = [
  { name: "哈希", logic: "从补数查找，到分组键设计，再到集合边界扩张。", items: [
    [1,"两数之和","two-sum","简单","遍历时把“需要的补数”放进哈希表；先查后存可避免重复使用同一元素。"],
    [49,"字母异位词分组","group-anagrams","中等","把排序串或 26 维字符频次向量作为分组键。"],
    [128,"最长连续序列","longest-consecutive-sequence","中等","只从没有前驱的数字开始扩张，保证每个数最多被访问一次。"],
  ]},
  { name: "双指针", logic: "先掌握同向覆盖，再练相向收缩，最后处理边界贡献。", items: [
    [283,"移动零","move-zeroes","简单","慢指针指向下一个非零写入位，快指针负责扫描。"],
    [11,"盛最多水的容器","container-with-most-water","中等","面积由短板决定，因此每轮移动更短的一侧。"],
    [15,"三数之和","3sum","中等","排序后固定一个数，另外两数相向移动并跳过重复值。"],
    [42,"接雨水","trapping-rain-water","困难","维护两侧最大高度；较低一侧的水量可被当场确定。"],
  ]},
  { name: "滑动窗口", logic: "先练无重复窗口，再练固定目标计数窗口。", items: [
    [3,"无重复字符的最长子串","longest-substring-without-repeating-characters","中等","右端扩张、左端收缩，用计数表维护窗口合法性。"],
    [438,"找到字符串中所有字母异位词","find-all-anagrams-in-a-string","中等","固定窗口长度，维护字符差分或匹配字符种类数。"],
  ]},
  { name: "子串", logic: "前缀和解决区间计数，单调队列解决窗口极值，欠账表解决最小覆盖。", items: [
    [560,"和为 K 的子数组","subarray-sum-equals-k","中等","当前前缀和为 s，历史 s-k 的出现次数就是新增答案。"],
    [239,"滑动窗口最大值","sliding-window-maximum","困难","单调递减队列保存仍在窗口内、且可能成为最大值的下标。"],
    [76,"最小覆盖子串","minimum-window-substring","困难","扩张直到覆盖，再收缩到刚好不覆盖；记录最短合法窗口。"],
  ]},
  { name: "普通数组", logic: "从局部最优与区间合并，推进到原地映射技巧。", items: [
    [53,"最大子数组和","maximum-subarray","中等","若前缀贡献为负就舍弃；同时维护以当前位置结尾的最优值。"],
    [56,"合并区间","merge-intervals","中等","按左端点排序，只需比较当前区间与结果尾部。"],
    [189,"轮转数组","rotate-array","中等","三次翻转：整体、前 k 段、后 n-k 段。"],
    [238,"除自身以外数组的乘积","product-of-array-except-self","中等","答案先存前缀积，再从右向左乘后缀积。"],
    [41,"缺失的第一个正数","first-missing-positive","困难","把值 x 原地放到下标 x-1，最后找第一个错位位置。"],
  ]},
  { name: "矩阵", logic: "先处理标记与遍历，再学习二维原地变换和单调搜索。", items: [
    [73,"矩阵置零","set-matrix-zeroes","中等","用首行首列充当标记，并单独记录它们自身是否需要清零。"],
    [54,"螺旋矩阵","spiral-matrix","中等","维护上下左右四条边界，每走完一边立即收缩并检查。"],
    [48,"旋转图像","rotate-image","中等","先沿主对角线转置，再逐行翻转。"],
    [240,"搜索二维矩阵 II","search-a-2d-matrix-ii","中等","从右上角开始，较大向左、较小向下。"],
  ]},
  { name: "链表", logic: "反转与快慢指针是底座，再进入分治、复制与缓存结构。", items: [
    [160,"相交链表","intersection-of-two-linked-lists","简单","双指针分别走 A+B 与 B+A，抵消长度差。"],
    [206,"反转链表","reverse-linked-list","简单","保存 next，反转当前指针，再推进 prev 与 cur。"],
    [234,"回文链表","palindrome-linked-list","简单","快慢指针找中点，反转后半段并逐一比较。"],
    [141,"环形链表","linked-list-cycle","简单","快慢指针若在有限步内相遇，则存在环。"],
    [142,"环形链表 II","linked-list-cycle-ii","中等","相遇后一个指针回到头节点，同速前进再次相遇即入口。"],
    [21,"合并两个有序链表","merge-two-sorted-lists","简单","哑节点统一头部处理，每次接入较小节点。"],
    [2,"两数相加","add-two-numbers","中等","同步遍历两链表，逐位相加并维护进位。"],
    [19,"删除链表的倒数第 N 个结点","remove-nth-node-from-end-of-list","中等","快指针先走 n 步，再与慢指针同步前进。"],
    [24,"两两交换链表中的节点","swap-nodes-in-pairs","中等","用哑节点维护每组前驱，交换后推进两步。"],
    [25,"K 个一组翻转链表","reverse-nodes-in-k-group","困难","先确认剩余节点足够 k 个，再局部反转并连接前后段。"],
    [138,"随机链表的复制","copy-list-with-random-pointer","中等","原节点后插入克隆节点，借相邻关系复制 random，再拆链。"],
    [148,"排序链表","sort-list","中等","快慢指针断链，归并排序两半；复杂度 O(n log n)。"],
    [23,"合并 K 个升序链表","merge-k-sorted-lists","困难","小根堆维护每条链表当前最小节点，或使用分治合并。"],
    [146,"LRU 缓存","lru-cache","中等","哈希表负责 O(1) 定位，双向链表负责 O(1) 更新新旧顺序。"],
  ]},
  { name: "二叉树", logic: "遍历与递归定义优先，然后学习路径状态、构造与公共祖先。", items: [
    [94,"二叉树的中序遍历","binary-tree-inorder-traversal","简单","递归定义是左-根-右；迭代用栈模拟递归路径。"],
    [104,"二叉树的最大深度","maximum-depth-of-binary-tree","简单","深度等于左右子树最大深度加一。"],
    [226,"翻转二叉树","invert-binary-tree","简单","交换左右子树，再递归处理两个孩子。"],
    [101,"对称二叉树","symmetric-tree","简单","递归比较外侧与内侧节点是否镜像相等。"],
    [543,"二叉树的直径","diameter-of-binary-tree","简单","后序计算高度，同时用左右高度之和更新直径。"],
    [102,"二叉树的层序遍历","binary-tree-level-order-traversal","中等","队列 BFS；每轮固定读取当前队列长度即一层。"],
    [108,"将有序数组转换为二叉搜索树","convert-sorted-array-to-binary-search-tree","简单","选择中点为根，递归构造左右区间以保持平衡。"],
    [98,"验证二叉搜索树","validate-binary-search-tree","中等","递归传递合法上下界；不是只比较父子节点。"],
    [230,"二叉搜索树中第 K 小的元素","kth-smallest-element-in-a-bst","中等","BST 中序遍历天然递增，访问第 k 个节点即可。"],
    [199,"二叉树的右视图","binary-tree-right-side-view","中等","层序遍历记录每层最后一个节点。"],
    [114,"二叉树展开为链表","flatten-binary-tree-to-linked-list","中等","后序维护已展开链表头，按右-左-根逆序连接。"],
    [105,"从前序与中序遍历序列构造二叉树","construct-binary-tree-from-preorder-and-inorder-traversal","中等","前序确定根，中序确定左右子树范围；哈希表加速定位。"],
    [437,"路径总和 III","path-sum-iii","中等","DFS 前缀和；路径数来自当前和减目标和的历史出现次数。"],
    [236,"二叉树的最近公共祖先","lowest-common-ancestor-of-a-binary-tree","中等","若左右递归都返回非空，当前节点就是最近公共祖先。"],
    [124,"二叉树中的最大路径和","binary-tree-maximum-path-sum","困难","向父节点只返回单边最大贡献，但答案可取左右两边加根。"],
  ]},
  { name: "图论", logic: "网格 DFS/BFS 起步，接着掌握拓扑排序和 Trie。", items: [
    [200,"岛屿数量","number-of-islands","中等","遇到陆地就 DFS/BFS 淹没整个连通块，并把答案加一。"],
    [994,"腐烂的橘子","rotting-oranges","中等","多源 BFS：所有腐烂橘子同时入队，按层推进分钟数。"],
    [207,"课程表","course-schedule","中等","拓扑排序统计入度；最终处理节点数等于课程数才无环。"],
    [208,"实现 Trie","implement-trie-prefix-tree","中等","每个节点维护字符孩子与单词结束标记。"],
  ]},
  { name: "回溯", logic: "明确路径、选择列表、结束条件，再学习剪枝与去重。", items: [
    [46,"全排列","permutations","中等","路径长度等于 n 时收集答案，用 used 数组标记已选元素。"],
    [78,"子集","subsets","中等","每个元素都有选或不选；也可从 start 开始枚举下一选择。"],
    [17,"电话号码的字母组合","letter-combinations-of-a-phone-number","中等","每层对应一个数字，枚举映射字母并递归下一位。"],
    [39,"组合总和","combination-sum","中等","允许重复选择当前数，因此下一层仍从当前下标开始。"],
    [22,"括号生成","generate-parentheses","中等","左括号数小于 n 可放；右括号数小于左括号数才可放。"],
    [79,"单词搜索","word-search","中等","从每个格子尝试 DFS，进入前标记，回退时恢复。"],
    [131,"分割回文串","palindrome-partitioning","中等","枚举切割终点，仅当当前片段是回文时继续递归。"],
    [51,"N 皇后","n-queens","困难","逐行放皇后，用列、主对角线、副对角线集合剪枝。"],
  ]},
  { name: "二分查找", logic: "先统一区间定义，再处理答案边界、旋转数组和双数组分割。", items: [
    [35,"搜索插入位置","search-insert-position","简单","找第一个大于等于 target 的位置，使用左闭右开模板。"],
    [74,"搜索二维矩阵","search-a-2d-matrix","中等","把矩阵视为长度 m×n 的有序数组，通过除余映射坐标。"],
    [34,"在排序数组中查找元素的第一个和最后一个位置","find-first-and-last-position-of-element-in-sorted-array","中等","分别找第一个 ≥target 与第一个 >target 的位置。"],
    [33,"搜索旋转排序数组","search-in-rotated-sorted-array","中等","每轮至少一半有序，判断 target 是否落在有序半边。"],
    [153,"寻找旋转排序数组中的最小值","find-minimum-in-rotated-sorted-array","中等","比较中点与右端：中点更大说明最小值在右侧。"],
    [4,"寻找两个正序数组的中位数","median-of-two-sorted-arrays","困难","在较短数组上二分分割线，使左半数量固定且左右有序。"],
  ]},
  { name: "栈", logic: "括号匹配与状态栈起步，再进入单调栈的“下一个更大”和边界计算。", items: [
    [20,"有效的括号","valid-parentheses","简单","遇左括号入栈，遇右括号必须与栈顶匹配。"],
    [155,"最小栈","min-stack","中等","辅助栈同步保存当前位置对应的历史最小值。"],
    [394,"字符串解码","decode-string","中等","遇左括号保存当前字符串与倍数，遇右括号弹栈拼接。"],
    [739,"每日温度","daily-temperatures","中等","单调递减栈保存尚未找到更高温度的下标。"],
    [84,"柱状图中最大的矩形","largest-rectangle-in-histogram","困难","单调递增栈在柱高下降时确定被弹柱子的左右边界。"],
  ]},
  { name: "堆", logic: "从固定大小 Top-K 到频次排序，最后维护数据流中位数。", items: [
    [215,"数组中的第 K 个最大元素","kth-largest-element-in-an-array","中等","维护大小为 k 的小根堆，堆顶就是第 k 大。"],
    [347,"前 K 个高频元素","top-k-frequent-elements","中等","先计数，再用小根堆把候选维持在 k 个。"],
    [295,"数据流的中位数","find-median-from-data-stream","困难","大根堆存较小一半，小根堆存较大一半，并保持数量平衡。"],
  ]},
  { name: "贪心", logic: "识别局部选择能否保留全局最优，重点练覆盖范围。", items: [
    [121,"买卖股票的最佳时机","best-time-to-buy-and-sell-stock","简单","维护历史最低价格，用当前价减最低价更新最大利润。"],
    [55,"跳跃游戏","jump-game","中等","遍历可达位置并持续更新最远可达下标。"],
    [45,"跳跃游戏 II","jump-game-ii","中等","当前跳跃边界内统计下一步最远范围，到边界时增加步数。"],
    [763,"划分字母区间","partition-labels","中等","记录每个字符最后位置，扫描中不断扩张当前片段右边界。"],
  ]},
  { name: "动态规划", logic: "从一维状态转移开始，覆盖完全背包、序列 DP 与区间合法性。", items: [
    [70,"爬楼梯","climbing-stairs","简单","到第 i 阶的方法数来自 i-1 与 i-2。"],
    [118,"杨辉三角","pascals-triangle","简单","每个内部元素等于上一行相邻两数之和。"],
    [198,"打家劫舍","house-robber","中等","状态只依赖偷当前的 i-2 与不偷当前的 i-1。"],
    [279,"完全平方数","perfect-squares","中等","完全背包：枚举平方数，更新组成每个和的最少数量。"],
    [322,"零钱兑换","coin-change","中等","dp[x] 表示凑出金额 x 的最少硬币数，由所有可用硬币转移。"],
    [139,"单词拆分","word-break","中等","dp[i] 表示前 i 个字符可拆分，枚举最后一个单词起点。"],
    [300,"最长递增子序列","longest-increasing-subsequence","中等","维护不同长度递增子序列的最小末尾，用二分更新。"],
    [152,"乘积最大子数组","maximum-product-subarray","中等","负数会交换最大与最小，因此同时维护两种状态。"],
    [416,"分割等和子集","partition-equal-subset-sum","中等","转化为 0/1 背包：能否选出和为总和一半的子集。"],
    [32,"最长有效括号","longest-valid-parentheses","困难","dp[i] 表示以 i 结尾的最长合法长度，按前一个字符分类转移。"],
  ]},
  { name: "多维动态规划", logic: "二维网格、双序列、区间回文，重点明确状态含义和遍历方向。", items: [
    [62,"不同路径","unique-paths","中等","dp[i][j] 来自上方与左方路径数之和。"],
    [64,"最小路径和","minimum-path-sum","中等","dp[i][j] 等于当前格值加上方、左方较小者。"],
    [5,"最长回文子串","longest-palindromic-substring","中等","区间两端相等且内部为回文，则整个区间为回文。"],
    [1143,"最长公共子序列","longest-common-subsequence","中等","字符相等取左上加一，否则取上方与左方最大值。"],
    [72,"编辑距离","edit-distance","中等","插入、删除、替换分别对应三个相邻状态。"],
  ]},
  { name: "技巧", logic: "位运算、投票、三路快排、排列规律和环检测收尾。", items: [
    [136,"只出现一次的数字","single-number","简单","异或满足相同数抵消、与顺序无关。"],
    [169,"多数元素","majority-element","简单","Boyer-Moore 投票：不同元素互相抵消，多数元素最终留下。"],
    [75,"颜色分类","sort-colors","中等","三指针把 0、未知区间、2 原地分隔。"],
    [31,"下一个排列","next-permutation","中等","从右找第一个下降位，与右侧最小较大值交换，再反转后缀。"],
    [287,"寻找重复数","find-the-duplicate-number","中等","把数组值视作 next 指针，用快慢指针寻找环入口。"],
  ]},
];

const categories = rawCategories.map((group) => ({
  ...group,
  problems: group.items.map(([id, title, slug, difficulty, focus], order) => ({ id, title, slug, difficulty, focus, category: group.name, order })),
}));
const seedProblems: Problem[] = categories.flatMap((group) => group.problems);

const lessonLibrary: Lesson[] = [
  { id:"baidu-recsys-interview", title:"百度推荐策略部：定制面试作战室", category:"面试表达", description:"基于简历与转正答辩材料整理的预测题库、项目风险审计、自我介绍、项目讲稿与 12 题计时模拟面试。", href:"/lessons/baidu-recsys-interview.html", depth:"定制模拟面试", source:"简历、转正答辩与公开业务资料" },
  { id:"defense-script-qa", title:"转正答辩：逐页讲稿与 40 个追问", category:"面试表达", description:"对应 29 页答辩 PPT 的逐页话术、自然转场、时间分配、评委高概率追问、建议回答与五条表达红线。", href:"/lessons/defense-script-qa.html", depth:"18 分钟逐页演练", source:"凌浩然转正答辩 PPT" },
  { id:"lr-best-structures", title:"两个 LR 的最佳结构：为什么有效", category:"工业 LR", description:"统一讲清短视频六路序列 Self-Attention 与货架 ESCM²-DR-SNIPS：改动位置、数学机制、失败路径、证据边界与答辩表达。", href:"/lessons/two-lr-best-structures.html", depth:"优化结构精讲", source:"两份 LR、对应模型与转正答辩" },
  { id:"paper-map", title:"推荐模型内部诊断：论文方法地图", category:"专题入口", description:"按归因、剪枝、Shuffle、多任务冲突和推荐场景方法组织 12 篇论文精读与证据边界。", href:"/lessons/paper-methods-map.html", depth:"12 篇方法地图", source:"论文调研对话" },
  { id:"agent-research-map", title:"Agent-as-MLE × Auto Research", category:"专题入口", description:"用状态—动作—评估—记忆统一 7 个代表方法，逐篇拆解设计思想、数学原理、Memory 结构与证据边界。", href:"/lessons/agent-mle-auto-research.html", depth:"7 篇论文精读", source:"Agent Research 方法调研" },
  { id:"auto-research-framework", title:"从零理解 RecHarness 与 RSI", category:"Harness 设计", description:"面向零基础读者讲清 Agent、Harness、RSI 双循环、三个 Codex Skills、长期 Memory、诊断算法与工具演化门槛。", href:"/lessons/auto-research-framework.html", depth:"框架原理与交互案例", source:"rec-harness 当前三 Skill 设计" },
  { id:"hot100-14", title:"14 天刷完 Hot 100", category:"算法", description:"按天拆分高频模式、复习节奏与验收标准，适合配合当前题单执行。", href:"/lessons/hot100-14days.html", depth:"14 天计划", source:"LeetCode 对话整理" },
  { id:"lr2", title:"LR2：ReAct × Memory × ESCM²", category:"工业 LR", description:"从原理侧复盘货架精排的人机协作、选择偏差、SNIPS、梯度隔离与 batch 充分性。", href:"/lessons/lr2-react-memory-escm2.html", depth:"原理讲解", source:"货架 LR 讲解对话" },
  { id:"three-projects", title:"RecHarness × EviSearch × OPD", category:"项目总览", description:"把三个项目的研究问题、方法链路、工业价值与面试表达放到同一套框架中。", href:"/lessons/three-projects.html", depth:"三项目串讲", source:"转正答辩对话" },
  { id:"defense-manual", title:"转正答辩备战手册", category:"面试表达", description:"覆盖项目故事线、关键指标、追问边界与高压问答，适合答辩和面试前复习。", href:"/lessons/defense-manual.html", depth:"问答手册", source:"转正答辩对话" },
  { id:"harness-opd-evi", title:"长程 Harness · OPD · EviHarness", category:"Harness 设计", description:"以网页演示方式串联长程实验闭环、OPD 训练与可验证搜索 Harness。", href:"/lessons/harness-opd-evi.html", depth:"完整演示", source:"答辩 HTML 对话" },
  { id:"miso", title:"MISO 论文精读：面向 LLM 背景", category:"论文方法", description:"用 LLM 读者熟悉的路由、表征与优化视角解释模块内部状态诊断。", href:"/lessons/miso-paper.html", depth:"论文精读", source:"MISO 精读对话" },
  { id:"miso-primitives", title:"MISO 三类原语：公式与模型实例", category:"论文方法", description:"用一个实际小模型推导 Ranking、Alignment、Comparison，并给出复现顺序。", href:"/lessons/miso-primitives.html", depth:"公式附录", source:"MISO 公式讲解" },
  { id:"evisearch", title:"EviSearch-RL 与长程信用分配", category:"Agentic RL", description:"从 Evidence State、可验证动作空间到 SFT/GRPO，解释搜索智能体的训练闭环。", href:"/lessons/evisearch-rl.html", depth:"完整教程", source:"EviSearch 对话" },
];

const seedKnowledge: Knowledge[] = [
  { id:"recharness", title:"RecHarness", category:"Harness 设计", subtitle:"Evidence-gated RSI · Three Codex Skills", summary:"把框架无关诊断、长期研究记忆和新工具验证组合为双循环，使模型优化与研究能力一起进化。", details:["Diagnostic Toolkit 把 Embedding、Hook 数组与配对干预表转成可审计证据。","Harness Updater 用 Episode、Failure、Lesson 与 Strategy 保存多轮学习。","Tool Evolver 从合格信息缺口出发，经过六级验证后才把新方法晋升稳定工具集。"], metrics:["3 个公开 Skills","2 个 RSI 循环","6 级工具验证门"], recall:"为什么诊断、Memory 更新和工具演化必须拆成三种独立职责？", lessonIds:["auto-research-framework","agent-research-map","three-projects","harness-opd-evi","defense-manual"] },
  { id:"short-video", title:"短视频精排：序列内交互", category:"推荐系统", subtitle:"Self-Attention · item×item · Share-bottom", summary:"最终瓶颈不是聚合后容量不足，而是原始行为序列缺少 item×item 二阶交互。", details:["在序列聚合前加入 Self-Attention，再接入原有排序主干。","共享主干与任务保持不变，把归因集中到序列内部交互。"], metrics:["离线 AUC +0.09pp","线上业务指标正向"], recall:"为什么增强聚合后表征无效，而聚合前的 Self-Attention 有效？", lessonIds:["defense-script-qa","lr-best-structures","harness-opd-evi","defense-manual"] },
  { id:"shelf-ranking", title:"货架电商：ESCM²-DR-SNIPS", category:"推荐系统", subtitle:"选择偏差 · 梯度隔离 · 样本充分性", summary:"方法一度因小 batch 被误判失败；检查正例充分性后调整训练统计条件，修正了实验结论。", details:["使用 SNIPS 稳定反事实权重。","隔离辅助目标对共享表征的梯度干扰。","通过样本充分性检查避免误判算法方向。"], metrics:["离线 AUC +0.29pp","线上业务指标正向"], recall:"这个项目中，真正改变实验结论的“未验证前提”是什么？", lessonIds:["defense-script-qa","lr-best-structures","lr2","defense-manual"] },
  { id:"recmodel-kit", title:"RecModel Kit", category:"模型诊断", subtitle:"Activation × Gradient · Paired Shuffle", summary:"从只看 AUC 扩展到模块依赖与表示冗余，用 AXG 排名提出候选，再用配对 Shuffle 做干预验证。", details:["按 activation、gradient 与 AXG 筛选候选模块。","使用配对 Shuffle/Zero 验证模型依赖。","比较损失、预测变化和置信区间。"], metrics:["Top > Random > Bottom","诊断结果可审计"], recall:"AXG 排名为什么还需要配对 Shuffle 验证？两者分别回答什么问题？", lessonIds:["paper-map","miso-primitives"] },
  { id:"opd2", title:"OPD² 与 On-Policy Distillation", category:"Agentic RL", subtitle:"Policy shift · Teacher token policy", summary:"让 Teacher 适配 Student 的 on-policy 采样分布，避免离线教师在学生轨迹上失配。", details:["比较 Vanilla OPD、Relay-OPD 与 OPD²。","用 ΔT = log πT - log πT₀ 显式获取 Teacher policy shift。","持续关注效果是否依赖 Student checkpoint 与 Teacher loss 地形。"], metrics:["mean@K 36.63% → 62.56%","pass@1 36.23% → 64.58%","8×H800 · 100 steps"], recall:"OPD² 解决的 distribution mismatch 是什么？Teacher 为什么要随 Student 轨迹调整？", lessonIds:["three-projects","harness-opd-evi","defense-manual"] },
  { id:"search-harness", title:"EviSearch-RL / Search Harness", category:"Agentic RL", subtitle:"Evidence State · SFT · GRPO", summary:"将检索过程显式化为 SEARCH / OPEN / KEEP / VERIFY / ANSWER，使证据保留、验证和回答可以被训练与审计。", details:["结合 BM25 与 Dense/RRF 检索，并把已确认事实写入 Evidence State。","构造多跳问答 Recovery 数据，约束 legal action space 与 Go-Gate。","训练链路覆盖 SFT、Retriever、FSDP Actor、vLLM Rollout 与组内 GRPO。"], metrics:["6,000 正式样本","8×A800 FSDP Actor","4×H800 vLLM Rollout"], recall:"把 Evidence State 独立出来，相比让模型直接生成答案，有哪三点收益？", lessonIds:["agent-research-map","evisearch","three-projects","harness-opd-evi"] },
  { id:"geo-agent", title:"Shopee SEO / GEO Agent", category:"搜索与 Agent", subtitle:"Query Fan-out · RAG · Fact Checker", summary:"围绕 SERP Top 10 模拟搜索意图，以 Review/Sales Prompt、事实校验和 RAG 形成可上线的 List Page Agent。", details:["Query Fan-out 扩展真实搜索意图与页面覆盖。","Fact Checker 降低生成式页面中的事实风险。","GEO 评估关注生成式搜索中的召回与内容可引用性。"], metrics:["1,600+ 真实查询","Recall@5 > 0.75","内容生产效率 +70%"], recall:"SEO Agent 与 GEO 评估的目标函数有什么不同？如何设计离线验证？" },
  { id:"long-harness", title:"长程 Self-evolving Harness", category:"未来规划", subtitle:"Rec-Skill · Execution Evidence · Replay", summary:"把模型诊断能力沉淀成可调用的 Rec-Skill，再用真实实验轨迹更新 Skill、Routing、Memory 与恢复规则。", details:["Execution Evidence 保存任务、假设、代码 diff、配置、日志、指标与失败原因。","Harness 更新器只提出局部 Patch，并通过回放对照与人审验收。","人工约束覆盖实验预算、更新准入、回滚策略与上线门槛。"], metrics:["轨迹 → Patch → 回放 → 验收","成功与失败都入库","局部更新可回滚"], recall:"什么样的 Harness 更新才允许进入下一版本？为什么必须有 replay gate？", lessonIds:["agent-research-map","harness-opd-evi","three-projects","defense-manual"] },
];

const today = () => new Date().toISOString().slice(0, 10);
const addDays = (base: string, days: number) => { const d = new Date(`${base}T12:00:00`); d.setDate(d.getDate() + days); return d.toISOString().slice(0, 10); };

export default function Home() {
  const [tab, setTab] = useState<Tab>("hot");
  const [repeatOnly, setRepeatOnly] = useState(false);
  const [reviewOnly, setReviewOnly] = useState(false);
  const [category, setCategory] = useState("全部");
  const [search, setSearch] = useState("");
  const [progress, setProgress] = useState<Record<number, ProblemProgress>>({});
  const [knowledgeProgress, setKnowledgeProgress] = useState<Record<string, KnowledgeProgress>>({});
  const [customKnowledge, setCustomKnowledge] = useState<Knowledge[]>([]);
  const [customProblems, setCustomProblems] = useState<Problem[]>([]);
  const [activeProblem, setActiveProblem] = useState<Problem | null>(null);
  const [activeKnowledge, setActiveKnowledge] = useState<string | null>(seedKnowledge[0].id);
  const [problemNotes, setProblemNotes] = useState("");
  const [showAdd, setShowAdd] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [importText, setImportText] = useState("");
  const [importing, setImporting] = useState(false);
  const [celebration, setCelebration] = useState(0);
  const [newCard, setNewCard] = useState({ title:"", category:"自定义", summary:"", recall:"" });
  const [hydrated, setHydrated] = useState(false);
  const [toast, setToast] = useState("");
  const [libraryOpen, setLibraryOpen] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const hydrationTask = window.setTimeout(() => {
      try {
        setProgress(JSON.parse(localStorage.getItem("lings-lab-problems") || "{}"));
        setKnowledgeProgress(JSON.parse(localStorage.getItem("lings-lab-knowledge-progress") || "{}"));
        setCustomKnowledge(JSON.parse(localStorage.getItem("lings-lab-custom-knowledge") || "[]"));
        setCustomProblems(JSON.parse(localStorage.getItem("lings-lab-custom-problems") || "[]"));
      } catch { /* start clean if local data is malformed */ }
      setHydrated(true);
    }, 0);
    return () => window.clearTimeout(hydrationTask);
  }, []);
  useEffect(() => { if (hydrated) localStorage.setItem("lings-lab-problems", JSON.stringify(progress)); }, [progress, hydrated]);
  useEffect(() => { if (hydrated) localStorage.setItem("lings-lab-knowledge-progress", JSON.stringify(knowledgeProgress)); }, [knowledgeProgress, hydrated]);
  useEffect(() => { if (hydrated) localStorage.setItem("lings-lab-custom-knowledge", JSON.stringify(customKnowledge)); }, [customKnowledge, hydrated]);
  useEffect(() => { if (hydrated) localStorage.setItem("lings-lab-custom-problems", JSON.stringify(customProblems)); }, [customProblems, hydrated]);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(""), 2200); return () => window.clearTimeout(timer); }, [toast]);
  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setTab("hot");
        window.setTimeout(() => searchInputRef.current?.focus(), 0);
      }
      if (event.key === "Escape" && !importing) {
        setActiveProblem(null);
        setShowImport(false);
        setShowAdd(false);
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [importing]);

  const allKnowledge = [...seedKnowledge, ...customKnowledge];
  const allProblems = useMemo(() => [...seedProblems, ...customProblems], [customProblems]);
  const categoryNames = useMemo(() => customProblems.length ? [...categories.map((group) => group.name), "自定义导入"] : categories.map((group) => group.name), [customProblems.length]);
  const solved = allProblems.filter((p) => (progress[p.id]?.times || 0) > 0).length;
  const repeatCount = allProblems.filter((p) => progress[p.id]?.repeat).length;
  const dueKnowledge = allKnowledge.filter((k) => knowledgeProgress[k.id]?.due && knowledgeProgress[k.id].due! <= today());
  const learnedKnowledge = allKnowledge.filter((k) => (knowledgeProgress[k.id]?.reviews || 0) > 0).length;

  const filteredProblems = useMemo(() => allProblems.filter((p) => {
    const matchesCategory = category === "全部" || p.category === category;
    const q = search.trim().toLowerCase();
    const matchesSearch = !q || `${p.id}${p.title}${p.category}`.toLowerCase().includes(q);
    const matchesRepeat = !repeatOnly || progress[p.id]?.repeat;
    return matchesCategory && matchesSearch && matchesRepeat;
  }), [allProblems, category, search, repeatOnly, progress]);

  const currentGroup = categories.find((g) => g.problems.some((p) => !progress[p.id]?.times)) || categories[0];
  const visibleKnowledge = reviewOnly ? dueKnowledge : allKnowledge;

  const recordSolve = (problem: Problem) => {
    setProgress((old) => ({ ...old, [problem.id]: { times:(old[problem.id]?.times || 0) + 1, repeat:old[problem.id]?.repeat || false, notes:old[problem.id]?.notes || "", lastPracticed:today() } }));
    setCelebration(Date.now());
    window.setTimeout(() => setCelebration(0), 950);
    setToast(`${problem.title}：刷题次数 +1`);
  };
  const trackPointer = (event: ReactPointerEvent<HTMLElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    event.currentTarget.style.setProperty("--mx", `${event.clientX - rect.left}px`);
    event.currentTarget.style.setProperty("--my", `${event.clientY - rect.top}px`);
  };
  const toggleRepeat = (problem: Problem) => setProgress((old) => ({ ...old, [problem.id]: { times:old[problem.id]?.times || 0, repeat:!old[problem.id]?.repeat, notes:old[problem.id]?.notes || "", lastPracticed:old[problem.id]?.lastPracticed } }));
  const openProblem = (problem: Problem) => { setActiveProblem(problem); setProblemNotes(progress[problem.id]?.notes || ""); };
  const saveProblemNotes = () => {
    if (!activeProblem) return;
    setProgress((old) => ({ ...old, [activeProblem.id]: { times:old[activeProblem.id]?.times || 0, repeat:old[activeProblem.id]?.repeat || false, notes:problemNotes, lastPracticed:old[activeProblem.id]?.lastPracticed } }));
    setActiveProblem(null); setToast("学习讲解已保存");
  };

  const scheduleKnowledge = (id: string) => {
    setKnowledgeProgress((old) => ({ ...old, [id]: { level:1, interval:1, due:addDays(today(), 1), reviews:(old[id]?.reviews || 0) + 1, notes:old[id]?.notes || "" } }));
    setToast("已加入明日复习队列");
  };
  const reviewKnowledge = (id: string, grade: "forgot" | "fuzzy" | "remember") => {
    setKnowledgeProgress((old) => {
      const current = old[id] || { level:0, interval:1, reviews:0, notes:"" };
      const interval = grade === "forgot" ? 1 : grade === "fuzzy" ? Math.max(2, Math.round(current.interval * 1.8)) : Math.max(3, Math.round(current.interval * 2.7));
      const level = grade === "forgot" ? 1 : Math.min(5, current.level + 1);
      return { ...old, [id]: { ...current, level, interval, due:addDays(today(), interval), reviews:current.reviews + 1 } };
    });
    setToast(grade === "forgot" ? "明天再见一次" : grade === "fuzzy" ? "已安排短间隔复习" : "记忆间隔已拉长");
  };
  const updateKnowledgeNote = (id: string, notes: string) => setKnowledgeProgress((old) => ({ ...old, [id]: { level:old[id]?.level || 0, interval:old[id]?.interval || 1, due:old[id]?.due, reviews:old[id]?.reviews || 0, notes } }));
  const importProblemIds = async (event: FormEvent) => {
    event.preventDefault();
    const linkPattern = /https?:\/\/(?:www\.)?leetcode\.(?:cn|com)\/problems\/[^\s,，]+/gi;
    const links = Array.from(new Set((importText.match(linkPattern) || []).map((link) => link.replace(/[)\]}>。；;、]+$/, ""))));
    const numberOnlyText = importText.replace(linkPattern, " ");
    const ids = Array.from(new Set((numberOnlyText.match(/\d+/g) || []).map(Number).filter((id) => id > 0 && id < 100000)));
    if (!ids.length && !links.length) { setToast("没有识别到题号或题目链接"); return; }

    setImporting(true);
    try {
      let parsed: ParsedQuestion[] = [];
      if (links.length) {
        const response = await fetch("/api/leetcode", {
          method:"POST",
          headers:{ "content-type":"application/json" },
          body:JSON.stringify({ links }),
        });
        if (!response.ok) throw new Error("metadata request failed");
        const payload = await response.json() as { results?: ParsedQuestion[] };
        parsed = payload.results || [];
      }

      const difficultyMap: Record<string, Difficulty> = { Easy:"简单", Medium:"中等", Hard:"困难" };
      const fromLinks: Problem[] = parsed.filter((item): item is ParsedQuestion & { id: number; title: string; slug: string; href: string } => Boolean(item.id && item.title && item.slug && item.href)).map((item, index) => ({
        id:item.id,
        title:item.title,
        slug:item.slug,
        difficulty:difficultyMap[item.difficulty || ""] || "待标注",
        focus:"先补充题意与暴力解法，再写核心状态、数据结构、复杂度、易错边界和同类题迁移。",
        category:"自定义导入",
        order:customProblems.length + index,
        imported:true,
        href:item.href,
      }));
      const fromIds: Problem[] = ids.map((id, index) => ({
        id,
        title:"待补充题名",
        slug:"",
        difficulty:"待标注",
        focus:"先补充题意与暴力解法，再写核心状态、数据结构、复杂度、易错边界和同类题迁移。",
        category:"自定义导入",
        order:customProblems.length + fromLinks.length + index,
        imported:true,
        href:`https://leetcode.cn/problemset/?search=${id}`,
      }));
      const candidates = [...fromLinks, ...fromIds];
      const existingIds = new Set(allProblems.map((problem) => problem.id));
      const seen = new Set<number>();
      const imported = candidates.filter((problem) => !existingIds.has(problem.id) && !seen.has(problem.id) && seen.add(problem.id));
      const metadataById = new Map(fromLinks.map((problem) => [problem.id, problem]));
      const upgradedIds = new Set(customProblems.filter((problem) => metadataById.has(problem.id) && (problem.title === "待补充题名" || problem.difficulty === "待标注" || !problem.slug)).map((problem) => problem.id));
      const failed = parsed.filter((item) => item.error).length;
      const duplicates = Math.max(0, candidates.length - imported.length - upgradedIds.size);

      if (!imported.length && !upgradedIds.size) {
        setToast(failed ? `${failed} 个链接解析失败，其余题目已在题单中` : "这些题目已在题单中");
        return;
      }
      setCustomProblems((old) => [
        ...old.map((problem) => upgradedIds.has(problem.id) ? { ...problem, ...metadataById.get(problem.id), order:problem.order } : problem),
        ...imported,
      ]);
      setCategory("自定义导入");
      setRepeatOnly(false);
      setSearch("");
      setImportText("");
      setShowImport(false);
      setToast(`${imported.length ? `已导入 ${imported.length} 道题` : "已补全题目信息"}${upgradedIds.size ? `，更新 ${upgradedIds.size} 道标题与难度` : ""}${duplicates ? `，跳过 ${duplicates} 道重复题` : ""}${failed ? `，${failed} 个链接解析失败` : ""}`);
    } catch {
      setToast("暂时无法解析链接，请稍后重试或直接输入题号");
    } finally {
      setImporting(false);
    }
  };
  const removeImportedProblem = (problem: Problem) => {
    setCustomProblems((old) => old.filter((item) => item.id !== problem.id));
    if (customProblems.length === 1) setCategory("全部");
    setActiveProblem(null);
    setToast(`已移除 LeetCode #${problem.id}`);
  };
  const addCard = (event: FormEvent) => {
    event.preventDefault(); if (!newCard.title.trim()) return;
    const card: Knowledge = { id:`custom-${Date.now()}`, title:newCard.title.trim(), category:newCard.category.trim() || "自定义", subtitle:"自定义知识卡", summary:newCard.summary.trim() || "在这里补充核心结论。", details:[], metrics:[], recall:newCard.recall.trim() || "尝试不看笔记，用自己的话复述这张卡片。" };
    setCustomKnowledge((old) => [...old, card]); setActiveKnowledge(card.id); setShowAdd(false); setNewCard({ title:"", category:"自定义", summary:"", recall:"" }); setToast("知识卡已加入");
  };

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">L</span><span>Ling&apos;s Lab</span></div>
        <nav aria-label="主导航">
          <button className={`nav-item ${tab === "hot" ? "active" : ""}`} onClick={() => { setTab("hot"); setRepeatOnly(false); }}><span>⌁</span><b>Hot 100</b><small>{solved}/{allProblems.length}</small></button>
          <button className={`nav-item ${tab === "projects" && !reviewOnly ? "active" : ""}`} onClick={() => { setTab("projects"); setReviewOnly(false); setLibraryOpen(false); }}><span>▤</span><b>项目知识库</b><small>{allKnowledge.length} 张</small></button>
          <button className={`nav-item ${tab === "projects" && reviewOnly ? "active" : ""}`} onClick={() => { setTab("projects"); setReviewOnly(true); setLibraryOpen(false); }}><span>◴</span><b>今日复习</b><small>{dueKnowledge.length}</small></button>
        </nav>
        <div className="sidebar-note"><span>CURRICULUM</span><p>由你的转正答辩与个人简历生成，可继续扩充。</p></div>
        <div className="profile-card"><div className="avatar">LH</div><div><strong>凌浩然</strong><small>Agentic RL · Harness · LLM</small></div></div>
      </aside>

      <section className="workspace">
        <header className="topbar pointer-glow" onPointerMove={trackPointer}>
          <div className="title-block"><p className="eyebrow">LING&apos;S LEARNING SYSTEM</p><h1>{tab === "hot" ? "把算法模式，练成直觉。" : reviewOnly ? "在遗忘之前，再见一次。" : "把项目经验，沉淀成能力。"}</h1><p className="header-subtitle">{tab === "hot" ? "按模式建立解题框架，用复刷记录守住薄弱环节。" : "把答辩、简历和项目复盘，变成一套可以反复调用的知识系统。"}</p></div>
          <div className="header-actions"><button className="focus-button" onClick={() => { setTab("projects"); setReviewOnly(true); setLibraryOpen(false); }}><span>今日复习</span><b>{dueKnowledge.length}</b></button><div className="avatar">LH</div></div>
        </header>

        <div className="stats-row">
          <div className="stat primary pointer-glow" onPointerMove={trackPointer}><div><span>当前题单总进度</span><strong>{String(solved).padStart(2,"0")}<small>/{allProblems.length}</small></strong></div><div className="progress-copy"><small>距离完成当前题单还有 {allProblems.length-solved} 题</small><div className="progress"><i style={{width:`${allProblems.length ? solved / allProblems.length * 100 : 0}%`}} /></div></div></div>
          <button className="stat stat-button pointer-glow" onPointerMove={trackPointer} onClick={() => { setTab("hot"); setRepeatOnly(true); }}><span>需要复刷</span><strong>{String(repeatCount).padStart(2,"0")}</strong><small>打开薄弱题单 <i>↗</i></small></button>
          <button className="stat stat-button pointer-glow" onPointerMove={trackPointer} onClick={() => { setTab("projects"); setReviewOnly(true); setLibraryOpen(false); }}><span>今日到期</span><strong>{String(dueKnowledge.length).padStart(2,"0")}</strong><small>进入记忆复习 <i>↗</i></small></button>
          <div className="stat pointer-glow" onPointerMove={trackPointer}><span>知识循环</span><strong>{learnedKnowledge}<small>/{allKnowledge.length}</small></strong><small>已开始间隔复习</small></div>
        </div>

        {tab === "hot" ? (
          <section className="hot-layout">
            <div className="sequence-panel panel pointer-glow" onPointerMove={trackPointer}>
              <div><p className="eyebrow">NEXT PATTERN · {String(categories.indexOf(currentGroup)+1).padStart(2,"0")}</p><h2>{currentGroup.name}</h2><p>{currentGroup.logic}</p></div>
              <div className="sequence-track">{currentGroup.problems.map((p, i) => <button key={p.id} className={progress[p.id]?.times ? "done" : i === 0 ? "current" : ""} onClick={() => openProblem(p)}><span>{String(i+1).padStart(2,"0")}</span>{p.title}</button>)}</div>
            </div>

            <div className="problem-panel panel">
              <div className="panel-head">
                <div><p className="eyebrow">LEETCODE HOT 100</p><h2>{repeatOnly ? "复刷清单" : category === "全部" ? "同类题顺序" : category}</h2></div>
                <div className="filters"><button className="import-button" onClick={() => setShowImport(true)}>＋ 导入题目</button><label className="search"><span>⌕</span><input ref={searchInputRef} aria-label="搜索题目" placeholder="题号 / 题名 / 类型" value={search} onChange={(e) => setSearch(e.target.value)} /><kbd>⌘K</kbd></label>{repeatOnly && <button className="chip active" onClick={() => setRepeatOnly(false)}>仅复刷 ×</button>}</div>
              </div>
              <div className="category-strip" aria-label="题型筛选">{["全部",...categoryNames].map((name) => <button key={name} className={category === name ? "active" : ""} onClick={() => setCategory(name)}>{name}</button>)}</div>
              <div className="table-head"><span>顺序 / 题目</span><span>难度</span><span>刷题次数</span><span>复刷</span><span>操作</span></div>
              <div className="problem-list">
                {filteredProblems.map((p) => {
                  const state = progress[p.id];
                  return <article className="problem-row" key={p.id}>
                    <button className="problem-title" onClick={() => openProblem(p)}><b>{String(p.order+1).padStart(2,"0")}</b><div><strong>{p.id}. {p.title}</strong><small>{p.category} · {state?.notes ? "已有学习讲解" : "补充学习讲解"}</small></div></button>
                    <span className={`difficulty d-${p.difficulty}`}>{p.difficulty}</span>
                    <span className="times"><b>{state?.times || 0}</b> 次{state?.lastPracticed && <small>{state.lastPracticed.slice(5)}</small>}</span>
                    <button aria-label={`${p.title}标记复刷`} className={`repeat-toggle ${state?.repeat ? "on" : ""}`} onClick={() => toggleRepeat(p)}><i />{state?.repeat ? "需要" : "不用"}</button>
                    <button className="record" onClick={() => recordSolve(p)}>+ 完成一次</button>
                  </article>;
                })}
                {!filteredProblems.length && <div className="empty"><strong>没有符合条件的题目</strong><p>换个关键词，或取消“仅复刷”筛选。</p></div>}
              </div>
              <footer className="list-footer"><span>显示 {filteredProblems.length} / {allProblems.length} 道题{customProblems.length ? ` · 已导入 ${customProblems.length} 道` : ""}</span><div><a href="/lessons/hot100-14days.html" target="_blank" rel="noreferrer">14 天讲解计划 ↗</a><a href="https://leetcode.cn/studyplan/top-100-liked/" target="_blank" rel="noreferrer">力扣官方题单 ↗</a></div></footer>
            </div>
          </section>
        ) : (
          <section className="knowledge-layout">
            <aside className="knowledge-index panel">
              <div className="index-head"><div><p className="eyebrow">KNOWLEDGE MAP</p><h2>{reviewOnly ? "今日到期" : "项目主线"}</h2></div><button className="add-button" onClick={() => setShowAdd(true)} aria-label="新增知识卡">＋</button></div>
              {!reviewOnly && <button className={`library-link ${libraryOpen ? "active" : ""}`} onClick={() => setLibraryOpen(true)}><span>HTML</span><div><strong>深度讲解资料馆</strong><small>{lessonLibrary.length} 篇 · 来自历史对话整理</small></div><i>→</i></button>}
              {!visibleKnowledge.length ? <div className="empty compact"><strong>今天没有到期卡片</strong><p>先学习一张项目卡，系统会从明天开始安排复习。</p><button className="solid" onClick={() => setReviewOnly(false)}>浏览知识库</button></div> : visibleKnowledge.map((k, index) => {
                const state = knowledgeProgress[k.id];
                return <button key={k.id} className={`knowledge-link ${!libraryOpen && activeKnowledge === k.id ? "active" : ""}`} onClick={() => { setActiveKnowledge(k.id); setLibraryOpen(false); }}><span>{String(index+1).padStart(2,"0")}</span><div><strong>{k.title}</strong><small>{k.category} · {state?.reviews || 0} 次复习</small></div><i>{state?.due && state.due <= today() ? "到期" : state?.level ? `L${state.level}` : "NEW"}</i></button>;
              })}
              {!reviewOnly && <button className="add-row" onClick={() => setShowAdd(true)}>＋ 添加新的项目知识</button>}
            </aside>

            {(() => {
              if (libraryOpen && !reviewOnly) return <article className="lesson-library panel">
                <div className="library-hero"><div><p className="eyebrow">CURATED FROM YOUR CHATS</p><h2>深度讲解 HTML 资料馆</h2><p>已从其他对话生成的资料中去重，按算法、推荐系统、模型诊断、Harness 与 Agentic RL 重新归档。</p></div><strong>{String(lessonLibrary.length).padStart(2,"0")}<small>篇</small></strong></div>
                {Array.from(new Set(lessonLibrary.map((lesson) => lesson.category))).map((lessonCategory) => <section className="lesson-group" key={lessonCategory}><div className="lesson-group-title"><span>{lessonCategory}</span><i /></div><div className="lesson-grid">{lessonLibrary.filter((lesson) => lesson.category === lessonCategory).map((lesson) => <a className="lesson-card pointer-glow" onPointerMove={trackPointer} href={lesson.href} target="_blank" rel="noreferrer" key={lesson.id}><div><span>{lesson.depth}</span><i>HTML ↗</i></div><h3>{lesson.title}</h3><p>{lesson.description}</p><small>{lesson.source}</small></a>)}</div></section>)}
              </article>;
              const selected = visibleKnowledge.find((k) => k.id === activeKnowledge) || visibleKnowledge[0];
              if (!selected) return <div className="knowledge-card panel empty-state"><span>✓</span><h2>今日复习完成</h2><p>间隔复习不是每天堆量，而是在恰好要忘记时出现。</p></div>;
              const state = knowledgeProgress[selected.id];
              const relatedLessons = lessonLibrary.filter((lesson) => selected.lessonIds?.includes(lesson.id));
              return <article className="knowledge-card panel pointer-glow" onPointerMove={trackPointer}>
                <div className="knowledge-hero"><div><p className="eyebrow">{selected.category}</p><h2>{selected.title}</h2><p className="subtitle">{selected.subtitle}</p></div><div className="mastery"><span>记忆强度</span><div>{[1,2,3,4,5].map((n) => <i key={n} className={n <= (state?.level || 0) ? "filled" : ""} />)}</div><small>{state?.reviews ? `已复习 ${state.reviews} 次` : "尚未进入复习循环"}</small></div></div>
                <p className="summary">{selected.summary}</p>
                {!!selected.metrics.length && <div className="metric-row">{selected.metrics.map((m) => <span key={m}>{m}</span>)}</div>}
                {!!selected.details.length && <div className="detail-block"><p className="eyebrow">核心拆解</p>{selected.details.map((d, i) => <div className="detail-line" key={d}><span>{String(i+1).padStart(2,"0")}</span><p>{d}</p></div>)}</div>}
                {!!relatedLessons.length && <div className="related-lessons"><div className="related-head"><div><p className="eyebrow">DEEP DIVE HTML</p><h3>相关知识点讲解</h3></div><button onClick={() => setLibraryOpen(true)}>查看全部 {lessonLibrary.length} 篇 →</button></div><div className="related-grid">{relatedLessons.map((lesson) => <a href={lesson.href} target="_blank" rel="noreferrer" key={lesson.id}><span>{lesson.depth}</span><strong>{lesson.title}</strong><small>{lesson.description}</small><i>打开讲解 HTML ↗</i></a>)}</div></div>}
                <div className="recall-card"><span>ACTIVE RECALL</span><h3>闭眼回答</h3><p>{selected.recall}</p></div>
                <label className="knowledge-notes"><span>我的补充 / 面试追问</span><textarea value={state?.notes || ""} onChange={(e) => updateKnowledgeNote(selected.id, e.target.value)} placeholder="记录容易忘的边界、推导、失败实验或面试追问…" /></label>
                {state?.due && state.due <= today() ? <div className="review-actions"><div><span>这次记得怎么样？</span><small>你的选择会决定下一次出现时间</small></div><button onClick={() => reviewKnowledge(selected.id,"forgot")}>忘记</button><button onClick={() => reviewKnowledge(selected.id,"fuzzy")}>模糊</button><button className="remember" onClick={() => reviewKnowledge(selected.id,"remember")}>记住了</button></div> : <div className="schedule-row"><div><span>{state?.due ? `下次复习：${state.due}` : "学完后，明天会第一次复习"}</span><small>{state?.interval ? `当前间隔 ${state.interval} 天` : "按记忆表现自动拉长间隔"}</small></div><button className="solid" onClick={() => scheduleKnowledge(selected.id)}>{state?.reviews ? "今天额外复习一次" : "学完并安排复习"} →</button></div>}
              </article>;
            })()}
          </section>
        )}
      </section>

      {showImport && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget && !importing) setShowImport(false); }}><form className="modal import-modal" onSubmit={importProblemIds}><button type="button" className="close" disabled={importing} onClick={() => setShowImport(false)}>×</button><p className="eyebrow">EXPAND YOUR PROBLEM SET</p><h2>导入 LeetCode 题目</h2><div className="import-guide"><span>链接自动解析</span><p>粘贴 LeetCode CN 或国际站题目链接，将自动识别题号、标题与难度。也可继续直接输入题号，例如：<code>146, 215, 347</code>。</p></div><label><span>题目链接 / 题号</span><textarea disabled={importing} value={importText} onChange={(e) => setImportText(e.target.value)} placeholder={"https://leetcode.cn/problems/two-sum/\nhttps://leetcode.com/problems/lru-cache/\n或直接输入：215, 347"} /></label><p className="import-tip">支持批量粘贴、自动去重。链接解析完成后，新题会带着真实标题和难度进入“自定义导入”分类。</p><div className="modal-actions right"><button type="button" className="chip" disabled={importing} onClick={() => setShowImport(false)}>取消</button><button className="solid" disabled={importing} type="submit">{importing ? "正在解析…" : "解析并导入"}</button></div></form></div>}

      {activeProblem && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget) setActiveProblem(null); }}><section className="modal" role="dialog" aria-modal="true" aria-label={`${activeProblem.title} 学习讲解`}>
        <button className="close" onClick={() => setActiveProblem(null)}>×</button><p className="eyebrow">{activeProblem.category} · 顺序 {String(activeProblem.order+1).padStart(2,"0")}</p><h2>{activeProblem.id}. {activeProblem.title}</h2>
        <div className="built-in-note"><span>核心抓手</span><p>{activeProblem.focus}</p></div>
        <div className="learning-template"><span>建议讲解结构</span><ol><li>先说暴力解法与瓶颈</li><li>定义状态 / 指针 / 数据结构</li><li>解释为什么不会漏解、复杂度是多少</li><li>补充边界与同类题迁移</li></ol></div>
        <label><span>我的学习讲解</span><textarea value={problemNotes} onChange={(e) => setProblemNotes(e.target.value)} placeholder="用自己的话写：思路、伪代码、易错点、复杂度、复刷原因…" /></label>
        <div className="modal-actions"><a href={activeProblem.href || `https://leetcode.cn/problems/${activeProblem.slug}/`} target="_blank" rel="noreferrer">{activeProblem.imported ? "在力扣搜索此题 ↗" : "打开题目 ↗"}</a>{activeProblem.imported && <button className="remove-problem" onClick={() => removeImportedProblem(activeProblem)}>移除此题</button>}<button className={`chip ${progress[activeProblem.id]?.repeat ? "active" : ""}`} onClick={() => toggleRepeat(activeProblem)}>{progress[activeProblem.id]?.repeat ? "✓ 已标记复刷" : "标记需要复刷"}</button><button className="solid" onClick={saveProblemNotes}>保存讲解</button></div>
      </section></div>}

      {showAdd && <div className="modal-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget) setShowAdd(false); }}><form className="modal add-modal" onSubmit={addCard}><button type="button" className="close" onClick={() => setShowAdd(false)}>×</button><p className="eyebrow">EXPAND YOUR MAP</p><h2>新增项目知识卡</h2><div className="form-grid"><label><span>标题 *</span><input required value={newCard.title} onChange={(e) => setNewCard({...newCard,title:e.target.value})} placeholder="例如：FSDP 通信与显存分析" /></label><label><span>分类</span><input value={newCard.category} onChange={(e) => setNewCard({...newCard,category:e.target.value})} placeholder="工程 / 推荐 / RL" /></label></div><label><span>核心结论</span><textarea value={newCard.summary} onChange={(e) => setNewCard({...newCard,summary:e.target.value})} placeholder="这张卡片最需要记住的一句话…" /></label><label><span>主动回忆问题</span><textarea value={newCard.recall} onChange={(e) => setNewCard({...newCard,recall:e.target.value})} placeholder="不看笔记时，我应该能回答什么？" /></label><div className="modal-actions right"><button type="button" className="chip" onClick={() => setShowAdd(false)}>取消</button><button className="solid" type="submit">加入知识库</button></div></form></div>}
      {celebration > 0 && <div className="celebration" key={celebration} aria-hidden="true">{Array.from({length:12}, (_, index) => <i key={index} />)}</div>}
      {toast && <div className="toast">✓ {toast}</div>}
    </main>
  );
}
