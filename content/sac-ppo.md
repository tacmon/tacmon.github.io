# 像读竞赛题解一样理解 PPO 与 SAC

把强化学习看成一道“只能调用交互接口、不能查看完整地图的最优化题”，PPO 和 SAC 就是两种学习解法。本文按题意、建模、核心观察、更新公式、伪代码和易错点展开。PPO 使用 PPO-Clip；SAC 使用连续动作、双 Q、无需独立 V 网络的版本。

## 1. 题意：在未知环境中取得尽可能高的总分

每一步观察状态 $s_t$，选择动作 $a_t$，环境返回奖励 $r_t$ 和下一状态 $s_{t+1}$。例如控制小车时，状态可以是位置与速度，动作可以是转向角与油门，奖励可以是前进距离减去碰撞惩罚。

环境转移规律未知，只能通过交互收集经验。我们要学习一个策略 $\pi_\theta(a\mid s)$：离散动作时它是概率，连续动作时它是概率密度。

基本目标为：

$$
\max_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right],\qquad 0\le\gamma<1.
$$

与普通竞赛题不同，输入并不完整：算法需要一边收集信息，一边改进决策。

## 2. 基础建模：把价值函数理解为近似 DP 表

定义状态价值和动作价值：

$$
V^\pi(s)=\mathbb E_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s\right],
$$

$$
Q^\pi(s,a)=\mathbb E_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s,a_t=a\right].
$$

$V$ 表示从某状态开始按策略行动的预期得分；$Q$ 表示第一步指定一个动作，之后再按策略行动的预期得分。

Bellman 递推为：

$$
Q^\pi(s,a)=\mathbb E\left[r+\gamma V^\pi(s')\right].
$$

这就是“当前收益 + 后续价值”。由于状态和动作可能连续，不能开数组存完整 DP 表，于是用神经网络拟合。Actor 负责选择动作，Critic 负责估值；两者交替改进。

## 3. PPO：好动作多做一点，但每次不要改得太猛

PPO-Clip 根据新采样的轨迹更新策略，并通过截断代理目标减少过大更新的激励。[PPO 原论文](https://arxiv.org/abs/1707.06347)

### 3.1 核心观察：动作好不好，要与当前状态的平均水平比较

优势函数为：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

若一个状态平均能得 10 分，而选择动作 $a$ 预计能得 13 分，那么优势为 3，应增加它的概率。优势为负则应降低概率。实际训练使用采样得到的估计 $\hat A_t$。

### 3.2 用概率比表示新旧策略的变化

采样策略固定为 $\pi_{\mathrm{old}}$，待更新策略为 $\pi_\theta$。定义：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

若旧概率为 0.2，新概率为 0.3，则比值为 1.5。最大化 $\mathbb E_t[\rho_t\hat A_t]$ 会倾向于增加正优势动作的概率、减少负优势动作的概率。但有限样本有噪声，反复优化同一批数据可能改过头。

### 3.3 关键优化：截断继续改进的收益

PPO-Clip 最大化：

$$
L^{\mathrm{clip}}(\theta)=\mathbb E_t\left[\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

按优势的正负分类讨论，单个样本的目标为：

$$
\ell_t=
\begin{cases}
\hat A_t\min(\rho_t,1+\epsilon),&\hat A_t\ge0,\\
\hat A_t\max(\rho_t,1-\epsilon),&\hat A_t<0.
\end{cases}
$$

好动作的概率增加到一定程度后，不再给额外收益；坏动作的概率降低到一定程度后，也不再给额外收益。若变化方向错误，则仍保留纠正它的激励。

取 $\epsilon=0.2$、$\hat A=2$：

| 概率比 | 普通目标 | PPO 截断目标 |
| --- | --- | --- |
| 1.0 | 2.0 | 2.0 |
| 1.1 | 2.2 | 2.2 |
| 1.5 | 3.0 | 2.4 |

截断的是目标中的收益，并非硬性限制真实概率比。共享网络参数、其他样本及多次更新仍可能使概率比超出区间，所以它不保证新旧策略距离一定很小。[PPO 官方讲解](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### 3.4 优势估计：一次倒序扫描计算 GAE

令 $b_t$ 表示是否允许从下一状态 bootstrap，定义 TD 误差：

$$
\delta_t=r_t+\gamma b_tV_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t).
$$

GAE 可以倒序递推：

$$
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

$c_t$ 表示是否能继续连接到同一轨迹的下一条采样记录。真正终止时 $b_t=c_t=0$；仅因时间上限截断时，通常允许用最后状态 bootstrap，但不能递推到重置后的新回合。采样段末尾也应停止优势递推。

$\lambda$ 控制多步 TD 信息的权重。价值回归目标可以取：

$$
\hat R_t=\hat A_t+V_{\mathrm{old}}(s_t),\qquad
L_V(\phi)=\mathbb E_t[(V_\phi(s_t)-\hat R_t)^2].
$$

更新时固定优势和回归目标。若要标准化优势，应先用原始优势构造回归目标。

### 3.5 伪代码

```text
初始化策略 πθ 和价值网络 Vφ
重复：
    用当前策略收集 N 步数据
    保存状态、动作、奖励、边界标记、old_log_prob、old_value
    倒序计算优势 A
    R = A + old_value

    重复 K 轮：
        打乱数据并划分小批次
        对每个小批次：
            ratio = exp(new_log_prob - old_log_prob)
            actor_loss = -mean(min(ratio*A, clip(ratio,1-ε,1+ε)*A))
            critic_loss = mean((Vφ(s)-R)^2)
            更新 Actor 和 Critic

    丢弃本批数据，重新采样
```

PPO 属于 on-policy 方法：同一批新数据可以训练多轮，但通常不长期复用旧轨迹。

## 4. SAC：学一张动作价值表，并保留有价值的探索

SAC 将熵纳入学习目标，并使用经验回放。这里采用双 Q 和目标 Q 网络的连续动作版本。[SAC 论文](https://arxiv.org/html/1812.05905v2)

### 4.1 核心观察：不要过早认定某个动作最好

若当前估计两个动作的价值分别为 10 和 9.9，而估值本身还不准确，始终选择前者可能错失后者的真实价值。SAC 在奖励之外加入熵：

$$
J(\pi)=\mathbb E_\pi\left[\sum_{t=0}^{\infty}\gamma^t\left(r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))\right)\right],
$$

$$
\mathcal H(\pi(\cdot\mid s))=-\mathbb E_{a\sim\pi}[\log\pi(a\mid s)].
$$

$\alpha>0$ 控制熵与奖励的相对权重。离散动作的熵反映分布的分散程度；连续动作使用微分熵，其值可以为负。

### 4.2 从 DP 推导 Soft：把 max 换成平滑的选择

先考虑有限个动作，令动作概率为 $p_i$。固定 Q 后，需要求解：

$$
\max_{p_i}\left[\sum_i p_iQ_i-\alpha\sum_i p_i\log p_i\right],\qquad p_i\ge0,\quad\sum_i p_i=1.
$$

引入拉格朗日乘子 $\eta$，求导得到：

$$
Q_i-\alpha(\log p_i+1)+\eta=0.
$$

因此：

$$
p_i=\frac{\exp(Q_i/\alpha)}{\sum_j\exp(Q_j/\alpha)}.
$$

代回最优值为：

$$
V(s)=\alpha\log\sum_a\exp(Q(s,a)/\alpha).
$$

硬性的最大值变成 log-sum-exp。$\alpha$ 小时更偏向最高价值动作，较大时分布更分散。连续动作下求和变为积分，通常难以直接求解，于是用 Actor 近似这种策略。

### 4.3 Critic 更新：带熵的 Bellman 递推

对给定策略，定义：

$$
V^\pi_{\mathrm{soft}}(s)=\mathbb E_{a\sim\pi}[Q^\pi_{\mathrm{soft}}(s,a)-\alpha\log\pi(a\mid s)].
$$

soft Q 的递推为 $Q^\pi_{\mathrm{soft}}(s,a)=\mathbb E[r+\gamma V^\pi_{\mathrm{soft}}(s')]$。这里的 Q 包含未来的熵收益，与只累计环境奖励的普通 Q 不同。

从经验池抽取 $(s,a,r,s',d)$，用当前策略采样 $a'\sim\pi_\theta(\cdot\mid s')$，构造：

$$
y=r+\gamma(1-d)\left[\min_{i=1,2}Q_{\bar\phi_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right].
$$

$d$ 表示真正终止。固定目标 $y$，分别最小化：

$$
L_{Q_i}=\mathbb E[(Q_{\phi_i}(s,a)-y)^2].
$$

取两个 Q 的较小值有助于缓解高估，但不是对真实价值的严格下界。目标网络通过缓慢跟随在线网络减少目标波动：

$$
\bar\phi_i\leftarrow(1-\tau)\bar\phi_i+\tau\phi_i.
$$

### 4.4 Actor 更新：追求高价值，也保留熵

状态从经验池抽取，动作则由当前策略重新生成。最小化：

$$
L_\pi(\theta)=\mathbb E_{s\sim D,a\sim\pi_\theta}\left[\alpha\log\pi_\theta(a\mid s)-\min_iQ_{\phi_i}(s,a)\right].
$$

连续动作通常使用重参数化：

$$
\xi\sim\mathcal N(0,I),\quad u=\mu_\theta(s)+\sigma_\theta(s)\odot\xi,\quad a=\tanh u.
$$

固定噪声后，动作是策略参数的可微函数，可以沿 $\theta\to a\to Q(s,a)$ 反向传播。更新 Actor 时固定 Critic 参数，但保留 Q 对动作的梯度。

### 4.5 为什么能复用历史数据？

历史数据记录的是“在 s 执行 a，得到 r，到达 s′”。环境规律不变时，这条经验仍可以帮助拟合 Bellman 递推；下一动作由当前策略生成。因此 SAC 是 off-policy 方法，可以长期使用经验池。数据覆盖不足和估值误差仍会影响效果。

### 4.6 伪代码

下面固定温度 α，突出主体流程。

```text
初始化 Actor πθ、两个 Critic Qφ1 和 Qφ2
复制得到两个目标 Q 网络
初始化经验池 D

重复：
    用当前策略采样动作并与环境交互
    将 (s,a,r,s',terminated) 存入 D
    回合结束则重置环境

    数据足够时执行若干次更新：
        从 D 随机抽取小批次
        不计算梯度：
            用当前策略在 s' 采样 a'
            计算带熵的目标 y
        最小化 (Qφi(s,a)-y)^2，更新两个 Critic
        在 s 上用当前 Actor 重参数化采样 anew
        最小化 α logπθ(anew|s)-min Qφi(s,anew)，更新 Actor
        软更新目标网络
```

也可以自动调整 α：熵低于目标时增大它，高于目标时减小它。[自动温度调整](https://arxiv.org/html/1812.05905v2#S5)

## 5. 算法比较与复杂度

| 比较项 | PPO-Clip | SAC |
| --- | --- | --- |
| 核心设计 | 截断策略更新的代理目标 | 最大熵目标与经验回放 |
| Actor 的依据 | 轨迹估计的优势 | Q 对当前生成动作的估值 |
| Critic 常见形式 | V(s) | 两个 Q(s,a) |
| 数据使用 | 新数据训练若干轮后丢弃 | 历史经验反复抽样 |
| 动作空间 | 离散、连续均可 | 本文版本针对连续动作 |
| 熵的地位 | 可选的附加奖励 | 进入策略目标和价值递推 |

离散 SAC 需要相应改写，不能直接照搬连续动作的重参数化流程。

设一次单样本网络训练成本为 C，单条记录大小为 D，忽略不同网络的常数差异：

- PPO 每批 N 条数据训练 K 轮，更新成本约 O(KNC)，数据存储约 O(ND)。
- SAC 每次更新批大小 B，共更新 U 次，成本约 O(UBC)；容量 M 的经验池存储约 O(MD)。

这些空间估计不包含参数、优化器状态和训练激活。它们描述给定更新次数的计算量，不是收敛到最优策略的步数保证。

## 6. 实现时最容易 WA 的地方

| 算法 | 错误 | 后果 |
| --- | --- | --- |
| PPO | 更新期间重新计算旧策略概率 | 失去固定参照 |
| PPO | 最小化截断目标时忘记负号 | 优化方向相反 |
| PPO | 未固定优势和回归目标 | 梯度进入不该更新的路径 |
| PPO | 把 clip 当成概率比的硬约束 | 高估对策略变化的限制 |
| SAC | Actor 更新使用经验池里的旧动作 | 失去通过当前动作生成过程优化的路径 |
| SAC | detach 掉 Actor 损失中的整个 Q | 切断 Q 对动作的指导梯度 |
| SAC | TD 目标未停止梯度 | 目标参与错误更新 |
| SAC | tanh 后仍使用原高斯密度 | 熵与策略损失错误 |
| 两者 | 混淆真正终止和时间截断 | bootstrap 目标错误 |

对于 $a=\tanh u$，密度修正为：

$$
\log\pi(a\mid s)=\log p_U(u\mid s)-\sum_j\log(1-\tanh^2u_j).
$$

实现时需处理数值稳定性；若还要缩放到实际动作范围，应继续考虑变量变换。

读代码时，优先追踪两条数据流：PPO 从轨迹计算优势，再用概率比与截断目标更新策略；SAC 从经验更新 soft Q，再通过 Q 和熵更新策略。逐项检查哪些量固定、哪些量需要梯度，就能掌握算法主体。

## 参考资料

1. Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
2. Haarnoja et al. [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905), 2018/2019.
3. OpenAI Spinning Up. [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
