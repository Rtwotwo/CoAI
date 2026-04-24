# Reforcement Learning Logs

# Proximal Policy Optimization(PPO)

近端策略优化算法核心是通过在与环境交互采样数据与使用随机梯度上升优化代理目前之间交替进行，能够实现多个批处理更新的周期。PPO通过优化带有裁剪概率比的新目标，并在对性能估计形成悲观下届。