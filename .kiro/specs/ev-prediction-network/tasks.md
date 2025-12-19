# 实现计划

- [x] 1. 创建核心数据模型和网络架构
  - [x] 1.1 创建EVPredictionNetwork神经网络类
    - 在 `training/ev_prediction_network.py` 中实现
    - 包含共享编码器和三个预测头（EV、动作EV、策略）
    - 支持可配置的动作数量和隐藏层维度
    - _Requirements: 1.1, 1.2, 1.3, 6.1, 6.2, 6.3_
  - [x] 1.2 编写属性测试：网络输出维度正确性
    - **Property 1: 网络输出维度正确性**
    - **Validates: Requirements 1.1, 1.2, 1.3, 6.2, 6.3**
  - [x] 1.3 编写属性测试：策略概率归一化
    - **Property 2: 策略概率归一化**
    - **Validates: Requirements 1.4, 6.4**

- [x] 2. 实现数据加载和预处理
  - [x] 2.1 创建EVDataset数据集类
    - 在 `training/ev_dataset.py` 中实现
    - 从JSON验证数据文件加载场景
    - 提取4维输入特征和3种目标值
    - 处理无效数据（NaN/Inf）
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [x] 2.2 编写属性测试：数据解析往返一致性
    - **Property 4: 数据解析往返一致性**
    - **Validates: Requirements 2.1**
  - [x] 2.3 编写单元测试：数据集加载和索引
    - 测试数据集长度、索引访问、特征提取
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. 实现训练器和损失函数
  - [x] 3.1 创建EVTrainer训练器类
    - 在 `training/ev_trainer.py` 中实现
    - 支持多任务损失（EV MSE + 动作EV MSE + 策略交叉熵）
    - 支持可配置的损失权重
    - 实现训练循环和评估方法
    - _Requirements: 3.1, 3.2, 3.3, 5.1, 5.2, 5.3, 5.4_
  - [x] 3.2 编写单元测试：损失函数计算
    - 测试MSE和交叉熵损失的正确性
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 4. 实现模型保存和加载
  - [x] 4.1 在EVTrainer中实现save_checkpoint和load_checkpoint方法
    - 保存模型权重、优化器状态、训练配置
    - 加载时恢复完整训练状态
    - 处理文件不存在的错误情况
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 4.2 编写属性测试：模型保存/加载往返一致性
    - **Property 3: 模型保存/加载往返一致性**
    - **Validates: Requirements 4.2**

- [x] 5. Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 6. 创建训练脚本和CLI接口
  - [x] 6.1 创建训练入口脚本
    - 在 `train_ev_prediction.py` 中实现
    - 支持命令行参数配置（学习率、批次大小、轮数等）
    - 实现训练进度日志输出
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [x] 6.2 添加模型评估和结果报告功能
    - 输出各项指标的统计摘要
    - _Requirements: 5.4_

- [x] 7. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。
