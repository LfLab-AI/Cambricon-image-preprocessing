本研究基于飞腾CPU和寒武纪MLU异构计算平台，提出了高效的图像预处理算子加速方法。通过深入分析MLU的多核并行架构和存储体系，重新设计了算子的计算逻辑和任务划分，结合多种优化策略，显著提升了算子的计算性能。
# 1.代码说明
./mlu下为针对寒武纪MLU开发的Bang C算子库，包括Resize、GrayScale、Perspective、Adjust_Contrast等十二种常见的图像预处理算子，核心算法实现在interface.mlu文件中;
./paddlepaddle为使用paddlepaddle进行图像预处理的示例；
./ml为使用我们开发的Bang C算子以及使用paddlepaddle算子进行图像预处理以及机器学习推理的代码
# 2.Bang C算子性能测试
cncc 版本: 4.8.2，需要配备飞腾处理器和寒武纪AI芯片
```bash
cd mlu
#编译生成mlu.so算子库
make
#pymlu.py调用mlu.so里的算子
#执行相应的算子
python op.py
#测试算子kernel执行时间
cncc resize1.mlu -o test --bang-mlu-arch=mtp_372 -O3 -std=c++11 -lstdc++ -lopencv_core -lopencv_imgcodecs -I/usr/include/opencv4 -D__ARM_NEON__=0 -lcnrt -lcndev -lcndrv
cnperf-cli record --pmu ./test
cnperf-cli kernel
```

# 3. paddlepaddle算子性能测试
paddlepaddle: paddle-custom-mlu  0.0.0
```bash
cd paddlepaddle
#需要配置paddlepaddle环境
python  testop.py
```
