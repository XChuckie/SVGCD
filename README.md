# SVGCD, WWW Companion‘2025

Overview
--------
Implementation of WWW Companion‘25 paper "Semantic-tailored Variational-Contrastive Graph Learning for Cognitive Diagnosis".
Our work focuses on contrastive learning of Graph Cognitive Diagnosis(GCD) and explores a **divide-and-conquer contrastive learning (DCL) strategy** considering **semantic heterogeneity** of CD.
<img src="https://github.com/XChuckie/SVGCD/blob/main/framework.jpg" width=100% center>

Prerequisites
--------
Please refer to **`requirements.txt`**. If meeting other questions about packages, you can solve them according to system prompts.

Datasets
--------
We provide data preparation about **ASSIST0190 dataset**.
You need to run code in **`./scripts/data/rawToMid/assist-0910/process.ipynb`** to get **` intermediate data(located in ./scripts/data/middata/assist-0910)`** for training and testing.

Quick Run
--------
We take **ASSIST0910 data** as an example to introduce **how run our code quickly**.
```python
# 1. Make Sure that you have make `train, test, and Q matrix`.
# 2. Enter the Run File.
cd ./scripts/run.py
# 3. Run Code.
python run.py
```

Notice
------
* Experimental results reported in the paper are based on Pytorch framework on a GeForce RTX 3090 GPU.
* To our credit, our code is a generalized framework that can support multiple datasets and algorithms. If interested, you can deeply analyze the implementation of each part of the code. 

Citation
--------
If you find this useful for your research, please kindly cite the following paper:<be>
```
@article{SVGCD2025,
  title={Semantic-tailored Variational-Contrastive Graph Learning for Cognitive Diagnosis},
  author={Chenao Xia, Fei Liu, Zihan Wang, Zhuangzhuang He, Pengyang Shao, Haowei Zhou, and Yonghui Yang}
  jconference={Companion Proceedings of the ACM Web Conference 2025},
  year={2025}
}
```
