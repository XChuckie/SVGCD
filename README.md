# Semantic-tailored Variational-Contrastive Graph Learning for Cognitive Diagnosis

## 🔥 Experiments

Here, we describe the execution of algorithm in detail, including: **Prerequisites, Data Preparation and Quick Run**. To our credit, our code is a `generalized framework` that can `support multiple datasets and algorithms`. If interested, you can deeply analyze the implementation of each part of the code. And note that:

- We use Pytorch framework to conduct all experiments.
- All experiments are conducted on a GeForce RTX 3090 GPU.
- **This source code is fully compliant with WWW's blind review requirements, and it will not contain any author or affiliation information.**

### ✨ Prerequisites

The version of key packages are described in the **`requirements.txt`**. If meeting other questions about packages, you can solve it according to system prompts.

### ✨ Data Preparation

**Outline**: Our Code is easily adapted to **`Custom Type`**. And we also provide data preparation about **`ASSIST0190 dataset`**.

1. 🦧 How to make code adapt to **`Custom Type`**?

- If having a `custom-divided train/test data`, you just need to store them into directory: `./scripts/data/middata/name`, where `"name"` is a customized dataset name. For example, if making a custom data about `Junyi dataset`, you can set `"name"` as `junyi`. Note that:
- You need to rename `train/test data` as `name-train.inter.csv` and `name-test.inter.csv`.
- You need provide Q matrix data anda renme it as `name-assist-0910-Q`.
- If having a `custom-divided code`, you just can store it into directory: `./scripts/data/rawToMid/name`. And you can set raw data into directory: `./scripts/data/rawdata/name`. In addition, we suggest that you can rename all results following the above.

2. 🦧 Take **`ASSIST0910 dataset`** as an example: you need to run code in **`./scripts/data/rawToMid/assist-0910/process.ipynb`** to get `intermediate data(located in ./scripts/data/middata/assist-0910)` for training and testing.

### ✨Quick Run

Here, we take **`ASSIST0910 data`** as an example to introduce **how run our code quickly**.

1. 🦧 Process of executing code

   ```python
   # 1. Make Sure that you have make `train, test and Q matrix`, and put them into directory "./scripts/data/middata/assist-0910" correctly.
   # 2. Enter the Run File Interface.
   cd ./scripts/run.py
   # 3. Run Code
   python run.py
   ```

2. 🦧 You can **modify the parameters** in the location (`stl-edu-model/confs/exMatCons/assist-0910/scgnn.yaml`) directly.

### ✨Wandb Tool

In practice, we utilize **wandb tool (an efficient and convenient package for visualizing)** to visualize our results. If you have not tried it or do not like it, `it is easy to comment out as follwoing`:

```python
# Annotation Code Location 1: ./scripts/run.py
wandb.init(project='slt-educd', config=config, mode='online')
update_config_with_sweep_parameters(config)
# Annotation Code Location 2: ./utils/callback/callbacks/baseLogger.py
wandb.log(logs)
```

Otherwise, you can cancel these annotation and use them.
