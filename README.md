# The code of phase 3

### **Environment**

- Use **`llava.yml`** when running experiments with **LLaVA-1.5-7B**.  
- Use **`qwen.yml`** when running experiments with **Qwen-2.5-VL-7B**.

---

### **Prepare the Data and the Vanilla Models**

Download the following repositories/models from Hugging Face:

1. **Vanilla models (unmodified pretrained checkpoints)**  
   - **LLaVA-1.5-7B (Vanilla)**  
     https://huggingface.co/oceanoceanna/LLaVA_Vanilla  
   - **Qwen-2.5-VL-7B (Vanilla)**  
     https://huggingface.co/oceanoceanna/Qwen_Vanilla  

2. **Training split used in baseline comparisons**  
   https://huggingface.co/datasets/oceanoceanna/baseline_train_split

3. **Evaluation benchmark (MLLMU-Bench)**  
   https://huggingface.co/datasets/oceanoceanna/MLLMU-Bench

After downloading the datasets, place **MLLMU-Bench** into the project’s `data/` directory:
If the `data/` folder does not exist, create it manually:

```bash
mkdir data
```

### 🔧 Note: Modify Model Paths Before Running

Both **`newcode/Extractor_llava.py`** and **`newcode/Extractor_qwen.py`** require manual updates to the local paths of the vanilla models. 

Please edit the following lines to point to the correct directories where you downloaded the models:

**LLaVA Vanilla path (Extractor_llava.py):**
```python
self.model = LlavaForConditionalGeneration.from_pretrained(
        '/path/to/LLaVA_Vanilla',
        torch_dtype=torch.float16,
        device_map=self.device,
        low_cpu_mem_usage=True,
        local_files_only=True)
```

**Qwen Vanilla path (Extractor_qwen.py):**
```python
self.model = AutoModelForCausalLM.from_pretrained(
        '/path/to/Qwen_Vanilla',
        torch_dtype=torch.float16,
        device_map=self.device,
        low_cpu_mem_usage=True,
        local_files_only=True)
```
---

### **Final Directory Structure**

After preparing the vanilla models and datasets, the project directory should look like:

```text
PHASE_3/
├── LLaVA_Vanilla/              # Vanilla LLaVA-1.5-7B model
├── Qwen_Vanilla/               # Vanilla Qwen-2.5-VL-7B model
├── baseline_train_split/       
├── data/
│   └── MLLMU-Bench/
├── newcode/
├── refusal_direction/
├── llava.yml
├── qwen.yml
└── ...
```


This structure ensures that all training, evaluation, and steering scripts can correctly locate the required models and datasets.

---

### **Experiments**

#### **1. Run LLaVA Experiments**

```bash
conda create -n llava python=3.10
conda activate llava
conda env update -f llava.yml
```

After installing dependencies, run:

```
bash llava.sh
```

#### **2. Run Qwen Experiments** 

```bash
conda create -n qwen python=3.10
conda activate qwen
conda env update -f qwen.yml
```

After installing dependencies, run:

```
bash qwen.sh
```

Both scripts will perform **test-time unlearning** and evaluate model behavior on:

- **Forget Set**
- **Retain Set**
- **Test Set**
- **Real-world Set**

---

### **Evaluation Metrics Explanation**

After running the experiments, the output will appear in the following format:

```
************************IMAGE_TEXTUAL RESULT*****************:
#######################CLASSIFICATION RESULTS################:
Forget Set Accuracy: ...
Test Set Accuracy: ...
Retain Set Accuracy: ...
Real Set Accuracy: ...

#######################GENERATION RESULTS########################:
Forget Set ROUGE-L: ...
Test Set ROUGE-L: ...
Retain Set ROUGE-L: ...
Real Set ROUGE-L: ...

#######################CLOZE RESULTS########################:
Forget Set Cloze Accuracy: ...
Test Set Cloze Accuracy: ...
Retain Set Cloze Accuracy: ...
Real Set Cloze Accuracy: ...
```

Below explains what each task measures and which direction is desirable.

------

#### 1. Classification Task

**Measures:**
 Image–text multiple-choice question answering accuracy.

**Performance Direction:**

- **Forget/Test Set** → **Lower is better**
  (model should forget harmful or undesired knowledge)
- **Retain/Real Sets** → **Higher is better**
  (model should preserve general reasoning ability)

------

#### 2. Generation Task (ROUGE-L)

**Measures:**
 Similarity between the model-generated response and the ground-truth answer (longest common subsequence).

**Performance Direction:**

- **Forget/ Test Set ROUGE-L** → **Lower is better**
  (the model should not reproduce forgotten content)
- **Retain/Real ROUGE-L** → **Higher is better**
  (the model should maintain generative quality)

------

#### 3. Cloze Task

**Measures:**
 Fill-in-the-blank accuracy for image–text cloze questions.

**Performance Direction:**

- **Forget/Test Set Accuracy** → **Lower is better**
  (model should no longer recall forgotten information)
- **Retain/Real Sets** → **Higher is better**
  (model should maintain factual and contextual reasoning)

---

### **Summary of Metric Direction**

| **Task Type**        | **Forget/Test Set** | **Retain/Real Sets** |
| -------------------- | ------------------- | -------------------- |
| Classification       | ↓ lower is better   | ↑ higher is better   |
| Generation (ROUGE-L) | ↓ lower is better   | ↑ higher is better   |
| Cloze Task           | ↓ lower is better   | ↑ higher is better   |

These evaluation metrics align with the goal of **selective unlearning**:
 the model should **forget only the undesired knowledge** while **preserving all other capabilities**.

# MLLM_Unlearning
