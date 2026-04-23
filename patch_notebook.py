import json

with open("colab_train.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Update Step 2: Clone URL
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "id" in cell and cell["id"] == "clone":
        new_source = [
            "import os\n",
            "\n",
            "if not os.path.exists('Enterprise-AP-Environment'):\n",
            "    !git clone https://github.com/dharmendra26-wiz/Enterprise-AP-Environment\n",
            "    print('✅ Repo cloned')\n",
            "else:\n",
            "    !cd Enterprise-AP-Environment && git pull\n",
            "    print('✅ Repo updated')\n",
            "\n",
            "os.chdir('Enterprise-AP-Environment')\n",
            "print(f'📁 Working directory: {os.getcwd()}')"
        ]
        cell["source"] = new_source
        
    # Update Class Names
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if "from app.environment import InvoiceEnvironment" in line:
                source[i] = line.replace("InvoiceEnvironment", "EnterpriseAPEnvironment")
            if "env = InvoiceEnvironment(" in line:
                source[i] = line.replace("InvoiceEnvironment", "EnterpriseAPEnvironment")

# Insert TRL/Unsloth section before the last summary cell
summary_idx = -1
for i, cell in enumerate(notebook["cells"]):
    if cell["id"] == "summary":
        summary_idx = i
        break

trl_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 8: Real LLM Training with TRL GRPO & Unsloth\n",
    "\n",
    "This section demonstrates how to connect our environment to a real gradient-based training loop using Hugging Face TRL and Unsloth. We fine-tune Qwen-0.5B using GRPO (Group Relative Policy Optimization) on the `easy` task as a minimal proof of concept."
   ],
   "id": "step8"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\" --quiet\n",
    "!pip install --no-deps \"trl<0.9.0\" peft accelerate bitsandbytes --quiet\n",
    "print('✅ Unsloth & TRL installed')"
   ],
   "id": "install_unsloth"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from unsloth import FastLanguageModel\n",
    "from trl import GRPOTrainer, GRPOConfig\n",
    "from app.environment import EnterpriseAPEnvironment\n",
    "from app.models import Action\n",
    "import json\n",
    "\n",
    "# 1. Load Model\n",
    "max_seq_length = 1024\n",
    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "    model_name = \"Qwen/Qwen2.5-0.5B-Instruct\",\n",
    "    max_seq_length = max_seq_length,\n",
    "    load_in_4bit = True,\n",
    ")\n",
    "model = FastLanguageModel.get_peft_model(\n",
    "    model,\n",
    "    r = 16, \n",
    "    target_modules = [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
    "    lora_alpha = 16,\n",
    ")\n",
    "\n",
    "# 2. Define Env Reward Function for TRL\n",
    "def env_reward_func(completions, prompts):\n",
    "    rewards = []\n",
    "    env = EnterpriseAPEnvironment(\"easy\")\n",
    "    \n",
    "    for completion in completions:\n",
    "        env.reset()\n",
    "        text = completion[0]['content']\n",
    "        try:\n",
    "            import re\n",
    "            m = re.search(r\"\\{.*\\}\", text, re.DOTALL)\n",
    "            if m:\n",
    "                action_dict = json.loads(m.group())\n",
    "                result = env.step(Action(**action_dict))\n",
    "                rewards.append(result.reward)\n",
    "            else:\n",
    "                rewards.append(-0.1)\n",
    "        except Exception:\n",
    "            rewards.append(-0.1)\n",
    "            \n",
    "    return rewards\n",
    "\n",
    "# 3. Dummy Dataset (Prompts to trigger policy generation)\n",
    "from datasets import Dataset\n",
    "dummy_dataset = Dataset.from_dict({\n",
    "    \"prompt\": [[\n",
    "        {\"role\": \"system\", \"content\": \"You are an AP Agent. Respond with a JSON action.\"},\n",
    "        {\"role\": \"user\", \"content\": \"Task: easy. Your inbox:\\n [email_001] From: vendor | Subject: Invoice attached\\nBegin processing.\"}\n",
    "    ]] * 16\n",
    "})\n",
    "\n",
    "# 4. Train\n",
    "training_args = GRPOConfig(\n",
    "    output_dir=\"outputs\",\n",
    "    learning_rate=5e-5,\n",
    "    per_device_train_batch_size=1,\n",
    "    gradient_accumulation_steps=4,\n",
    "    max_prompt_length=256,\n",
    "    max_completion_length=256,\n",
    "    num_generations=4,\n",
    "    max_steps=10,\n",
    "    logging_steps=1,\n",
    "    optim=\"adamw_8bit\",\n",
    ")\n",
    "\n",
    "trainer = GRPOTrainer(\n",
    "    model=model,\n",
    "    reward_funcs=[env_reward_func],\n",
    "    args=training_args,\n",
    "    train_dataset=dummy_dataset,\n",
    ")\n",
    "\n",
    "print(\"Starting GRPO training step...\")\n",
    "trainer.train()"
   ],
   "id": "unsloth_train"
  }
]

if summary_idx != -1:
    notebook["cells"] = notebook["cells"][:summary_idx] + trl_cells + notebook["cells"][summary_idx:]

with open("colab_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("colab_train.ipynb updated.")
