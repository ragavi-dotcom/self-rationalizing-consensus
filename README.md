Self-Rationalizing Multi-Agent Consensus
A Proposer-Critic multi-agent system where one LLM instance argues with itself to catch its own reasoning mistakes, using dynamic test-time compute scaling instead of a fixed number of debate rounds — then honestly benchmarked against a plain single-prompt baseline to see if any of it actually helps.

Why This Project
LLMs have a habit of confidently giving the first answer that "feels" right, even when it's wrong — the classic bat-and-ball riddle is a perfect example. This project builds a small system where the model has to defend its own reasoning to a second, stricter version of itself before an answer is accepted, and only spends extra effort on problems that seem to actually need it. It's built end to end as a real experiment, including the benchmark that tests whether the idea holds up — not just the code that makes it run.

Tech Stack

Core: Python, PyTorch, Hugging Face Transformers, BitsAndBytes (4-bit quantization), Accelerate
Models: microsoft/Phi-3-mini-4k-instruct (4-bit quantized), roberta-large-mnli (contradiction detection between debate rounds), all-MiniLM-L6-v2 (tried, dropped — see write-up)
Evaluation: Hugging Face Datasets (GSM8K), a custom benchmark script comparing against a single-prompt baseline
Environment: Google Colab, free T4 GPU — no paid compute used anywhere
Everything is written as plain Python classes — no LangChain or agent frameworks, so the whole thing stays readable

How to Run

# Open the notebook in Google Colab
# Runtime → Change runtime type → T4 GPU

!pip install -q -U transformers accelerate bitsandbytes sentence-transformers datasets

Run the notebook cells in order from the top. It builds the quantized model, the Proposer and Critic agents, the extraction and cycle-detection logic, the NLI checker, and finally the orchestrator and benchmark — all in one place, no local install needed.

Limitations

Only 17 benchmark problems, and results shift noticeably between runs, so accuracy is reported as a range across 3 runs rather than one clean number
The Critic itself turned out to be unreliable at times — it rejected correct answers and approved wrong ones in testing, likely tied to running on a 4-bit quantized model
The scoring function checks for exact answer substrings, so it can mark a correct-but-differently-worded answer as wrong

Future Work

Log actual token usage per round instead of approximating compute cost from round count
Try majority-vote across multiple Critic calls to see if that fixes the reliability issue
Swap in a larger, non-quantized Critic to check whether the unreliability is really about quantization or just small models in general

Project Structure

self-rationalizing-consensus/
├── Self_Rationalizing_Multi_Agent_Consensus_...ipynb   
├── project_writeup.md   
└── README.md

Built as a Master's application project for DGIST, aimed at Prof. Jisoo Mok's and Prof. Soobin Um's labs.
