# Prompting-Techniques-Testing-pipeline
This repo contains the testing framework used to evaluate 7 prompting techniques across 4 LLMs, measuring their accuracy, hallucination rate and avg response time. 
All the models were accessed through the Free Groq API- to make it fully reproducible. 
Prompting Techniques Evaluated 
Technique         Description
Zero-shot         Direct question, no examples or instructions
One-shotO         One worked example given for the model to understand the reasoning 
Few-shot          Three worked examples given to the model
Chain-of-Thought  Step-by-step reasoning instruction
Tree-of-Thought   Three reasoning paths generated and evaluated
Thread-of-Thought Continuous self-referential reasoning thread
Graph-of-Thought  Node-edge graph structure for problem decomposition 

Metrics - 
Accuracy (%) — Proportion of responses matching ground truth
Hallucination Rate (%) — Proportion of responses containing fabricated or contradictory content
Response Time (s) — Wall-clock time from prompt to full response

Output
When complete, a results/ folder is created containing:
raw_results_TIMESTAMP.csv
Every single API response logged with all metrics.

