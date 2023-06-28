# Brookins and DeBacker, "Playing games with GPT: What can we learn about a large language model from canonical strategic games?"
## Replication files for [Brookins and DeBacker (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4493398)

## Abstract:
We aim to understand fundamental preferences over fairness and cooperation embedded in artificial intelligence (AI). We do this by having a large language model (LLM), GPT-3.5, play two classic games: the dictator game and the prisoner's dilemma. We compare the decisions of the LLM to those of humans in laboratory experiments. We find that the LLM replicates human tendencies towards fairness and cooperation. It does not choose the optimal strategy in most cases. Rather, it shows a tendency towards fairness in the dictator game, even more so than human participants. In the prisoner's dilemma, the LLM displays rates of cooperation much higher than human participants (about 65% versus 37% for humans). These findings aid our understanding of the ethics and rationality embedded in AI.

# Replication instructions:

The Python scripts in this directory use the [`openai` Python library](https://github.com/openai/openai-python) to access the GPT-3.5 model via the OpenAI API.  The library can be installed via `pip install --upgrade openai`.

Note that to access the OpenAI API, you will need an [OpenAI account](https://platform.openai.com/signup) and to [retrieve your API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) from it.

If you installed the [Anaconda distribution](https://www.anaconda.com/download) of Python, all other required packages should already be installed (e.g., Numpy, Pandas, `os`).

Once you have those Python requirements met, you can run the following scripts with the modifications noted below
* `GPT_dictator.py`
  * Replace the string on line 15 with your secret API key
  * Output will be saved to the `data` directory
* `GPT_prisoners_dilemma.py`
  * Replace the string on line 14 with your secret API key
  * Output will be saved to the `data` directory
* `GPT_tables_figures.py`
  * This will use the data in the `data` directory to replicate the tables and figures in the paper.  Tables and figures will be saved to the `./code/tables` and `./code/images` directories, respectively.

# Reproducibility notes:
Given the nature of the GPT model, one cannot reproduce exactly the same results with each model interaction.  But hopefully the results will be close enough to replicate the results in the paper.  Data generated for the simulations used in the paper are in the `data` directory. These data can be used to replicate the tables and figures in the paper exactly.

Our simulations were done between June 9 and June 20, 2023 with `gpt-3.5-turbo`. This model is continuously updated, but a snapshot of the model from June 13, 2033 is available from OpenAI by specifying `gpt-3.5-turbo-0613` as the model when using the OpenAI API.