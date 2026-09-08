# AI Use Disclosure - Homework 2

## 1. What AI tool did you use?

I used ChatGPT as an assistant while completing Homework 2.

## 2. How did you use AI?

I used ChatGPT to help understand the assignment requirements, organize the implementation into individual steps, explain errors and outputs, and review whether the experiments satisfied the homework requirements.

I also used it to help structure the evaluation tables and interpret the results from the embedding transfer learning, RAG, and training optimization experiments.

## 3. What parts of the work did you verify yourself?

I executed the code in Google Colab and verified the outputs produced by each experiment. I checked the nearest-neighbor results and cosine similarities in Q1, inspected the retrieved Wikipedia chunks and generated answers in Q2, and compared runtime, GPU memory, and loss measurements in Q3.

I also manually checked whether the retrieved chunks in the RAG experiment contained the information needed to answer each question.

## 4. Describe one specific mistake made by the AI assistant and how you corrected it.

During the RAG experiment, the initial approach for loading the Jurassic Park Wikipedia page retrieved content associated with the Jurassic Park franchise rather than the exact movie page required for the experiment.

I identified the problem by inspecting the retrieved document content. The loading method was then changed so that the exact Jurassic Park movie Wikipedia page was used. After correcting the source document, I rebuilt the chunks and vector store and reran the retrieval and generation experiment.

This demonstrated why AI-generated implementation suggestions still need to be validated against the actual data and experiment outputs.