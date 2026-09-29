# AI_USE.md — Homework 5: LoRA Fine-Tuning

## 1. Parts for which I used assistance

I used ChatGPT to help understand the LoRA fine-tuning workflow, organize the notebook according to the assignment tasks, troubleshoot compatibility issues, and verify the comparison between the baseline, rank-4, and rank-16 models.

I ran the experiments myself in Google Colab, checked the outputs, saved the LoRA adapters, and interpreted the final results.

## 2. Specific incorrect output or configuration

During development, some of the initially suggested code was not fully compatible with the versions of the Hugging Face libraries available in my Colab environment.

The initial code caused compatibility issues when setting up the training workflow instead of running successfully as written.

## 3. How I found out

I found the problem when I ran the notebook and the training setup produced errors instead of starting normally. I checked the error messages and verified the installed library behavior before continuing.

After correcting the incompatible code, the notebook ran successfully and both LoRA experiments completed with finite training losses.

## 4. What I changed and why it works

I updated the training code to match the APIs supported by the installed Hugging Face and PEFT versions in my Colab environment. I then reran the notebook from the beginning to verify the complete workflow.

The corrected version successfully trained both LoRA configurations, saved the rank-4 and rank-16 adapters, and produced the required inference outputs. I also verified that the baseline, rank-4, and rank-16 models produced the same summaries for the two evaluated dialogues, so I reported that no visible output improvement was observed rather than claiming an improvement that was not present.
