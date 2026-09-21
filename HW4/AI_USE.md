# HW4 — AI Use

## 1. Which parts did I use an assistant for, and which did I do myself?

I used ChatGPT to help draft the PyTorch code for character tokenization, the sliding-window dataset, masked multi-head attention, the decoder blocks, training, and the three decoding methods. I ran each cell in Colab, checked its output, trained the model on the course Shakespeare file, and compared the generated samples myself.

## 2. What was one specific thing the assistant got wrong?

The assistant initially gave me a file path that assumed `Shakespeare.txt` had been uploaded directly to Colab:

`/content/Shakespeare.txt`

The cell failed with:

`FileNotFoundError: Shakespeare.txt was not found. Upload it using Colab's folder icon.`

My file was actually named `shakespeare.txt` and was stored in my Google Drive HW4 folder.

## 3. How did I find out?

I ran the dataset-loading cell and received the `FileNotFoundError`. I then checked where I had saved the file and confirmed its exact filename in Google Drive.

## 4. What did I change, and why does it work?

I mounted Google Drive and changed the path to:

`/content/drive/MyDrive/GEN AI HWS/HW4/shakespeare.txt`

The corrected cell read the file and printed its character count and a text preview. I also checked that the attention block preserved the shape `(2, 128, 128)`, the model produced logits of shape `(1, 128, 49)`, and training loss decreased from 3.0354 to 1.9780 over six epochs.
