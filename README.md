# Custom-150M-LM-Pretrain-Finetune

## 🎯 Project Goal

This repository documents the end-to-end process of building a Neural Machine Translation (NMT) model from scratch using modern Transformer architecture. The primary goal of this project is **not to achieve state-of-the-art performance**, but to **demonstrate a deep, practical understanding of the entire deep learning pipeline** for a complex NLP task.

While highly optimized, production-ready models like those from Helsinki-NLP offer superior performance, this project focuses on the foundational skills required to build such systems. It showcases the journey from raw data to a functional, fine-tuned model, covering data engineering, custom architecture implementation, and advanced training strategies.

## 🌟 Key Skills & Features Demonstrated

- **From-Scratch Architecture in PyTorch:**
  - **Grouped-Query Attention (GQA):** Implemented to balance computational efficiency and model quality.
  - **Rotary Positional Embeddings (RoPE):** For robust handling of sequence length and relative positions.
  - **SwiGLU Activation:** Utilizing modern activation functions for better performance.
  - **Pre-Layer Normalization:** To ensure stable training of a deep network.
  - **KV Cache for Efficient Generation:** Implemented and utilized a Key-Value cache during autoregressive decoding. This avoids re-computing keys and values for the entire context at each step, dramatically speeding up inference.
- **End-to-End Data Pipeline:**
  - **Large-Scale Data Curation:** Developed a multi-stage filtering pipeline to process the massive OSCAR dataset, removing noise and selecting high-quality text using regex, statistical heuristics, and token-based analysis.
  - **Synthetic Data Generation:** Created a parallel corpus of ~80,000 sentence pairs by prompting the Gemini API, demonstrating a practical approach to overcome data scarcity.
  - **Custom Tokenizer Training:** Trained a `Unigram` tokenizer specifically for the German-English domain of the project.
- **Advanced Training Strategy:**
  1.  **Denoising Pre-training:** Employed a T5-style span corruption objective on monolingual data to teach the model robust language representations.
  2.  **Translation Fine-tuning:** Adapted the pre-trained model for the specific downstream task of machine translation.
- **Tooling & Optimization:**
  - Utilized `torch.compile` and `bfloat16` for efficient, modern PyTorch training.
  - Integrated `wandb` for experiment tracking and diagnostics.
  - Leveraged Hugging Face `datasets` for handling large, streaming datasets.

## 💻 Source Code & Reproducibility

The complete source code for this project is available in the Jupyter notebooks, organized by function:
- **`model.ipynb`**: Contains the core model architecture, the two-stage training process (pre-training and fine-tuning), and inference examples.
- **`tokenizer.ipynb`**: Details the process of training the custom `Unigram` tokenizer.
- **`dataset2.ipynb`**: Shows the pipeline for generating the synthetic parallel dataset using an external LLM.

The code is structured to be clear and readable, allowing for easy reproduction of the results or for use as a foundation for your own experiments.

## 💡 The Power of Pre-training

A key component of this project was a two-stage training strategy. The model was first pre-trained on a large monolingual dataset using a denoising objective before being fine-tuned on the translation task. This approach, often called Transfer Learning, proved highly effective.

The graph below compares the fine-tuning loss curves for two scenarios: one using a pre-trained model and another training from a randomly initialized state.

<img width="1628" height="931" alt="image" src="https://github.com/user-attachments/assets/502101ae-5bf2-4cb5-a154-caa03c4b02e7" />

As the comparison illustrates, the pre-trained model starts with a significantly lower loss (~3.7 vs. ~9.8) and converges faster to a better final value. This demonstrates that the denoising pre-training phase effectively equipped the model with a strong understanding of language structure, which served as a powerful foundation for the downstream translation task.

## 🚀 Optimization: The Impact of KV Caching

During autoregressive generation (creating text token by token), a significant performance bottleneck is the re-computation of attention scores for the entire sequence at every step. To mitigate this, a **Key-Value (KV) cache** was implemented. This mechanism stores the calculated keys and values from previous steps, allowing the model to only compute attention for the newest token.

Measurements conducted in the `model.ipynb` notebook showed that using the KV cache **accelerates the generation process by approximately 10-20%**. This is a critical optimization for making inference practical, and its implementation can be verified in the notebook's `generate` method.

## 📊 Translation Examples & Quality Analysis

To assess the model's capabilities, it was tested on German texts of varying complexity, from simple A1 sentences to abstract C1-level prose. The results provide a clear picture of the model's strengths and limitations.

| Level | German Source Text                                                                                                                                                                                                                                                                                           | Model's English Translation                                                                                                                                                                                                                         |
|:-----:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  **A1** | `Ich bin Anna. Ich fahre nach Berlin. Ich buche ein Hotel. Das Hotel ist gut. Berlin ist groß. Ich mag Berlin.`                                                                                                                                                                                                  | `I am Anna. I am going to Berlin. I book a hotel. The hotel is good. I like Berlin.`                                                                                                                                                                |
|  **A2** | `Hallo, mein Name ist Anna. Nächste Woche fliege ich nach Berlin, weil ich die Stadt besuchen will. Ich habe schon ein Hotelzimmer gebucht. Ich möchte die Sehenswürdigkeiten sehen und vielleicht in einem Restaurant essen.`                                                                                  | `Hello, my name is Anna. Next week I fly to Berlin because I want to visit the city. I have booked a hotel room already. I would like to see the sights and maybe eat it in a restaurant.`                                                          |
|  **B1** | `Ich bereite gerade meine Reise nach Berlin vor und freue mich schon sehr darauf. Mir wurde gesagt, dass man unbedingt das Brandenburger Tor besuchen sollte. Falls du Zeit hast, könnten wir uns vielleicht treffen.`                                                                                       | `I just wanted to go on my trip to Berlin and am already very excited about it. I was told that you should definitely visit the Brandenburg Tor. If you have time, we might be able to meet together.`                                               |
|  **B2** | `Bei der Planung meiner Berlin-Reise lege ich besonderen Wert darauf, nicht nur die touristischen Hauptattraktionen abzuklappern, sondern auch einen authentischen Eindruck vom Leben in den verschiedenen Stadtteilen zu bekommen.`                                                                        | `When planning my Berlin trip, I put particular emphasis on not only the tourist main attractions, but to get an authentic impression of life in the various neighborhoods.`                                                                         |
|  **C1** | `Die Faszination Berlins ergibt sich aus dem ständigen Spannungsfeld zwischen seiner turbulenten Vergangenheit und seiner dynamischen Gegenwart. Um die Stadt wirklich zu begreifen, genügt es nicht, die Überreste der Mauer zu betrachten; man muss sich mit den vielschichtigen Narrativen auseinandersetzen.` | `The fascination of Berlin is derived from the constant tension field between its turbulent past and its dynamic present. To really attack the city, it is not enough to consider the remains of the wall; one must deal with the multifaceted narcotics.` |

### Analysis of Results
-   **A1/A2 Levels:** The model demonstrates a strong command of basic grammar and vocabulary, producing accurate and fluent translations. A minor error ("eat it") appears in the A2 text, but the overall meaning is preserved.
-   **B1/B2 Levels:** As complexity increases, the model maintains grammatical structure but starts to struggle with nuance and idiomatic language. For example, it translates "Ich bereite gerade vor" (I am currently preparing) slightly incorrectly and misses the common German compound "Brandenburger Tor".
-   **C1 Level:** The model's limitations become clear with abstract and complex vocabulary. The incorrect translation of "Narrativen" (narratives) to "narcotics" is a significant semantic error, indicating that the model's vocabulary and contextual understanding are not sufficient for high-level academic or literary texts.

This performance is consistent with the project's scope and honestly reflects the capabilities of a model of this size trained on the given data.

## 📚 Open-Source Synthetic Dataset

To overcome the challenge of limited high-quality parallel data, a synthetic dataset of ~80,000 German-English sentence pairs was generated. This dataset has been made publicly available to aid other researchers and developers.

You can access and use the dataset on the Hugging Face Hub:
**[arhunov/oscar-en-de-synthetic](https://huggingface.co/datasets/arhunov/oscar-en-de-synthetic)**

## 🎓 Reflections & Learning Journey

This project was far more than a technical exercise; it was an immersive journey into the heart of modern NLP. The path, which spanned over a month, was filled with challenges, experiments, and invaluable lessons. It began with a theoretical understanding of Transformers and culminated in a deep, practical intuition for what it takes to bring a complex model to life.

Building everything from scratch-from the tokenizer to the final fine-tuning loop-forced a confrontation with subtle architectural details and the profound impact of data quality. There were moments of frustration when training runs failed and moments of discovery when a new technique unlocked a significant improvement. This iterative cycle of trial, error, and refinement is where the most meaningful learning occurred.

Ultimately, this project solidified my understanding that building successful deep learning systems is a craft that blends scientific principles with engineering pragmatism. It's about navigating the trade-offs between computational cost and performance, finding creative solutions to data scarcity, and having the persistence to see a complex idea through to completion. The experience gained is priceless and serves as a solid foundation for tackling future challenges in the field.
