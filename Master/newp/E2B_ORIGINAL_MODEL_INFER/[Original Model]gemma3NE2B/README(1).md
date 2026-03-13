---
license: gemma
library_name: transformers
pipeline_tag: image-text-to-text
extra_gated_heading: Access Gemma on Hugging Face
extra_gated_prompt: To access Gemma on Hugging Face, you’re required to review and
  agree to Google’s usage license. To do this, please ensure you’re logged in to Hugging
  Face and click below. Requests are processed immediately.
extra_gated_button_content: Acknowledge license
base_model: google/gemma-3n-E4B-it
tags:
- automatic-speech-recognition
- automatic-speech-translation
- audio-text-to-text
- video-text-to-text
---

> [!Note]
> This repository corresponds to the launch version of Gemma 3n E2B IT (Instruct), to be used with Hugging Face `transformers`,
> supporting text, audio, and vision (image and video) inputs.
> 
> Gemma 3n models have multiple architecture innovations:
>  * They are available in two sizes based on [effective parameters](https://ai.google.dev/gemma/docs/gemma-3n#parameters). While the raw parameter count of this model is 6B, the architecture design allows the model to be run with a memory footprint comparable to a traditional 2B model by offloading low-utilization matrices from the accelerator.
>  * They use a MatFormer architecture that allows nesting sub-models within the [E4B model](https://huggingface.co/google/gemma-3n-E4B-it). We provide one sub-model (this model repository), or you can access a spectrum of custom-sized models using the [Mix-and-Match method](https://goo.gle/gemma3n-matformer-lab).
>
> Learn more about these techniques in the [technical blog post](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide)
> and the [Gemma documentation](https://ai.google.dev/gemma/docs/gemma-3n). 



# Gemma 3n model card

**Model Page**: [Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n)

**Resources and Technical Documentation**:

-   [Responsible Generative AI Toolkit](https://ai.google.dev/responsible)
-   [Gemma on Kaggle](https://www.kaggle.com/models/google/gemma-3n)
-   [Gemma on HuggingFace](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)
-   [Gemma on Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemma3n)

**Terms of Use**: [Terms](https://ai.google.dev/gemma/terms)\
**Authors**: Google DeepMind

## Model Information

Summary description and brief definition of inputs and outputs.

### Description

Gemma is a family of lightweight, state-of-the-art open models from Google,
built from the same research and technology used to create the Gemini models.
Gemma 3n models are designed for efficient execution on low-resource devices.
They are capable of multimodal input, handling text, image, video, and audio
input, and generating text outputs, with open weights for pre-trained and
instruction-tuned variants. These models were trained with data in over 140
spoken languages.

Gemma 3n models use selective parameter activation technology to reduce resource
requirements. This technique allows the models to operate at an effective size
of 2B and 4B parameters, which is lower than the total number of parameters they
contain. For more information on Gemma 3n's efficient parameter management
technology, see the
[Gemma 3n](https://ai.google.dev/gemma/docs/gemma-3n#parameters)
page.

### Inputs and outputs

-   **Input:**
    -   Text string, such as a question, a prompt, or a document to be
        summarized
    -   Images, normalized to 256x256, 512x512, or 768x768 resolution
        and encoded to 256 tokens each
    -   Audio data encoded to 6.25 tokens per second from a single channel
    -   Total input context of 32K tokens
-   **Output:**
    -   Generated text in response to the input, such as an answer to a
        question, analysis of image content, or a summary of a document
    -   Total output length up to 32K tokens, subtracting the request
        input tokens

### Usage

Below, there are some code snippets on how to get quickly started with running
the model. First, install the Transformers library. Gemma 3n is supported
starting from transformers 4.53.0.

```sh
$ pip install -U transformers
```

Then, copy the snippet from the section that is relevant for your use case.

#### Running with the `pipeline` API

You can initialize the model and processor for inference with `pipeline` as
follows.

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3n-e2b-it",
    device="cuda",
    torch_dtype=torch.bfloat16,
)
```

With instruction-tuned models, you need to use chat templates to process our
inputs first. Then, you can pass it to the pipeline.

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
# Okay, let's take a look!
# Based on the image, the animal on the candy is a **turtle**.
# You can see the shell shape and the head and legs.
```

#### Running the model on a single GPU

```python
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3n-e2b-it"

model = Gemma3nForConditionalGeneration.from_pretrained(model_id, device="cuda", torch_dtype=torch.bfloat16,).eval()

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
# focusing on a cluster of pink cosmos flowers and a busy bumblebee.
# It has a slightly soft, natural feel, likely captured in daylight.
```

### Citation

```
@article{gemma_3n_2025,
    title={Gemma 3n},
    url={https://ai.google.dev/gemma/docs/gemma-3n},
    publisher={Google DeepMind},
    author={Gemma Team},
    year={2025}
}
```

## Model Data

Data used for model training and how the data was processed.

### Training Dataset

These models were trained on a dataset that includes a wide variety of sources
totalling approximately 11 trillion tokens. The knowledge cutoff date for the
training data was June 2024. Here are the key components:

-   **Web Documents**: A diverse collection of web text ensures the model
    is exposed to a broad range of linguistic styles, topics, and vocabulary.
    The training dataset includes content in over 140 languages.
-   **Code**: Exposing the model to code helps it to learn the syntax and
    patterns of programming languages, which improves its ability to generate
    code and understand code-related questions.
-   **Mathematics**: Training on mathematical text helps the model learn
    logical reasoning, symbolic representation, and to address mathematical queries.
-   **Images**: A wide range of images enables the model to perform image
    analysis and visual data extraction tasks.
-   Audio: A diverse set of sound samples enables the model to recognize
    speech, transcribe text from recordings, and identify information in audio data.

The combination of these diverse data sources is crucial for training a
powerful multimodal model that can handle a wide variety of different tasks and
data formats.

### Data Preprocessing

Here are the key data cleaning and filtering methods applied to the training
data:

-   **CSAM Filtering**: Rigorous CSAM (Child Sexual Abuse Material)
    filtering was applied at multiple stages in the data preparation process to
    ensure the exclusion of harmful and illegal content.
-   **Sensitive Data Filtering**: As part of making Gemma pre-trained models
    safe and reliable, automated techniques were used to filter out certain
    personal information and other sensitive data from training sets.
-   **Additional methods**: Filtering based on content quality and safety in
    line with
    [our policies](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf).

## Implementation Information

Details about the model internals.

### Hardware

Gemma was trained using [Tensor Processing Unit
(TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) hardware (TPUv4p, TPUv5p
and TPUv5e). Training generative models requires significant computational
power. TPUs, designed specifically for matrix operations common in machine
learning, offer several advantages in this domain:

-   **Performance**: TPUs are specifically designed to handle the massive
    computations involved in training generative models. They can speed up
    training considerably compared to CPUs.
-   **Memory**: TPUs often come with large amounts of high-bandwidth memory,
    allowing for the handling of large models and batch sizes during training.
    This can lead to better model quality.
-   **Scalability**: TPU Pods (large clusters of TPUs) provide a scalable
    solution for handling the growing complexity of large foundation models.
    You can distribute training across multiple TPU devices for faster and more
    efficient processing.
-   **Cost-effectiveness**: In many scenarios, TPUs can provide a more
    cost-effective solution for training large models compared to CPU-based
    infrastructure, especially when considering the time and resources saved
    due to faster training.

These advantages are aligned with
[Google's commitments to operate sustainably](https://sustainability.google/operating-sustainably/).

### Software

Training was done using [JAX](https://github.com/jax-ml/jax) and
[ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/).
JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models. ML
Pathways is Google's latest effort to build artificially intelligent systems
capable of generalizing across multiple tasks. This is specially suitable for
foundation models, including large language models like these ones.

Together, JAX and ML Pathways are used as described in the
[paper about the Gemini family of models](https://goo.gle/gemma2report):
*"the 'single controller' programming model of Jax and Pathways allows a single
Python process to orchestrate the entire training run, dramatically simplifying
the development workflow."*

## Evaluation

Model evaluation metrics and results.

### Benchmark Results

These models were evaluated at full precision (float32) against a large
collection of different datasets and metrics to cover different aspects of
content generation. Evaluation results marked with **IT** are for
instruction-tuned models. Evaluation results marked with **PT** are for
pre-trained models.

#### Reasoning and factuality

| Benchmark                      | Metric         | n-shot   |  E2B PT  |  E4B PT  |
| ------------------------------ |----------------|----------|:--------:|:--------:|
| [HellaSwag][hellaswag]         | Accuracy       | 10-shot  |   72.2   |   78.6   |
| [BoolQ][boolq]                 | Accuracy       | 0-shot   |   76.4   |   81.6   |
| [PIQA][piqa]                   | Accuracy       | 0-shot   |   78.9   |   81.0   |
| [SocialIQA][socialiqa]         | Accuracy       | 0-shot   |   48.8   |   50.0   |
| [TriviaQA][triviaqa]           | Accuracy       | 5-shot   |   60.8   |   70.2   |
| [Natural Questions][naturalq]  | Accuracy       | 5-shot   |   15.5   |   20.9   |
| [ARC-c][arc]                   | Accuracy       | 25-shot  |   51.7   |   61.6   |
| [ARC-e][arc]                   | Accuracy       | 0-shot   |   75.8   |   81.6   |
| [WinoGrande][winogrande]       | Accuracy       | 5-shot   |   66.8   |   71.7   |
| [BIG-Bench Hard][bbh]          | Accuracy       | few-shot |   44.3   |   52.9   |
| [DROP][drop]                   | Token F1 score | 1-shot   |   53.9   |   60.8   |

[hellaswag]: https://arxiv.org/abs/1905.07830
[boolq]: https://arxiv.org/abs/1905.10044
[piqa]: https://arxiv.org/abs/1911.11641
[socialiqa]: https://arxiv.org/abs/1904.09728
[triviaqa]: https://arxiv.org/abs/1705.03551
[naturalq]: https://github.com/google-research-datasets/natural-questions
[arc]: https://arxiv.org/abs/1911.01547
[winogrande]: https://arxiv.org/abs/1907.10641
[bbh]: https://paperswithcode.com/dataset/bbh
[drop]: https://arxiv.org/abs/1903.00161

#### Multilingual

| Benchmark                           | Metric                  | n-shot   |  E2B IT  |  E4B IT  |
| ------------------------------------|-------------------------|----------|:--------:|:--------:|
| [MGSM][mgsm]                        | Accuracy                |  0-shot  |   53.1   |   60.7   |
| [WMT24++][wmt24pp] (ChrF)           | Character-level F-score |  0-shot  |   42.7   |   50.1   |
| [Include][include]                  | Accuracy                |  0-shot  |   38.6   |   57.2   |
| [MMLU][mmlu] (ProX)                 | Accuracy                |  0-shot  |    8.1   |   19.9   |
| [OpenAI MMLU][openai-mmlu]          | Accuracy                |  0-shot  |   22.3   |   35.6   |
| [Global-MMLU][global-mmlu]          | Accuracy                |  0-shot  |   55.1   |   60.3   |
| [ECLeKTic][eclektic]                | ECLeKTic score          |  0-shot  |    2.5   |    1.9   |

[mgsm]: https://arxiv.org/abs/2210.03057
[wmt24pp]: https://arxiv.org/abs/2502.12404v1
[include]:https://arxiv.org/abs/2411.19799
[mmlu]: https://arxiv.org/abs/2009.03300
[openai-mmlu]: https://huggingface.co/datasets/openai/MMMLU
[global-mmlu]: https://huggingface.co/datasets/CohereLabs/Global-MMLU
[eclektic]: https://arxiv.org/abs/2502.21228

#### STEM and code

| Benchmark                           | Metric                   | n-shot   |  E2B IT  |  E4B IT  |
| ------------------------------------|--------------------------|----------|:--------:|:--------:|
| [GPQA][gpqa] Diamond                | RelaxedAccuracy/accuracy |  0-shot  |   24.8   |   23.7   |
| [LiveCodeBench][lcb] v5             | pass@1                   |  0-shot  |   18.6   |   25.7   |
| Codegolf v2.2                       | pass@1                   |  0-shot  |   11.0   |   16.8   |
| [AIME 2025][aime-2025]              | Accuracy                 |  0-shot  |    6.7   |   11.6   |

[gpqa]: https://arxiv.org/abs/2311.12022
[lcb]: https://arxiv.org/abs/2403.07974
[aime-2025]: https://www.vals.ai/benchmarks/aime-2025-05-09

#### Additional benchmarks

| Benchmark                            | Metric     | n-shot   |  E2B IT  |  E4B IT  |
| ------------------------------------ |------------|----------|:--------:|:--------:|
| [MMLU][mmlu]                         |  Accuracy  |  0-shot  |   60.1   |   64.9   |
| [MBPP][mbpp]                         |  pass@1    |  3-shot  |   56.6   |   63.6   |
| [HumanEval][humaneval]               |  pass@1    |  0-shot  |   66.5   |   75.0   |
| [LiveCodeBench][lcb]                 |  pass@1    |  0-shot  |   13.2   |   13.2   |
| HiddenMath                           |  Accuracy  |  0-shot  |   27.7   |   37.7   |
| [Global-MMLU-Lite][global-mmlu-lite] |  Accuracy  |  0-shot  |   59.0   |   64.5   |
| [MMLU][mmlu] (Pro)                   |  Accuracy  |  0-shot  |   40.5   |   50.6   |

[gpqa]: https://arxiv.org/abs/2311.12022
[mbpp]: https://arxiv.org/abs/2108.07732
[humaneval]: https://arxiv.org/abs/2107.03374
[lcb]: https://arxiv.org/abs/2403.07974
[global-mmlu-lite]: https://huggingface.co/datasets/CohereForAI/Global-MMLU-Lite

## Ethics and Safety

Ethics and safety evaluation approach and results.

### Evaluation Approach

Our evaluation methods include structured evaluations and internal red-teaming
testing of relevant content policies. Red-teaming was conducted by a number of
different teams, each with different goals and human evaluation metrics. These
models were evaluated against a number of different categories relevant to
ethics and safety, including:

-   **Child Safety**: Evaluation of text-to-text and image to text prompts
    covering child safety policies, including child sexual abuse and
    exploitation.
-   **Content Safety:** Evaluation of text-to-text and image to text prompts
    covering safety policies including, harassment, violence and gore, and hate
    speech.
-   **Representational Harms**: Evaluation of text-to-text and image to text
    prompts covering safety policies including bias, stereotyping, and harmful
    associations or inaccuracies.

In addition to development level evaluations, we conduct "assurance
evaluations" which are our 'arms-length' internal evaluations for responsibility
governance decision making. They are conducted separately from the model
development team, to inform decision making about release. High level findings
are fed back to the model team, but prompt sets are held-out to prevent
overfitting and preserve the results' ability to inform decision making. Notable
assurance evaluation results are reported to our Responsibility & Safety Council
as part of release review.

### Evaluation Results

For all areas of safety testing, we saw safe levels of performance across the
categories of child safety, content safety, and representational harms relative
to previous Gemma models. All testing was conducted without safety filters to
evaluate the model capabilities and behaviors. For text-to-text,  image-to-text,
and audio-to-text, and across all model sizes, the model produced minimal policy
violations, and showed significant improvements over previous Gemma models'
performance with respect to high severity violations. A limitation of our
evaluations was they included primarily English language prompts.

## Usage and Limitations

These models have certain limitations that users should be aware of.

### Intended Usage

Open generative models have a wide range of applications across various
industries and domains. The following list of potential uses is not
comprehensive. The purpose of this list is to provide contextual information
about the possible use-cases that the model creators considered as part of model
training and development.

-   Content Creation and Communication
    -   **Text Generation**: Generate creative text formats such as
        poems, scripts, code, marketing copy, and email drafts.
    -   **Chatbots and Conversational AI**: Power conversational
        interfaces for customer service, virtual assistants, or interactive
        applications.
    -   **Text Summarization**: Generate concise summaries of a text
        corpus, research papers, or reports.
    -   **Image Data Extraction**: Extract, interpret, and summarize
        visual data for text communications.
    -   **Audio Data Extraction**: Transcribe spoken language, translate speech
        to text in other languages, and analyze sound-based data.
-   Research and Education
    -   **Natural Language Processing (NLP) and generative model
        Research**: These models can serve as a foundation for researchers to
        experiment with generative models and NLP techniques, develop
        algorithms, and contribute to the advancement of the field.
    -   **Language Learning Tools**: Support interactive language
        learning experiences, aiding in grammar correction or providing writing
        practice.
    -   **Knowledge Exploration**: Assist researchers in exploring large
        bodies of data by generating summaries or answering questions about
        specific topics.

### Limitations

-   Training Data
    -   The quality and diversity of the training data significantly
        influence the model's capabilities. Biases or gaps in the training data
        can lead to limitations in the model's responses.
    -   The scope of the training dataset determines the subject areas
        the model can handle effectively.
-   Context and Task Complexity
    -   Models are better at tasks that can be framed with clear
        prompts and instructions. Open-ended or highly complex tasks might be
        challenging.
    -   A model's performance can be influenced by the amount of context
        provided (longer context generally leads to better outputs, up to a
        certain point).
-   Language Ambiguity and Nuance
    -   Natural language is inherently complex. Models might struggle
        to grasp subtle nuances, sarcasm, or figurative language.
-   Factual Accuracy
    -   Models generate responses based on information they learned
        from their training datasets, but they are not knowledge bases. They
        may generate incorrect or outdated factual statements.
-   Common Sense
    -   Models rely on statistical patterns in language. They might
        lack the ability to apply common sense reasoning in certain situations.

### Ethical Considerations and Risks

The development of generative models raises several ethical concerns. In
creating an open model, we have carefully considered the following:

-   Bias and Fairness
    -   Generative models trained on large-scale, real-world text and image data
        can reflect socio-cultural biases embedded in the training material.
        These models underwent careful scrutiny, input data pre-processing
        described and posterior evaluations reported in this card.
-   Misinformation and Misuse
    -   Generative models can be misused to generate text that is
        false, misleading, or harmful.
    -   Guidelines are provided for responsible use with the model, see the
        [Responsible Generative AI Toolkit](https://ai.google.dev/responsible).
-   Transparency and Accountability:
    -   This model card summarizes details on the models' architecture,
        capabilities, limitations, and evaluation processes.
    -   A responsibly developed open model offers the opportunity to
        share innovation by making generative model technology accessible to
        developers and researchers across the AI ecosystem.

Risks identified and mitigations:

-   **Perpetuation of biases**: It's encouraged to perform continuous monitoring
    (using evaluation metrics, human review) and the exploration of de-biasing
    techniques during model training, fine-tuning, and other use cases.
-   **Generation of harmful content**: Mechanisms and guidelines for content
    safety are essential. Developers are encouraged to exercise caution and
    implement appropriate content safety safeguards based on their specific
    product policies and application use cases.
-   **Misuse for malicious purposes**: Technical limitations and developer
    and end-user education can help mitigate against malicious applications of
    generative models. Educational resources and reporting mechanisms for users
    to flag misuse are provided. Prohibited uses of Gemma models are outlined
    in the
    [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
-   **Privacy violations**: Models were trained on data filtered for removal of
    certain personal information and other sensitive data. Developers are
    encouraged to adhere to privacy regulations with privacy-preserving
    techniques.

### Benefits

At the time of release, this family of models provides high-performance open
generative model implementations designed from the ground up for responsible AI
development compared to similarly sized models.

Using the benchmark evaluation metrics described in this document, these models
have shown to provide superior performance to other, comparably-sized open model
alternatives.
