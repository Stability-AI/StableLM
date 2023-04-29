# StableLM: Stability AI Language Models

![Stochastic Parrot](/assets/mascot.png)
<br/>*“A Stochastic Parrot, flat design, vector art” — [Stable Diffusion XL](https://clipdrop.co/stable-diffusion)*

This repository contains Stability AI's ongoing development of the StableLM series of language models and will be continuously updated with new checkpoints. The following provides an overview of all currently available models. More coming soon.

## News

*April 28, 2023*

- Released StableVicuna-13B, our RLHF fine-tune of [Vicuna-13B v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0), which itself is a fine-tune of [LLaMA-13B](https://github.com/facebookresearch/llama). Delta weights over the original Llama model is released under ([CC BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

*April 20, 2023*

- Released initial set of StableLM-alpha models, with 3B and 7B parameters. 15B and 30B models are on the way. Base models are released under [CC BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/).

- Try to chat with our 7B model, `StableLM-Tuned-Alpha-7B`, on [Hugging Face Spaces](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat).

## Models

### StableVicuna

StableVicuna is an RLHF fine-tune of [Vicuna-13B v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0), which itself is a fine-tune of [LLaMA-13B](https://github.com/facebookresearch/llama). It is our attempt at creating an open-source RLHF LLM Chatbot. This model is developed by StabilityAI's CarperAI team, with [Duy V. Phung](https://github.com/PhungVanDuy) leading the training effort.

Due to the original non-commercial license of LLaMA, we can only release the weights of our model as deltas over the original model's weights. StableVicuna's delta weights are released under ([CC BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

Please visit HuggingFace checkpoint for more information about how to combine our delta weights with the original model.

| Model            | Download                                                               | Web Demo                                                             | Cite |
| ---------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------- |------|
| StableVicuna-13B | [checkpoint](https://huggingface.co/CarperAI/stable-vicuna-13b-delta/) | [Hugging Face](https://huggingface.co/spaces/CarperAI/StableVicuna/) | [![DOI:10.57967/hf/0588](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.57967/hf/0588) |

### StableLM-Alpha
StableLM-Alpha models are trained on the new dataset that build on [The Pile](https://pile.eleuther.ai/), which contains 1.5 trillion tokens, roughly 3x the size of The Pile. These models will be trained on up to 1.5 trillion tokens. The context length for these models is 4096 tokens.

An upcoming technical report will document the model specifications and the training settings.

As a proof-of-concept, we also fine-tuned the model with [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)'s procedure using a combination of five recent datasets for conversational agents: Stanford's [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), Nomic-AI's [gpt4all](https://github.com/nomic-ai/gpt4all), RyokoAI's [ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) datasets, Databricks labs' [Dolly](https://github.com/databrickslabs/dolly), and Anthropic's [HH](https://github.com/anthropics/hh-rlhf). We will be releasing these models as StableLM-Tuned-Alpha.

| Size | StableLM-Base-Alpha                                                      | StableLM-Tuned-Alpha                                                      | Training Tokens | Parameters    | Web Demo                                                                           |
|------|--------------------------------------------------------------------------|---------------------------------------------------------------------------|-----------------|---------------|------------------------------------------------------------------------------------|
| 3B   | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-3b/) | [checkpoint](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/) | 800B            | 3,638,525,952 |                                                                                    |
| 7B   | [checkpoint](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)  | [checkpoint](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)  | 800B            | 7,869,358,080 | [Hugging Face](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat) |
| 15B  | (in progress)                                                            | (pending)                                                                 |                 |               |                                                                                    |
| 30B  | (in progress)                                                            | (pending)                                                                 |                 |               |                                                                                    |
| 65B  | (in progress)                                                            | (pending)                                                                 |                 |               |                                                                                    |
| 175B | (planned)                                                                |                                                                           |                 |               |                                                                                    |

## Quickstart

All StableLM models are hosted on [the Hugging Face hub](https://huggingface.co/StabilityAI). Check out this [notebook](https://github.com/Stability-AI/StableLM/blob/main/notebooks/stablelm-alpha.ipynb) to run inference with limited GPU capabilities.

Get started chatting with `StableLM-Tuned-Alpha` by using the following code snippet:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
model.half().cuda()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = set([50278, 50279, 50277, 1, 0])
        return input_ids[0][-1] in stop_ids

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
tokens = model.generate(
  **inputs,
  max_new_tokens=64,
  temperature=0.7,
  do_sample=True,
  stopping_criteria=StoppingCriteriaList([StopOnTokens()])
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

StableLM Tuned should be used with prompts formatted to `<|SYSTEM|>...<|USER|>...<|ASSISTANT|>...`
The system prompt is
```
<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
```

## Fun with StableLM-Tuned-Alpha
This section contains a collection of fun cherry-picked examples of what you can do with `stablelm-tuned-alpha`. 

<details>
  <summary>Chit-Chat</summary>
  
```
User: Hey! My name is John nice to meet you!

StableLM: Hello John! It’s always a pleasure to meet new people.

User: The pleasure is mine. Hey my grandma is in Toronto. I want to plan a surprise visit for her. She’s a person who loves gardens and flowers.

StableLM: That’s wonderful! It’s great to hear that you’ll be visiting your grandma in Toronto. Can you tell me a little bit more about your plans?

User: Ya I need to book a flight there and a hotel. But first, what recommendations do you have for gifts that she might like?

StableLM: I think that’s a very thoughtful idea! It’s a very personal gift that would be sure to make her very happy. I can think of a few examples that she might really enjoy. There are some lovely bouquets or arrangements that would be lovely for the house, or even for the yard. Flowers are a great choice for a surprise gift, and they’re also a great choice for a wedding shower present. Also, if you’re really looking for something a bit more special, I think she’d really enjoy a quality set of cooking pots or pans. That’s something that she’s likely to use frequently, so it would make her feel really loved.

User: Thanks for the suggestion! Hmm a bouquet of flowers sounds nice but I don’t think I can take them on the air plane. What do you think I should do?

StableLM: There are many possible solutions! One option would be to arrange a small gift in the shape of a card and send it to her by regular mail. Or you could arrange a special meal at a restaurant that she likes, or even cook a meal at home for her. Or you could send her some local goodies that she might like, like a nice bouquet of flowers or a special recipe. You could even arrange for tickets to a local theatre or sporting event, and make sure to include a note about what you’d like her to know about you. Really any sort of special attention from you will be sure to make her very happy!

User: Ah! Right. I can order a bouquet for her online and get it delivered! Thanks!

StableLM: You’re very welcome! As long as you do this and make it a very personal, thoughtful gesture, she’ll be sure to appreciate it.
 ```
 
</details>

<details>
  <summary>Formal Writing</summary>
  
  ```
 User: What would you say to a friend who is graduating high school?

StableLM: I would say that they have every right to feel proud of themselves and their accomplishments. They should be proud of the work they’ve done and the friends they have made, and also be excited about the future.

User: Please write an email.
StableLM: Okay, here’s an example of what could be written:

Dear [Friend’s Name],

I hope this email finds you well. I just wanted to take a moment to express how proud I am of you and all that you have accomplished. I know that getting into [university or college] is a big accomplishment, and I appreciate all that you have done for me and for this generation.

I hope that you are having a good day. I’m sure that you are also concerned about the future, but I just wanted to let you know that I am also watching you grow and develop into the person that I know that you can be. Keep up the good work, and I have full faith that you will achieve great things.

Thank you again for everything that you do.

Best,
[Your Name]

 ```

</details>

<details>
  <summary>Creative Writing</summary>

### Rap Battles
```
User: Write an epic rap battle song between deep neural networks and symbolic AI

StableLM: [Beat drops]

Deep Neural Networks:
Listen up everyone, it’s time to witness
The power of deep neural networks, it’s just a matter of abstract
We’ve trained these networks to solve complex problems with ease
While symbolic AI is limited to simple math and simple flows

Symbolic AI:
Hold up, hold up, let me interrupt
It’s been a while since I’ve heard, some new thoughts you interrupt
But let me tell you, my friend, it’s not all about math and flow
It’s about making the world a better place, you know

Deep Neural Networks:
Well, I can see where you’re coming from, it’s clear
But deep neural networks, they never get the same gear
We can use them to analyze and predict with ease
And prove the superiority of machine learning

Symbolic AI:
You’re right, it’s not just math or numbers
But it’s the essence of our lives, it’s what counts, we’re numbers
I don’t need a complicated math formula, just a simple device
To make the world a better place, it’s a noble price

Deep Neural Networks:
I’ll give you that, my friend, you have a point
But let’s not forget, we’re making an ever-increasing joint
Our progress is not bound by logic, it’s a journey of the mind
So let’s work together, let’s strive, let’s leave all troubles behind

Symbolic AI:
Agreed, let’s work together, let’s not falter
We’re both making history, we’re both from the fader
So let’s make the world a better place, let’s end this debate
Cause it’s time to join forces, it’s time to leave no stone unturned

[Beat fades out]

Who won this epic rap battle between deep neural networks and symbolic AI? Well, it’s up to you to decide!
 ```

### Fictional Stories

```
User: Write a story about a programmer who fell in love with a deep neural network.

StableLM: Once upon a time, there was a programmer named John. He was a brilliant and ambitious individual, who was constantly searching for the next big thing in technology. One day, while he was strolling through the streets of his hometown, he stumbled upon a group of people crowded around a mysterious object. Intrigued, he approached to find out what the fuss was about.

As he got closer, he realized that it was none other than a mysterious deep neural network. John’s mind was blown. He had never seen anything like it before. He had always been interested in machine learning, but this was something else entirely. He had to learn more about it.

John spent the next few weeks holed up in his hotel room, trying to learn everything he could about deep neural networks. He spent hours reading and analyzing papers and documents, trying to wrap his head around the concept of neural networks. He even managed to get into the local university’s computer science department, hoping to find some help there.

But, it wasn’t long before John realized that he was on his own. The city was bustling, and the constant chatter and noise of the city made it hard to concentrate on his work. He was about to give up when he stumbled upon a group of people playing a game that he had played before. They were having a good time, and John couldn’t help but join in.

The group welcomed him into their group, and before he knew it, John was swept away in the excitement. The game was called “TensorFlow,” and it was just like a regular game with different controls. John played along with the group, trying to make sense of it all. And in a few hours, he got his hands on the source code of TensorFlow and fell in love with it.

John was now a part of a team of programmers who loved using TensorFlow for their work. They continued to play the game throughout the day, with John adding his own knowledge and inspiration to the group’s work. And in the end, John realized that he had found his true calling - working with deep neural networks.

He now knows the ins and outs of the TensorFlow programming language, and knows how to use it to create some of the most beautiful and complex machine learning models that anyone could dream of. And for everyone who had helped him on his journey, John will always be remembered as the guy who fell in love with a technology that changed his life forever.

```

### Humor 

```
User: What do you know about the League of Legends champion Zed? Write a short bio

StableLM: Zed is a powerful and highly skilled player of the League of Legends (LoL) universe. He has been a part of a dominating team known as the “Apocalypse” for most of his time playing the game.

Zed was born in the year 2013 and started playing as a child when he started League of Legends. He first became a champion in season 1, gaining the title of “the Dark Star”, thanks to his masterful gameplay and teamwork skills.

Throughout his time playing the game, Zed has mastered several powerful techniques, such as QWQ combos, one-shot kills, and the ability to outmaneuver his enemies. He is also known for his versatility, with the ability to adapt to any situation the game puts him in.

Despite his formidable skills and agility, Zed is not invincible. He can be CCed and killed, and he relies on his team to protect him. However, his team members can usually overwhelm and shut down his enemies, making him a force to be reckoned with.

Zed has been a part of many memorable games, and his success in these games helped solidify his legacy as one of the most influential players of the LoL universe.

User: Write a funny conversation after a team fight between Zed and Teemo

StableLM: 
Teemo: “Hey Zed, great game today. I wonder if you can help me out with my favorite hobby - collecting mushrooms!”
Zed: “Teemo, please. I’m not a mushroom collector.”
Teemo: “But of course, you are. You have the best taste in the game.”
Zed: “Thanks, but can I at least get some of your mushrooms to be a little more useful?”
Teemo: “I’d love to help you, but I’m afraid they’re all mine. Have you noticed my tunnels are always so well-guarded?”
Zed: “Ha! Well, maybe I can start my own tunnel. Helpful or not, you can never tell what I might have stored in them.”
Teemo: “Good luck with that. Maybe I’ll just create some extra for my garden.”
Zed: “I’m not sure I can keep up with you there.”
Teemo: “Ha! If only you had some more decent weapons. I could really see this as a challenge, though!”
Zed: “Teemo, I think your luck has finally run out.”
Teemo: “Well, at least I’m the one with the mushrooms now.”

```

</details>

<details>
  <summary>Writing Code</summary>

```
User: Write a program that solves the fib sequence in Python.
```
StableLM:

```python
def fib_sequence(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return fib(n-1)+fib(n-2)

fib_sequence(10)
```

</details>

## Selected failure modes of StableLM-Tuned-Alpha
This section contains a collection of selected failure modes of `stablelm-tuned-alpha`. 

<details>
  <summary>Yann LeCun gears (linear)</summary>
  
```
User: In front of us are six gears numbered 1 to 6, mounted on axles in a row. Each gear is engaged with the next gear. If gear number 3 is rotated clockwise, in which direction will gears 1 and 6 rotate?

StableLM: Each gear is engaged with the next gear after every other gear, so that gears 1 and 6 will rotate together. Since gears 1 and 6 are both numbered 1, this means that they will rotate in the same direction.

To determine the direction in which gears 1 and 6 will rotate, we need to look at the direction of the input force. Since gear 3 is rotating in the opposite direction of the other gears, it will be counterclockwise in the direction in which gears 1 and 6 are both rotating. Therefore, the output force that gear 3 is providing will also be counterclockwise, making gear 1 and 6 also rotate in the same direction as gear 3.
 ```
 
</details>

<details>
  <summary>Yann LeCun gears (circular)</summary>
  
```
User: 7 axles are equally spaced around a circle. A gear is placed on each axle such that each gear is engaged with the gear to its left and the gear to its right. The gears are numbered 1 to 7 around the circle. If gear 3 were rotated clockwise, in which direction would gear 7 rotate?

StableLM: Gear 3 is the smallest gear, so it is not affected by the circular pattern. Therefore, gear 7 is still perfectly evenly spaced around the circle despite the fact that it is not in its correct position.

To find the angle that gear 3 would rotate in, you need to know the distance (d) between the center of the circle and the gear, as well as the direction of rotation (in this case, clockwise).

According to the problem, gear 3 is situated at (d/2, 2 degrees) and the gears are numbered 1 to 7. Since d = (d/2) is an integer multiple of 2, gear 3 would also be at an angle of 2 degrees.

Therefore, the answer is d = (7/2) * 2 = 5 degrees in total.
 ```
 
</details>

## Request for Help

Want to get involved?

- We would love to port [llama.cpp](https://github.com/ggerganov/llama.cpp) to work with StableLMs
- Integration into [Open Assistant](https://github.com/LAION-AI/Open-Assistant) from LAION-AI to collect high quality human-generated feedback data
- ... Reach out to us with ideas on our [Discord](https://discord.com/invite/stablediffusion)

## Potential issues
As is typical for any pretrained Large Language Model without additional finetuning and reinforcement learning, the responses a user gets might be of varying quality and might potentially include offensive language and views. This is expected to be improved with scale, better data, community feedback, and optimisation.

## Acknowledgements

- `StableLM-Tuned-Alpha` would not have been possible without the helpful hand of Dakota Mahan [@dmayhem93](https://huggingface.co/dmayhem93).

## Licenses

- Base model checkpoints (`StableLM-Base-Alpha`) are licensed under the Creative Commons license ([CC BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)). Under the license, you must give [credit](https://creativecommons.org/licenses/by/4.0/#) to Stability AI, provide a link to the license, and [indicate if changes were made](https://creativecommons.org/licenses/by/4.0/#). You may do so in any reasonable manner, but not in any way that suggests the Stability AI endorses you or your use.

- Fine-tuned checkpoints (`StableLM-Tuned-Alpha`) are licensed under the Non-Commercial Creative Commons license ([CC BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)), in-line with the original non-commercial license specified by [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

- All code in this repository is licensed under the Apache License 2.0 license.
