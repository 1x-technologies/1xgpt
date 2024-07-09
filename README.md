# 1X World Model Challenge

Progress in video generation may soon make it possible to evaluate robot policies in a completely learned world model. An end-to-end learned simulator of millions of robot environments would greatly accelerate progress in general-purpose robotics and provide a useful signal for scaling data and compute.

To accelerate progress in learned simulators for robots, we're announcing the 1X World Model Challenge, where the task is to predict future first-person observations of the [EVE Android](https://www.1x.tech/androids/eve). We provide over 100 hours of vector-quantized image and action tokens collected from operating EVE at 1X offices, baseline world models (LLM, GENIE), and a frame-level MAGVIT2 autoencoder that compresses images into 16x16 tokens and decodes them back into images.

We hope that this dataset will be helpful to roboticists who want to experiment with a diverse set of general-purpose robotics data in human environments. A sufficiently powerful world model will allow anyone to access a "neurally-simulated EVE". The evaluation challenge is the ultimate goal, and we have cash prizes for intermediate goals like fitting the data well (compression challenge) and sampling plausible videos (sampling challenge).

[Dataset on Huggingface](https://huggingface.co/datasets/1x-technologies/worldmodel)

[Join the Discord](https://discord.gg/vppHFmeC)


|||||||||
|---|---|---|---|---|---|---|---|
|![til](./assets/generated_offset2521107.gif)|![til](./assets/generated_offset6722954.gif)|![til](./assets/generated_offset8963939.gif)|![til](./assets/generated_offset3734974.gif)|![til](./assets/generated_offset8777190.gif)|![til](./assets/generated_offset4855467.gif)|![til](./assets/generated_offset5789210.gif)|![til](./assets/generated_offset186748.gif)|
|![til](./assets/generated_offset9617559.gif)|![til](./assets/generated_offset6629580.gif)|![til](./assets/generated_offset11485047.gif)|![til](./assets/generated_offset93374.gif)|![til](./assets/generated_offset4762092.gif)|![til](./assets/generated_offset3454851.gif)|![til](./assets/generated_offset7843446.gif)|![til](./assets/generated_offset5322338.gif)|
|![til](./assets/generated_offset4481969.gif)|![til](./assets/generated_offset2707856.gif)|![til](./assets/generated_offset8030195.gif)|![til](./assets/generated_offset10457929.gif)|![til](./assets/generated_offset2054236.gif)|![til](./assets/generated_offset8683816.gif)|![til](./assets/generated_offset5415713.gif)|![til](./assets/generated_offset10364554.gif)|
|![til](./assets/generated_offset2427733.gif)|![til](./assets/generated_offset7563323.gif)|![til](./assets/generated_offset1120492.gif)|![til](./assets/generated_offset11858544.gif)|![til](./assets/generated_offset7469949.gif)|![til](./assets/generated_offset9524185.gif)|![til](./assets/generated_offset2240984.gif)|![til](./assets/generated_offset9710934.gif)|

## Challenges

Each example is a sequence of 16 first-person images from the robot at 2Hz (so 8 seconds total), and your goal is to predict the next image given the previous ones.

- **Compression Challenge ($10k prize)**: Predict the discrete distribution of tokens in the next image. To participate in the challenge, fill in your model predictions in `evaluate.py`. Criteria will be released shortly.
- **Sampling Challenge ($10k prize)**: Future prediction methods are not necessarily restricted to next-logit prediction. You can, for example, use methods like GANs, Diffusion, and MaskGIT to generate future images. Criteria will be released shortly.
- **Evaluation Challenge (upcoming)**: given a set of N policies, $\pi_1, \pi_2, ... \pi_N$, where each policy $\pi_i(a_t|z_t)$ predicts action tokens from image tokens, can you evaluate all of the policies inside a "world model" $p(z_{t+1}|z_t, a_t)$ and tell us the ranked order of which policy is the best?

These challenges are largely inspired by the [commavq compression challenge](https://github.com/commaai/commavq). Please read the [Additional Challenge Details](#additional-challenge-details)

## Getting Started
We require `Python 3.10` or later. This code was tested with `Python 3.10.12`.

```
# Install dependencies and download data
./build.sh 

# Source the Python environment
source venv/bin/activate
```

## Training GENIE

This repo provides an implementation of the spatio-temporal transformer and MaskGIT sampler as described in [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). Note that this implementation only trains on video sequences, not actions (though it is trivial to add this via an additive embedding). To train this baseline, 

```
# Train the GENIE model
python train.py --genie_config genie/configs/magvit_n32_h8_d256.json --output_dir data/genie_model

# Generate frames from trained model
python genie/generate.py --checkpoint_dir data/genie_model/final_checkpt

# Visualize generated frames
python visualize.py --token_dir data/genie_generated

# Evaluate the trained model
python genie/evaluate.py --checkpoint_dir data/genie_model/final_checkpt

# Generate or evaluate the 1X baseline model
python genie/generate.py --checkpoint_dir 1x-technologies/GENIE_35M --output_dir data/genie_baseline_generated --example_ind 150  # 150 is cherry-picked
python visualize.py --token_dir data/genie_baseline_generated

python genie/evaluate.py --checkpoint_dir 1x-technologies/GENIE_35M
```
 
## Data Description

[See the Dataset Card on Huggingface](https://huggingface.co/datasets/1x-technologies/worldmodel).

The training dataset is stored in the `data/train_v1.0` directory.

## Participating in the Challenges: 

Please read the [Additional Challenge Details](#additional-challenge-details) first for clarification on rules.

Email source code + build script + some info about your approach to challenge@1x.tech. We will evaluate your submission on our held-out dataset and email you back with the results. 

Please send us the following:
- your chosen username (can be your real name or a pseudonym, will be tied 1:1 to your email)
- source code as a .zip file
- how many flops you used (approximately) to train the model
- any external data you may have used to train your model
- eval performance you got on the provided validation set (so we know roughly what you expect from your model)

After manually reviewing your code, we run evals in a 22.04 + CUDA 12.3 sandboxed environment like so:

```
./build.sh # installs any dependencies + model weights you need
./evaluate.py --val_data_dir <PATH-TO-HELD-OUT-DATA>  # runs your model on held-out data
```

## Challenge Details

1. We've provided `magvit2.ckpt` in the dataset download, which are the weights for a [MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2) encoder/decoder. The encoder allows you to tokenize external data to try to improve the metric.
2. The loss metric is nonstandard compared to LLMs due to the vocabulary size of the image tokens, which was changed as of v1.0 release (Jul 8, 2024). Instead of computing cross entropy loss on logits with 2^18 classes, we compute cross entropy losses on 2x 2^9 class predictions and sum them up. The rationale for this is that the large vocabulary size (2^18) makes it very memory-intensive to store a logit tensor of size `(B, 2^18, T, 16, 16)`. Therefore, the compression challenge considers families of models with a factorized pmfs of the form p(x1, x2) = p(x1)p(x2). For sampling and evaluation challenge, a factorized pmf is a necessary criteria.
3. For the compression challenge, we are making the deliberate choice to evaluate held-out data on the same factorized distribution p(x1, x2) = p(x1)p(x2) that we train on. Although unfactorized models of the form p(x1, x2) = f(x1, x2) ought to achieve lower cross entropy on test data by exploiting the off-block-diagonal terms of Cov(x1, x2), we want to encourage solutions that achieve lower losses while holding the factorization fixed.
4. Naive nearest-neighbor retrieval + seeking ahead to next frames from the training set will achieve reasonably good losses and sampling results on the dev-validation set, because there are similar sequences in the training set. However, we explicitly forbid these kinds of solutions (and the private test set penalizes these kinds of solutions).


### Metric Details
We evaluate the model under two different scenarios; in both cases, the model receives the tokens of the previous frame(s) as input, 
and the model should predict the tokens of the following frame. 
- **Autoregressive** (frame-level) is closer to an actual generation scenario, where the model receives $t$ x 20x20 tokens representing frames 0 to $t - 1$, 
and the model should auto-regressively predict all 20x20 tokens for frame $t$.
- (If applicable), **Teacher-forced** matches the typical training scenario with causal masking. 
It is simply a next token prediction task where all previous tokens, including any in the current frame, are ground-truth tokens as opposed to autoregressively predicted tokens.

## Leaderboard

All scores are evaluated on our held-out dataset.

| **User**                    | **Teacher-Forced CE Loss** | **Teacher-Forced Token Accuracy** | **Autoregressive CE Loss** | **Autoregressive Token Accuracy** | **Autoregressive LPIPS** | **Generation Time\* (secs/frame)** |
|-----------------------------|----------------------------|-----------------------------------|----------------------------|-----------------------------------|--------------------------|------------------------------------|
| 1x-technologies/GENIE_35M   | N/A                        | N/A                               | 9.305                      | 0.0385                            | 0.120                    | 0.017                              |

*Note that generation time is the time to generate latents on a RTX 4090 GPU, and excludes the time to decode latents to images.


## Help us Improve the Challenge!

Beyond the World Model Challenge, we also want to make the challenges and datasets more useful for *your* research questions. Want more data interacting with humans? More safety-critical tasks like carrying cups of hot coffee without spilling? More dextrous tool use? Robots working with other robots? Robots dressing themselves in the mirror? Think of 1X as the operations team for getting you high quality humanoid data in extremely diverse scenarios.

Email challenge@1x.tech with your requests (and why you think the data is important) and we will try to include it in a future data release. You can also discuss your data questions with the community on [Discord](https://discord.gg/UMnzbTkw). 

We also welcome donors to help us increase the bounty.


## Citation

If you use this software or dataset in your work, please cite it using the "Cite this repository" button on Github.

## Changelog

- v1.0 - More efficient MAGVIT2 tokenizer with 16x16 (C=2^18) mapping to 256x256 images, providing raw action data.
- v0.0.1 - Initial challenge release with 20x20 (C=1000) image tokenizer mapping to 160x160 images.


## Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">1X World Model Challenge</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/1x-technologies/1xgpt</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">A dataset of over 100 hours of compressed image + action tokens across a fleet of EVE robots.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">1X Technologies</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Apache 2.0</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
</table>
</div>