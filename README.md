# 1X World Model Challenge

Progress in video generation may soon make it possible to evaluate robot policies in a completely learned world model. An end-to-end learned simulator of millions of robot environments would greatly accelerate progress in general-purpose robotics and provide a useful signal for scaling data and compute.

To accelerate progress in learned simulators for robots, we're announcing the 1X World Model Challenge, where the task is to predict future first-person observations of the [EVE Android](https://www.1x.tech/androids/eve). We provide over 100 hours of vector-quantized image tokens and raw actions collected from operating EVE at 1X offices, baseline world model (GENIE-style), and a frame-level MAGVIT2 autoencoder that compresses images into 16x16 tokens and decodes them back into images.

We hope that this dataset will be helpful to roboticists who want to experiment with a diverse set of general-purpose robotics data in human environments. A sufficiently powerful world model will allow anyone to access a "neurally-simulated EVE". The evaluation challenge is the ultimate goal, and we have cash prizes for intermediate goals like fitting the data well (compression challenge) and sampling plausible videos (sampling challenge).

[Dataset on Huggingface](https://huggingface.co/datasets/1x-technologies/worldmodel)

[Join the Discord](https://discord.gg/vppHFmeC)


|||||||||
|---|---|---|---|---|---|---|---|
|![til](./assets/v1.0/generated_offset700100.gif)|![til](./assets/v1.0/generated_offset225100.gif)|![til](./assets/v1.0/generated_offset775100.gif)|![til](./assets/v1.0/generated_offset875100.gif)|![til](./assets/v1.0/generated_offset475100.gif)|![til](./assets/v1.0/generated_offset725100.gif)|![til](./assets/v1.0/generated_offset525100.gif)|![til](./assets/v1.0/generated_offset100.gif)|
|![til](./assets/v1.0/generated_offset925100.gif)|![til](./assets/v1.0/generated_offset975100.gif)|![til](./assets/v1.0/generated_offset625100.gif)|![til](./assets/v1.0/generated_offset675100.gif)|![til](./assets/v1.0/generated_offset400100.gif)|![til](./assets/v1.0/generated_offset175100.gif)|![til](./assets/v1.0/generated_offset850100.gif)|![til](./assets/v1.0/generated_offset100100.gif)|
|![til](./assets/v1.0/generated_offset125100.gif)|![til](./assets/v1.0/generated_offset375100.gif)|![til](./assets/v1.0/generated_offset275100.gif)|![til](./assets/v1.0/generated_offset800100.gif)|![til](./assets/v1.0/generated_offset600100.gif)|![til](./assets/v1.0/generated_offset1000100.gif)|![til](./assets/v1.0/generated_offset450100.gif)|![til](./assets/v1.0/generated_offset50100.gif)|
|![til](./assets/v1.0/generated_offset250100.gif)|![til](./assets/v1.0/generated_offset150100.gif)|![til](./assets/v1.0/generated_offset825100.gif)|![til](./assets/v1.0/generated_offset950100.gif)|![til](./assets/v1.0/generated_offset25100.gif)|![til](./assets/v1.0/generated_offset750100.gif)|![til](./assets/v1.0/generated_offset650100.gif)|![til](./assets/v1.0/generated_offset300100.gif)|


## Challenges

Each example is a sequence of 16 first-person images from the robot at 2Hz (so 8 seconds total), and your goal is to predict the next image given the previous ones.

- **Compression Challenge ($10k prize)**: Predict the discrete distribution of tokens in the next image.
  - Criteria: Be the first to achieve a **[temporally teacher-forced](#metric-details) loss below 8.0** on our private test set.
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

## GENIE

This repo provides an implementation of the spatio-temporal transformer and MaskGIT sampler as described in [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). Note that this implementation only trains on video sequences, not actions (though it is trivial to add this via an additive embedding).

```
# Train the GENIE model
python train.py --genie_config genie/configs/magvit_n32_h8_d256.json --output_dir data/genie_model --max_eval_steps 10

# Generate frames from trained model
python genie/generate.py --checkpoint_dir data/genie_model/final_checkpt

# Visualize generated frames
python visualize.py --token_dir data/genie_generated

# Evaluate the trained model
python genie/evaluate.py --checkpoint_dir data/genie_model/final_checkpt
```

### 1X GENIE Baseline
We provide two pre-trained GENIE models, linked in the [leaderboard](#leaderboard).
```
# Generate and visualize
output_dir='data/genie_baseline_generated'
for i in {0..240..10}; do
    python genie/generate.py --checkpoint_dir 1x-technologies/GENIE_138M \
        --output_dir $output_dir --example_ind $i --maskgit_steps 2 --temperature 0
    python visualize.py --token_dir $output_dir
    mv $output_dir/generated_offset0.gif $output_dir/example_$i.gif
    mv $output_dir/generated_comic_offset0.png $output_dir/example_$i.png
done

# Evaluate
python genie/evaluate.py --checkpoint_dir 1x-technologies/GENIE_138M --maskgit_steps 2
```
 
## Data Description

[See the Dataset Card on Huggingface](https://huggingface.co/datasets/1x-technologies/worldmodel).

The training dataset is stored in the `data/train_v1.1` directory.

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

## Additional Challenge Details

1. We've provided `magvit2.ckpt` in the dataset download, which are the weights for a [MAGVIT2](https://github.com/TencentARC/Open-MAGVIT2) encoder/decoder. The encoder allows you to tokenize external data to try to improve the metric.
2. The loss metric is nonstandard compared to LLMs due to the vocabulary size of the image tokens, which was changed as of v1.0 release (Jul 8, 2024). Instead of computing cross entropy loss on logits with 2^18 classes, we compute cross entropy losses on 2x 2^9 class predictions and sum them up. The rationale for this is that the large vocabulary size (2^18) makes it very memory-intensive to store a logit tensor of size `(B, 2^18, T, 16, 16)`. Therefore, the compression challenge considers families of models with a factorized pmfs of the form p(x1, x2) = p(x1)p(x2). For sampling and evaluation challenge, a factorized pmf is a necessary criteria.
3. For the compression challenge, we are making the deliberate choice to evaluate held-out data on the same factorized distribution p(x1, x2) = p(x1)p(x2) that we train on. Although unfactorized models of the form p(x1, x2) = f(x1, x2) ought to achieve lower cross entropy on test data by exploiting the off-block-diagonal terms of Cov(x1, x2), we want to encourage solutions that achieve lower losses while holding the factorization fixed.
4. For the compression challenge, submissions may only use the *prior* actions to the current prompt frame. Submissions can predict subsequent actions autoregressively to improve performance, but these actions will not be provided with the prompt.
5. Naive nearest-neighbor retrieval + seeking ahead to next frames from the training set will achieve reasonably good losses and sampling results on the dev-validation set, because there are similar sequences in the training set. However, we explicitly forbid these kinds of solutions (and the private test set penalizes these kinds of solutions).
6. We will not be able to award prizes to individuals in U.S. sanctioned countries. We reserve the right to not award a prize if it violates the spirit of the challenge.


### Metric Details
There are different scenarios for evaluation, which vary in the degree of ground truth context the model receives.
In decreasing order of context, these scenarios are:
- **Fully Autoregressive**: the model receives a predetermined number of ground truth frames and autoregressively predicts all remaining frames.
- **Temporally Teacher-forced**: the model receives all ground truth frames before the current frame and autoregressively predicts all tokens in the current frame.
- **Fully Teacher-forced**: the model receives all ground truth tokens before the current token, 
including tokens in the current frame. Only applicable for causal LMs.

As an example, consider predicting the final token of a video, corresponding to the lower right patch of frame 15. 
The context the model receives in each scenario is:
- Fully Autoregressive: the first $t$x16x16 tokens are ground truth tokens corresponding to the first $t$ prompt frames, 
and all remaining tokens are autoregressively generated, where $0 < t < 15$ is the predetermined number of prompt frames.
- Temporally Teacher-forced: the first 15x16x16 tokens are ground truth tokens corresponding to the first 15 frames, 
and all remaining tokens are autoregressively generated.
- Fully Teacher-forced: all previous (16x16x16 - 1) tokens are ground truth tokens.

The compression challenge uses the "temporally teacher-forced" scenario.
## Leaderboard

These are evaluation results on `data/val_v1.1`.
<table>
  <thead>
    <tr>
      <th>User</th>
      <th>Temporally Teacher-forced<br>CE Loss</th>
      <th>Temporally Teacher-forced<br>Token Accuracy</th>
      <th>Temporally Teacher-forced<br>LPIPS</th>
      <th>Generation Time* <br>(secs/frame)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/1x-technologies/GENIE_138M">1x-technologies/GENIE_138M</a><br>(<code>--maskgit_steps 2</code>)</td>
      <td>8.79</td>
      <td>0.0320</td>
      <td>0.207</td>
      <td>0.075</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/1x-technologies/GENIE_35M">1x-technologies/GENIE_35M</a><br>(<code>--maskgit_steps 2</code>)</td>
      <td>8.99</td>
      <td>0.0301</td>
      <td>0.217</td>
      <td>0.030</td>
    </tr>
  </tbody>
</table>
*Generation time is the time to generate latents on a RTX 4090 GPU, and excludes the time to decode latents to images.


## Help us Improve the Challenge!

Beyond the World Model Challenge, we also want to make the challenges and datasets more useful for *your* research questions. Want more data interacting with humans? More safety-critical tasks like carrying cups of hot coffee without spilling? More dextrous tool use? Robots working with other robots? Robots dressing themselves in the mirror? Think of 1X as the operations team for getting you high quality humanoid data in extremely diverse scenarios.

Email challenge@1x.tech with your requests (and why you think the data is important) and we will try to include it in a future data release. You can also discuss your data questions with the community on [Discord](https://discord.gg/UMnzbTkw). 

We also welcome donors to help us increase the bounty.


## Citation

If you use this software or dataset in your work, please cite it using the "Cite this repository" button on Github.

## Changelog

- v1.1 - Release compression challenge criteria; removed pauses and discontinuous videos from dataset; higher image crop.
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
    <td><code itemprop="description">A dataset of over 100 hours of compressed image tokens + raw actions across a fleet of EVE robots.</code></td>
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
