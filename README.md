# 1X World Model Challenge

Progress in video generation may soon make it possible to evaluate robot policies in a completely learned world model. An end-to-end learned simulator of millions of robot environments would greatly accelerate progress in general-purpose robotics and provide a useful signal for scaling data and compute.

To accelerate progress in learned simulators for robots, we're announcing a World Model competition, where the task is to predict future first-person observations of the [EVE Android](https://www.1x.tech/androids/eve). We provide over 100 hours of vector-quantized image and action tokens collected from operating EVE at 1X offices, baseline world models (LLM, GENIE), and a frame-level diffusion decoder to decode vector-quantized images into 160x160 images. See below for details on cash prizes.

Beyond the scope of these challenges, we hope that this dataset will be helpful to roboticists who want to experiment with a diverse set of general-purpose robotics data in human environments. A sufficiently powerful world model will allow anyone to access a "neurally-simulated EVE".

[Dataset on Huggingface](https://huggingface.co/datasets/1x-technologies/worldmodel)

[Join the Discord](https://discord.gg/UMnzbTkw)


|||||||||
|---|---|---|---|---|---|---|---|
|![til](./assets/generated_offset2521107.gif)|![til](./assets/generated_offset6722954.gif)|![til](./assets/generated_offset8963939.gif)|![til](./assets/generated_offset3734974.gif)|![til](./assets/generated_offset8777190.gif)|![til](./assets/generated_offset4855467.gif)|![til](./assets/generated_offset5789210.gif)|![til](./assets/generated_offset186748.gif)|
|![til](./assets/generated_offset9617559.gif)|![til](./assets/generated_offset6629580.gif)|![til](./assets/generated_offset11485047.gif)|![til](./assets/generated_offset93374.gif)|![til](./assets/generated_offset4762092.gif)|![til](./assets/generated_offset3454851.gif)|![til](./assets/generated_offset7843446.gif)|![til](./assets/generated_offset5322338.gif)|
|![til](./assets/generated_offset4481969.gif)|![til](./assets/generated_offset2707856.gif)|![til](./assets/generated_offset8030195.gif)|![til](./assets/generated_offset10457929.gif)|![til](./assets/generated_offset2054236.gif)|![til](./assets/generated_offset8683816.gif)|![til](./assets/generated_offset5415713.gif)|![til](./assets/generated_offset10364554.gif)|
|![til](./assets/generated_offset2427733.gif)|![til](./assets/generated_offset7563323.gif)|![til](./assets/generated_offset1120492.gif)|![til](./assets/generated_offset11858544.gif)|![til](./assets/generated_offset7469949.gif)|![til](./assets/generated_offset9524185.gif)|![til](./assets/generated_offset2240984.gif)|![til](./assets/generated_offset9710934.gif)|

## Challenges

Each example is a sequence of 16 first-person images from the robot at 2Hz (so 8 seconds total), and your goal is to predict the next image given the previous ones.

- **Compression Challenge ($10k prize)**: To participate in the challenge, fill in your model predictions in `evaluate.py`. Criteria will be released shortly.
- **Sampling Challenge ($10k prize)**: Future prediction methods are not necessarily restricted to next-logit prediction. You can, for example, use methods like GANs, Diffusion, and MaskGIT to generate future images. Criteria will be released shortly.
- **Evaluation Challenge (upcoming)**: given a set of N policies, $\pi_1, \pi_2, ... \pi_N$, where each policy $\pi_i(a_t|z_t)$ predicts action tokens from image tokens, can you evaluate all of the policies inside a "world model" $p(z_{t+1}|z_t, a_t)$ and tell us the ranked order of which policy is the best?

These challenges are largely inspired by the [commavq compression challenge](https://github.com/commaai/commavq).

## Getting Started

 and extract the dataset to `data/train_v0`, `data/val_v0`.

```
# install dependencies and download data
./build.sh 

# source the Python environment
source venv/bin/activate

# train a baseline model
python train_llm.py --output_dir data/video_llm --model_config 1x-technologies/Llama_1B_v0 --report_to wandb

# generate frames from the baseline
./generate.py --output_dir data/generated

# evaluate the baseline model
./evaluate.py --checkpoint_dir data/video_llm

# visualize the generated results
./visualize.py --token_dir data/generated 
```

## Training GENIE

This repo provides an implementation of the spatio-temporal transformer and MaskGIT sampler as described in [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391). Note that this implemention only trains on video sequences, not actions (though it is trivial to add this via an additive embedding). To train this baseline, 

```
source venv/bin/activate
pip install -r baselines/requirements.txt

# Train the GENIE model
python train_st_model.py --root_dir data/genie_model

# Generate frames from trained model
python genie/generate_genie.py --checkpoint <PATH_TO_CKPT?>

# visualize generated frames
./visualize.py --token_dir data/genie_generated --stride 1

# Evaluate
python genie/evaluate_genie.py --checkpoint <PATH_TO_CKPT?>

```

## Data (Version: 0.0.1)
[Download the dataset on Huggingface](https://huggingface.co/datasets/1x-technologies/worldmodel)
The full dataset is stored in the `data/train_v0` directory:

- **video.bin** - 20x20 image patches at 30hz, each patch is vector-quantized into 1000 possible integer values. These can be decoded into 160x160 RGB images.
- **actions.bin** - encoded robot whole body actions for each image frame, vector-quantized into 1000 possible integer values. You may want to use this to train action-conditioned world models. Raw actions (joint angles, driving velocities, gripper closures, neck pitch) are provided in the `actions/` subdirectory. 
- **segment_ids.bin** - for each frame `segment_ids[i]` uniquely points to the log index that frame `i` came from. You may want to use this to separate non-contiguous frames (transitions).

We also provide a small `val_v0` data split containing held-out examples not seen in the training set, in case you want to try evaluating your model on held-out frames.

## Participating in the Challenges: 

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
./evaluate.py --val_data_dir <PATH-TO-HELD-OUT-DATA>.bin # runs your model on held-out data
```

## Leaderboard

All scores are evaluated on our held-out dataset.

| **User**                                                           | **Teacher-Forced CE Loss** | **Teacher-Forced Token Accuracy** | **Autoregressive CE Loss** | **Autoregressive Token Accuracy** | **Autoregressive LPIPS** | **Generation Time\* (secs/frame)** |
|--------------------------------------------------------------------|----------------------------|-----------------------------------|----------------------------|-----------------------------------|--------------------------|------------------------------------|
| 1x-technologies/GENIE_210M (`maskgit_steps=1`, `temperature=1e-8`) | N/A                        | N/A                               | 3.17                       | 0.319                             | 0.20                     | 0.085                              |
| 1x-technologies/Llama_1B_v0                                        | 2.45                       | 0.399                             | 5.04                       | 0.269                             | 0.23                     | 2.22                               |

*Note that generation time is the time to generate latents on a RTX 4090 GPU, and excludes the time to decode latents to images.

### Metric Details
We evaluate the model under two different scenarios; in both cases, the model receives the tokens of the previous frame(s) as input, 
and the model should predict the tokens of the following frame. 
- **Autoregressive** (frame-level) is closer to an actual generation scenario, where the model receives $t$ x 20x20 tokens representing frames 0 to $t - 1$, 
and the model should auto-regressively predict all 20x20 for frame $t$.
- (If applicable), **Teacher-forced** matches the typical training scenario with causal masking. 
It is simply a next token prediction task where all previous tokens, including any in the current frame, are ground-truth tokens as opposed to autoregressively predicted tokens.


## Citation

If you use this software or dataset in your work, please cite it as follows:

```
@misc{WorldModelChallenge1X2024,
  author = {1X},
  title = {1X World Model Challenge},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/1x-technologies/1xgpt}},
  commit = {22843fe3c43e92576c967c932e1197cd035a63d0}
}
```

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