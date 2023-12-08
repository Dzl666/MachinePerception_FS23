# Machine Perception FS23 - MPGroup

## How to reproduce

> * `module load gcc/8.2.0 cuda/11.6.2 python/3.8.5 cudnn/8.0.5 cmake/3.19.8 eth_proxy`
> * `source ../env-machine-perception/bin/activate`

> Training `python train.py --data-root data/[public|private]`

> Use both training and validation datasets

> Testing public `python test.py --data-root data/public --pretrained-root ./log/results/test_model/public --model-name model-5999.pth`

> Testing private `python test.py --data-root data/private --pretrained-root ./log/results/test_model/private --model-name model-5999.pth`

The testing results will be stored in the folder **./log/results/folder/[public|private]**.

## SKELETON CODE AND DATA

The data loader returns a dictionary with two keys:

- **rays: a tensor of shape (B, N_ray, 6)** containing the camera origins (the first three elements) and ray directions (the last three elements).
- **imgs: a tensor of shape (B, N_ray, 3)** containing the RGB colors of the rays, normalized to the range [0, 1].

## Environment

- set_software_stack.sh new if **ERROR:105: Unable to locate a modulefile for 'gcc/8.2.0'**
- python -m venv env-machine-perception
- python -m pip install --upgrade pip
- pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
- cd project2
- pip install -r requirements.txt

## RUNNING IN INTERACTIVE MODE

To run and test your code interactively (for a maximum of 4 hours), you can use the srun to acquire a GPU. The following command will allocate 4 CPUs and 16GB memory with a GTX-1080Ti GPU:

> `srun -n 4 --mem-per-cpu=4000 --gpus=gtx_1080_ti:1 --pty bash`

Once you have acquired the GPU, you can run the training script with the following command:

> `python train.py --save-root /cluster/scratch/your_username --data-root data/[public|private] --wandb`

In this example, the experimental results will be stored on a scratch space located under **/cluster/scratch/your_username**. Please note that this is temporary storage and files older than 2 weeks will be automatically purged.

You can also store experimental results in your `$HOME` directory, but disk space is much more limited there. You can check your quota with the command `lquota`.

We use Weights & Biases to track our experiments. You can activate it by passing the --wandb flag.

## JOB SUBMISSION

To submit a GPU job, please use the following command (ensure that your virtual environment is activated before running the command).

> `sbatch -n 4 --time=24:00:00 --mem-per-cpu=4000 --gpus=1 --gres=gpumem:11G --output="training_log" --wrap="python train.py --save-root /cluster/scratch/your_username --data-root data/[public|private] --wandb"`

* `-n 4` Request 4 CPU cores.
* `--time=24:00:00` Wall time of 24 hours. After running for this amount of time, your job will be automatically cancelled.
This also determines the queue in which your job will land.
* `--mem-per-cpu=4000` Request 4 GB of memory per CPU core.
* `--gpus=1` Request 1 GPU (Note: only 1 GPU is allowed to be used at a time per group).
* `--gres=gpumem:11G` Request a GPU with 11 GB of GPU memory.
* `--output="training_log"` Output is written to this file in the current directory.
* `--wrap="......"` The command to be executed.

Once you have submitted the job, you can check its status using the command `squeue`. When the job is waiting in the queue, its status (ST) will be ***PD***, and once it starts running, the status will change to ***R***.

Each job is also assigned with a unique job id. To check details of the job, use `myjobs -j job_id` (e.g. how much memory your job used). To monitor what the job has written to standard output, use the command `vim training_log`.

## References

1. Mildenhall et al. (2020) [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://github.com/bmild/nerf)
2. Peng et al. (2021) [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction](https://github.com/Totoro97/NeuS)
3. Yariv et al. (2021) [Volume Rendering of Neural Implicit Surfaces]()
4. Shao etal. (2021) [DoubleField: Bridging the Neural Surface and Radiance Fields for High-fidelity Human Reconstruction and Rendering]()