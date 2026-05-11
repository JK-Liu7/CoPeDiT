import timeit
from generative.networks.schedulers.ddim import DDIMPredictionType
from torch.nn.parallel import DistributedDataParallel
from monai.losses.perceptual import PerceptualLoss
from monai.utils import first
from AutoEncoder.model.CoPeVAE_CardiacMRI import CopeVAE
from LDM.model.MDiT3D_Cardiac import *
from LDM.dataset.CardiacMRI_data import *
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import mean_absolute_error as mae
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import nibabel as nib
from LDM.inferers import inferer_DiT_Cardiac
from LDM.utils import *
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from einops import rearrange



def gather_first_dim_variable(x: torch.Tensor, gather_to_rank0: bool = True):
    assert dist.is_initialized()
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = x.device

    b_local = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    b_list = [torch.zeros_like(b_local) for _ in range(world)]
    dist.all_gather(b_list, b_local)
    b_list = [int(v.item()) for v in b_list]
    b_max = max(b_list)

    if x.shape[0] < b_max:
        pad = torch.zeros((b_max - x.shape[0],) + x.shape[1:], device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x
    out_list = [torch.empty_like(x_pad) for _ in range(world)]
    dist.all_gather(out_list, x_pad)
    out_list = [t[:b_list[i]] for i, t in enumerate(out_list)]
    x_all = torch.cat(out_list, dim=0)

    if gather_to_rank0:
        return x_all if rank == 0 else None
    return x_all


def inference(args, autoencoder, DiT):
    torch.set_grad_enabled(False)

    if args.rank == 0:
        logger = create_logger(args.log_dir, args.distributed)
    else:
        logger = create_logger(args.log_dir, args.distributed)

    datasets = get_cardiacMRI(args)
    train_files_combined, test_files_combined = get_data(datasets, args.dataset)
    train_transform, test_transform = get_transforms(args)
    dataloader_train, dataloader_test, train_sampler, test_sampler = get_loader(args, args.rank, args.world_size, train_files_combined, test_files_combined, train_transform, test_transform)

    DiT = DiT.to(args.device)
    autoencoder = autoencoder.to(args.device)

    # load autoencoder checkpoint
    autoencoder = autoencoder.to(args.device)
    ae_state_dict = args.ae_dir + 'vqvae/Autoencoder.pt'

    dit_state_dict = args.model_dir + 'DiT.pt'

    if not ae_state_dict is None:
        state_dict = torch.load(str(ae_state_dict))
        autoencoder.load_state_dict(state_dict["model"])
        if args.rank == 0:
            print('AE weighted loaded!')

    if not dit_state_dict is None:
        state_dict = torch.load(str(dit_state_dict))
        DiT.load_state_dict(state_dict["ema"])
        start_epoch = state_dict["epoch"]
        if args.rank == 0:
            print('DiT weighted loaded!')
            print(f"Inference from checkpoint, epoch: {start_epoch}")
        logger.info(f"Inference from checkpoint, epoch: {start_epoch}")

    if args.distributed:
        DiT = DistributedDataParallel(DiT, device_ids=[args.device], output_device=args.device)

    DiT.eval()
    autoencoder.eval()
    DiT.requires_grad_(False)
    autoencoder.requires_grad_(False)

    check_data = first(dataloader_train)
    with torch.no_grad():
        with autocast(enabled=True):
            z0 = autoencoder.encode_stage_2_inputs(check_data[0].to(args.device))
            z1 = autoencoder.encode_stage_2_inputs(check_data[1].to(args.device))
    scale_factor = 1.0
    print(f"scale_factor -> {scale_factor}.")
    torch.cuda.empty_cache()

    if args.noise_scheduler == 'linear':
        if args.pred_type == 'noise':
            scheduler_ddim = DDIMScheduler(num_train_timesteps=args.sample_steps, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195, clip_sample=False)
        elif args.pred_type == 'x':
            scheduler_ddim = DDIMScheduler(num_train_timesteps=args.sample_steps, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195,
                                           clip_sample=False, prediction_type=DDIMPredictionType.SAMPLE)
        elif args.pred_type == 'v':
            scheduler_ddim = DDIMScheduler(num_train_timesteps=args.sample_steps, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195,
                                           clip_sample=False, prediction_type=DDIMPredictionType.V_PREDICTION)
    elif args.noise_scheduler == 'cosine':
        if args.pred_type == 'noise':
            scheduler_ddim = DDIMScheduler(num_train_timesteps=args.sample_steps, schedule="cosine", clip_sample=True, clip_sample_min=0, clip_sample_max=1)
        elif args.pred_type == 'x':
            scheduler_ddim = DDIMScheduler(num_train_timesteps=args.sample_steps, schedule="cosine", clip_sample=True, clip_sample_min=0, clip_sample_max=1,
                                           prediction_type=DDIMPredictionType.SAMPLE)
        elif args.pred_type == 'v':
            scheduler_ddim = DDIMScheduler(num_train_timesteps=args.sample_steps, schedule="cosine", clip_sample=True, clip_sample_min=0, clip_sample_max=1,
                                           prediction_type=DDIMPredictionType.V_PREDICTION)
    scheduler_ddim.set_timesteps(num_inference_steps=args.sample_steps)

    inferer = inferer_DiT_Cardiac.LatentDiffusionInferer(scheduler_ddim, scale_factor=scale_factor)

    loss_perceptual = (
        PerceptualLoss(spatial_dims=2, network_type="squeeze").eval().to(args.device)
    )

    metric_epoch = {"psnr": 0, "ssim": 0, "mae": 0, "lpips": 0}

    start = timeit.default_timer()
    total = 0
    n = args.batch_size
    global_batch_size = args.batch_size * args.world_size
    total_batches = len(dataloader_test)
    for i, batch in enumerate(tqdm(dataloader_test, total=total_batches, desc="Processing")):
        with torch.no_grad():
            with autocast(enabled=args.amp):
                x_available, x_missing, missing_condition = batch
                x_available, x_missing, missing_condition = x_available.to(args.device), x_missing.to(args.device), missing_condition.to(args.device)

                noise = torch.randn_like(z1).to(args.device)
                prompts = autoencoder.get_condition(x_available)

                output_images = inferer.sample(
                    input_a=x_available, input_noise=noise,
                    autoencoder_model=autoencoder, diffusion_model=DiT,
                    scheduler=scheduler_ddim,
                    conditioning=prompts,
                    verbose=False
                )

                syn_images = output_images

                del x_available, missing_condition
                torch.cuda.empty_cache()

                # Save images
                processed_missing = rearrange(x_missing, 'b c h w d -> (b d) c h w').contiguous()
                processed_syn = rearrange(syn_images, 'b c h w d -> (b d) c h w').contiguous()

                if args.distributed:
                    rank = dist.get_rank()
                    index = i * args.world_size + rank + total
                    samples = syn_images.squeeze().detach().cpu().numpy()
                    GTs = x_missing.squeeze().detach().cpu().numpy()

                    for i in range(n):
                        out = np.float32(samples[i])
                        out_img = nib.Nifti1Image(out, np.eye(4))
                        out_filename = args.save_dir + str(index) + ".nii.gz"
                        nib.save(out_img, out_filename)

                        out_gt = np.float32(GTs[i])
                        out_img_gt = nib.Nifti1Image(out_gt, np.eye(4))
                        out_filename_gt = args.save_dir + str(index) + "_GT.nii.gz"
                        nib.save(out_img_gt, out_filename_gt)

                else:
                    samples = syn_images.squeeze().detach().cpu().numpy()
                    GTs = x_missing.squeeze().detach().cpu().numpy()

                    for i in range(n):
                        out = np.float32(samples[i])
                        out_img = nib.Nifti1Image(out, np.eye(4))
                        out_filename = args.save_dir + str(i) + ".nii.gz"
                        nib.save(out_img, out_filename)
                        out_gt = np.float32(GTs[i])
                        out_img_gt = nib.Nifti1Image(out_gt, np.eye(4))
                        out_filename_gt = args.save_dir + str(i) + "_GT.nii.gz"
                        nib.save(out_img_gt, out_filename_gt)

                metrics = {
                    "psnr": psnr(syn_images, x_missing),
                    "ssim": ssim(processed_syn, processed_missing),
                    "mae": mae(syn_images, x_missing),
                    "lpips": loss_perceptual(processed_syn, processed_missing)
                }

                del syn_images

            for metric_name, metric_value in metrics.items():
                metric_epoch[metric_name] += metric_value
            total += global_batch_size

    for key in metric_epoch:
        metric_epoch[key] /= len(dataloader_test)
    end = timeit.default_timer()
    time = end - start

    if args.rank == 0:
        print(
            "[[Time: %d][PSNR: %.3f][SSIM: %.3f][MAE: %.3f][LPIPS: %.3f]"
            % (time, metric_epoch['psnr'], metric_epoch['ssim'],
               metric_epoch['mae'], metric_epoch['lpips'])
        )
        logger.info(
            "[[Time: %d][PSNR: %.3f][SSIM: %.3f][MAE: %.3f][LPIPS: %.3f]"
            % (time, metric_epoch['psnr'], metric_epoch['ssim'],
               metric_epoch['mae'], metric_epoch['lpips'])
        )

    if args.distributed:
        cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='UKBB', type=str)
    parser.add_argument("--missing_num", default=8, type=int, help="number of missing slices", choices=[4, 8, 16, 24])
    parser.add_argument("--epochs", default=5000, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=3000, type=int, help="number of training iterations")
    parser.add_argument("--warmup_steps", default=50, type=int, help="warmup steps")
    parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--decay", default=0, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--val_interval", default=50, type=int)
    parser.add_argument("--ckpt_interval", default=200, type=int)

    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--cache', default=0.0, type=float)
    parser.add_argument('--distributed', default=True, action='store_true', help='distributed training')
    parser.add_argument("--gpu_ids", default=[0, 1, 2, 3], help="local rank")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    # MDiT3D Configuration
    parser.add_argument("--DiT", default='MDiT3D-B/2', type=str)
    parser.add_argument("--extras", default=0, type=int, help="whether to use learned prompts or conditions")
    parser.add_argument("--noise_scheduler", default='linear', type=str)
    parser.add_argument("--pred_type", default='x', type=str, choices=['noise', 'x', 'v'])
    parser.add_argument('--diffusion_steps', default=500, type=int)
    parser.add_argument('--sample_steps', default=250, type=int)
    parser.add_argument("--diff_loss", default='l2', type=str)

    # CoPeVAE Configuration
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--vae_channel", default=(32, 64, 128), type=Sequence[int])
    parser.add_argument("--res_channel", default=256, type=Sequence[int])
    parser.add_argument('--code_num', default=8192, type=int)
    parser.add_argument('--code_dim', default=8, type=int)
    parser.add_argument("--recon_loss", default='l1', type=str)
    parser.add_argument("--cardiac_pad", default=(192, 192, 16), type=Sequence[int])
    parser.add_argument("--cardiac_roi", default=(160, 160, 16), type=Sequence[int])
    parser.add_argument("--cardiac_size", default=(192, 192, 32), type=Sequence[int])
    parser.add_argument('--length', default=32, type=int)
    parser.add_argument('--latent_dim', default=8, type=int)
    parser.add_argument('--proj_dim', default=512, type=int)
    parser.add_argument('--contrast_dim', default=128, type=int)
    parser.add_argument('--lambda1', default=1.0, type=float)
    parser.add_argument('--lambda2', default=0.1, type=float)
    parser.add_argument('--lambda3', default=1.0, type=float)
    parser.add_argument('--vq_weight', default=1.0, type=float)
    parser.add_argument('--per_weight', default=0.01, type=float)
    parser.add_argument('--adv_weight', default=0.01, type=float)
    parser.add_argument('--pretext_weight', default=0.1, type=float)

    args = parser.parse_args()

    args.result_dir = '../result/MDiT3D/'
    os.makedirs(args.result_dir, exist_ok=True)
    args.log_dir = args.result_dir + '/log/Cardiac/' + str(args.missing_num) + '/'
    os.makedirs(args.log_dir, exist_ok=True)

    args.ae_dir = 'model_save/CoPeVAE/Cardiac/'
    args.model_dir = 'model_save/MDiT3D/Cardiac/' + str(args.missing_num) + '/'
    os.makedirs(args.model_dir, exist_ok=True)

    args.save_dir = 'img_save/Cardiac/' + str(args.missing_num) + '/'
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + '/Synthetic/', exist_ok=True)
    os.makedirs(args.save_dir + '/GT/', exist_ok=True)

    args.amp = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    args.distributed = False

    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.world_size = 1
    args.rank = 0

    if args.distributed:
        dist.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.device = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.device)
        num_gpus = torch.cuda.device_count()
        print(f"Setting up distributed training with {num_gpus} GPUs available")
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        args.device = torch.device("cuda:0")
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    autoencoder = CopeVAE(args)
    DiT = MDiT3D_models[args.DiT]()

    inference(args, autoencoder, DiT)