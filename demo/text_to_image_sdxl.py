# region - imports
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["GRADIO_TEMP_DIR"] = "/home/wanghesong/gradio_tmp"

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler, AutoencoderTiny
from main.sdxl.sdxl_text_encoder import SDXLTextEncoder
from main.utils import get_x0_from_noise
from transformers import AutoTokenizer
from accelerate import Accelerator
import gradio as gr    
import numpy as np
import argparse 
import torch
import time 
import PIL
from ipdb import set_trace as st
    
SAFETY_CHECKER = False
SDXL_CKPT_PATH = "/home/wanghesong/.cache/huggingface/hub/models--tianweiy--DMD2/snapshots/be22767697a1f3ca656b73c776e15fa335c86c6c/dmd2_sdxl_1step_unet_fp16.bin"
# endregion - imports

class ModelWrapper:
    def __init__(self, args, accelerator):
        super().__init__()
        # disable all gradient calculations
        torch.set_grad_enabled(False)  #* 禁用torch的梯度运算, 减少内存消耗和加速推理过程
        
        #* 设置数据精度
        if args.precision == "bfloat16":
            self.DTYPE = torch.bfloat16
        elif args.precision == "float16":
            self.DTYPE = torch.float16
        else:
            self.DTYPE = torch.float32
        self.device = accelerator.device

        #* AutoTokenizer: 用于自动加载与特定模型兼容的Tokenizer
        #* 有两个: 为了适应自定义的text encoder
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )

        #* 初始化自定义的SDXLTextEncoder
        self.text_encoder = SDXLTextEncoder(args, accelerator, dtype=self.DTYPE)

        # vanilla SDXL VAE needs to be kept in float32
        #* 加载预训练的vae模型
        #* AutoencoderKL.from_pretrained(): 加载一个预训练好的VAE, 通常从 Hugging Face Hub 或本地路径中加载
        self.vae = AutoencoderKL.from_pretrained(
            args.model_id, 
            subfolder="vae"
        ).float().to(self.device)
        self.vae_dtype = torch.float32

        #* 加载预训练的tiny vae
        self.tiny_vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", 
            torch_dtype=self.DTYPE
        ).to(self.device) 
        self.tiny_vae_dtype = self.DTYPE

        # Initialize Generator  #* 一个使用指定ckpt的UNet2DConditionModel
        self.model = self.create_generator(args).to(dtype=self.DTYPE).to(self.device)

        self.accelerator = accelerator
        self.image_resolution = args.image_resolution
        self.latent_resolution = args.latent_resolution
        self.num_train_timesteps = args.num_train_timesteps
        self.vae_downsample_ratio = self.image_resolution // self.latent_resolution  #* iamge -> latent

        self.conditioning_timestep = args.conditioning_timestep 

        self.scheduler = DDIMScheduler.from_pretrained(
            args.model_id,
            subfolder="scheduler"
        )
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # sampling parameters 
        self.num_step = args.num_step 
        self.conditioning_timestep = args.conditioning_timestep 

        # safety checker  #*暂且不管
        if SAFETY_CHECKER:
            # adopted from https://huggingface.co/spaces/ByteDance/SDXL-Lightning/raw/main/app.py
            from demo.safety_checker import StableDiffusionSafetyChecker
            from transformers import CLIPFeatureExtractor

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(device=self.device, dtype=self.DTYPE)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32", 
            )
    
    def check_nsfw_images(self, images):
        ''' 只有在SAFETY_CHECKER = True的时候才会调用, 先不管'''
        safety_checker_input = self.feature_extractor(images, return_tensors="pt") # .to(self.dviece)
        has_nsfw_concepts = self.safety_checker(
            clip_input=safety_checker_input.pixel_values.to(device=self.device, dtype=self.DTYPE),
            images=images
        )
        return has_nsfw_concepts

    def create_generator(self, args):
        ''' 加载一个UNet2DConditionModel, 并使用指定的ckpt'''
        generator = UNet2DConditionModel.from_pretrained(
            args.model_id,
            subfolder="unet"
        ).to(self.DTYPE)

        #* 加载ckpt, map_location="cpu": 确保权重首先加载到cpu; 避免内存不足问题, especially模型较大时
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")  #* 模型权重(的字典?)
        #* 将'state_dict'加载到'generator'中, strict=True：要求模型的架构和检查点中的权重必须完全匹配
        print(generator.load_state_dict(state_dict, strict=True))
        generator.requires_grad_(False)  #* 禁用梯度计算 \ 后下划线: 原地操作

        # st()
        # pruned_generator = UNet2DConditionModel.from_config("./demo/sdxl_config.json")
        # copy_weight_from_teacher(pruned_generator, generator, "bk_base")
        # pruned_generator.requires_grad_(False)  #* 禁用梯度计算 \ 后下划线: 原地操作

        return generator 
        # return pruned_generator
 
    def build_condition_input(self, height, width):
        ''' 应该是sdxl中用于尺寸控制的代码
        #? 暂时不清楚这个函数的作用是什么
        #? 构建条件输入, 为模型生成图像时提供关于图像尺寸和裁剪位置的额外信息(time_ids), 然后将这些信息转换为一个 PyTorch 张量, 用于后续模型生成过程中的条件控制'''
        original_size = (height, width)  #* default: [1024, 1024]
        target_size = (height, width)  #* 同上
        crop_top_left = (0, 0)  #* 从左上角开始使用整个图像

        add_time_ids = list(original_size + crop_top_left + target_size)  #* [1024, 1024, 0, 0, 1024, 1024]
        add_time_ids = torch.tensor([add_time_ids], device=self.device, dtype=self.DTYPE)  #* 类型转换为torch.tensor
        return add_time_ids
  
    def _encode_prompt(self, prompt):
        ''' 将prompt转换为模型可理解的向量, 也就是token ID序列'''
        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,  #* 77
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_two = self.tokenizer_two(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,  #* 77
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_dict = {
            'text_input_ids_one': text_input_ids_one.unsqueeze(0).to(self.device),
            'text_input_ids_two': text_input_ids_two.unsqueeze(0).to(self.device)
        }
        return prompt_dict  #* 返回一个字典

    @staticmethod  
    def _get_time():
        ''' 返回系统时间 \ 静态方法, 与类的实例无关'''
        torch.cuda.synchronize()  #* 同步CPU和GPU
        return time.time()  #* 返回当前的系统时间

    def sample(self, noise, unet_added_conditions, prompt_embed, fast_vae_decode):
        ''' 根据现有信息生成图像, noise -> image'''
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        if self.num_step == 1:
            all_timesteps = [self.conditioning_timestep]
            step_interval = 0 
        elif self.num_step == 4:
            all_timesteps = [999, 749, 499, 249]
            step_interval = 250 
        else:
            raise NotImplementedError()
        
        DTYPE = prompt_embed.dtype  #* [bsz, 77, 2048]
        
        for constant in all_timesteps:
            current_timesteps = torch.ones(len(prompt_embed), device=self.device, dtype=torch.long)  *constant  #* [bsz个current_timesteps]
            eval_images = self.model(  #* 一个unet \ unet的输出是类似字典的东西, 只有一个key, 是'sample'
                noise, current_timesteps, prompt_embed, added_cond_kwargs=unet_added_conditions
            ).sample  #* [bsz, 4, 128, 128]

            eval_images = get_x0_from_noise(  #* 通过<noise>和<pred_noise>(模型输出), 计算出clean image
                noise, eval_images, alphas_cumprod, current_timesteps
            ).to(self.DTYPE)

            #* refer to 原论文的 multi step generate, 先不管了
            next_timestep = current_timesteps - step_interval 
            noise = self.scheduler.add_noise(
                eval_images, torch.randn_like(eval_images), next_timestep
            ).to(DTYPE)  

        #* 把latent decode为image, latent -> image
        if fast_vae_decode:
            eval_images = self.tiny_vae.decode(eval_images.to(self.tiny_vae_dtype) / self.tiny_vae.config.scaling_factor, return_dict=False)[0]
        else:
            eval_images = self.vae.decode(eval_images.to(self.vae_dtype) / self.vae.config.scaling_factor, return_dict=False)[0]
        #* standardization, 把图像从 [-1, 1] 映射到 [0, 255]
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return eval_images 

    @torch.no_grad()  
    def inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_images: int,
        fast_vae_decode: bool
    ):
        ''' 推理过程, 生成图片'''
        print("Running model inference...")

        if seed == -1:
            seed = np.random.randint(0, 1000000)

        generator = torch.manual_seed(seed)  #* 设置随机种子生成器

        add_time_ids = self.build_condition_input(height, width).repeat(num_images, 1)  #* [bsz, 6]

        #* 生成latent的随机噪声
        noise = torch.randn(  #* [bsz, 4, 128, 128]: 128是latent的分辨率, 常规的是96
            num_images, 4, height // self.vae_downsample_ratio, width // self.vae_downsample_ratio, 
            generator=generator
        ).to(device=self.device, dtype=self.DTYPE) 

        #* 将prompt转换为模型可理解的向量(token ID)
        prompt_inputs = self._encode_prompt(prompt)  #* [1, 1, 77] \ 字典, 两个key分别是 'text_input_ids_one/two' 
        
        start_time = self._get_time()

        #* 把token ID转换为embedding: 普通的embedding & 池化的embedding(?用于模型的高层次条件信息)
        #*     prompt_embeds: 是拼接后的细粒度嵌入, 包含丰富的语义信息 -> self.sample() -> self.moderl()
        #*     pooled_prompt_embeds: 是整体的语义表示, 用于高层次的条件生成控制 -> unet_added_conditions -> self.model()
        prompt_embeds, pooled_prompt_embeds = self.text_encoder(prompt_inputs)  #* [1, 77, 2048], [1, 1280]

        #* 将嵌入批量化, 为生成的每一张image提供文本嵌入
        batch_prompt_embeds, batch_pooled_prompt_embeds = (  #* [bsz, 77, 2048], [bsz, 1, 1280]
            prompt_embeds.repeat(num_images, 1, 1),
            pooled_prompt_embeds.repeat(num_images, 1, 1)
        )

        #* 用于模型生成的附加条件信息 -> self.model()
        #? 暂时不太清楚有什么用处
        unet_added_conditions = {
            "time_ids": add_time_ids,  #* [bsz, 6]
            "text_embeds": batch_pooled_prompt_embeds.squeeze(1)  #* [bsz, 1280]
        }

        #* 采样, 生成图片
        eval_images = self.sample(  #* [bsz, 1024, 1024, 3], 就是生成的图片
            noise=noise,
            unet_added_conditions=unet_added_conditions,
            prompt_embed=batch_prompt_embeds,
            fast_vae_decode=fast_vae_decode
        )

        end_time = self._get_time()

        output_image_list = [] 
        for image in eval_images:  #* image: [1024, 1024, 3]
            output_image_list.append(PIL.Image.fromarray(image.cpu().numpy()))  #* 将图像转化为可显示格式

        if SAFETY_CHECKER:  #* 先不管
            has_nsfw_concepts = self.check_nsfw_images(output_image_list)
            if any(has_nsfw_concepts):
                return [PIL.Image.new("RGB", (512, 512))], "NSFW concepts detected. Please try a different prompt."

        return (
            output_image_list,
            f"run successfully in {(end_time-start_time):.2f} seconds"
        )


def copy_weight_from_teacher(unet_stu, unet_tea, student_type):
    #todo 返回继承了参数但未经过剪枝的学生模型
    #* 首先定义一个字典，用于存储stu和tea之间的映射关系
    connect_info = {} # connect_info['TO-student'] = 'FROM-teacher'
    if student_type in ["bk_base", "bk_small"]:
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.0.resnets.2.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.3.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.3.attentions.1.'] = 'up_blocks.3.attentions.2.'
    elif student_type in ["bk_tiny"]:
        connect_info['up_blocks.0.resnets.0.'] = 'up_blocks.1.resnets.0.'
        connect_info['up_blocks.0.attentions.0.'] = 'up_blocks.1.attentions.0.'
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.0.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.0.upsamplers.'] = 'up_blocks.1.upsamplers.'
        connect_info['up_blocks.1.resnets.0.'] = 'up_blocks.2.resnets.0.'
        connect_info['up_blocks.1.attentions.0.'] = 'up_blocks.2.attentions.0.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.1.upsamplers.'] = 'up_blocks.2.upsamplers.'
        connect_info['up_blocks.2.resnets.0.'] = 'up_blocks.3.resnets.0.'
        connect_info['up_blocks.2.attentions.0.'] = 'up_blocks.3.attentions.0.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.3.attentions.2.'       
    else:
        raise NotImplementedError


    for k in unet_stu.state_dict().keys(): #* 变量模型所有参数（的key）, .state_dict(): 返回模型的状态字典，包含了模型所有参数的名称和值
        flag = 0 #* 标志变量，用于指示当前键是否需要强制复制
        k_orig = k #* 初始的键名，用于在需要时替换前缀
        for prefix_key in connect_info.keys(): #* 遍历需要被复制的参数
            if k.startswith(prefix_key): #* 判断当前参数是否需要继承teacher model
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key]) #* 替换前缀, str.replace(old, new)         
                break

        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
        else:
            print(f"normal COPY {k_orig} -> {k}")
        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k_orig]) #* copy_(): 将一个张量的值复制到另一个张量中

    return unet_stu


def create_demo():
    ''' 创建一个基于gradio的前端页面'''
    TITLE = "# DMD2-SDXL Demo"  #* 自定义的markdown标题
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_resolution", type=int, default=128)  #* latent sapce的分辨率
    parser.add_argument("--image_resolution", type=int, default=1024)  #* 生成图像的分辨率
    parser.add_argument("--num_train_timesteps", type=int, default=1000)  #* 训练的时间步
    parser.add_argument("--checkpoint_path", type=str, default=SDXL_CKPT_PATH)
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--conditioning_timestep", type=int, default=999)  #* 在哪个时间步上使用文本进行conditioning
    parser.add_argument("--num_step", type=int, default=4, choices=[1, 4])  #* 生成图像用的步数
    parser.add_argument("--revision", type=str)
    args = parser.parse_args()

    #* 启用了TensorFloat-32优化, 再nvidia ampere gpu上加速矩阵运算
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 

    accelerator = Accelerator()

    model = ModelWrapper(args, accelerator)  #* ModelWrapper封装了模型的所有逻辑

    with gr.Blocks() as demo:  #* gr.Blocks(): gradio中的高级布局组件, 允许我们创建一个多组件布局
        gr.Markdown(TITLE)
        with gr.Row():  #* 创建一个行, 行里面有两列
            with gr.Column():  #* 第一列
                prompt = gr.Text(  #* 创建一个文本输入框
                    #* 默认值
                    value="An oil painting of two rabbits in the style of American Gothic, wearing the same clothes as in the original.",
                    label="Prompt",  #? 标签, 暂时不知道是什么东西
                    #* 提示信息, 文本框为空时显示
                    placeholder='e.g. An oil painting of two rabbits in the style of American Gothic, wearing the same clothes as in the original.'
                )
                run_button = gr.Button("Run")  #* 创建 一个按钮
                with gr.Accordion(label="Advanced options", open=True):  #* 创建一个课折叠的高级面板, 用于设置更高级的参数
                    seed = gr.Slider(  #* 创建滑块, 用于选择范围
                        label="Seed",
                        minimum=-1,
                        maximum=1000000,
                        step=1,
                        value=0,
                        info="If set to -1, a different seed will be used each time.",
                    )
                    num_images = gr.Slider(
                        label="Number of generated images",
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=16,
                        info="Use smaller number if you get out of memory error."
                    )
                    fast_vae_decode = gr.Checkbox(  #* 创建复选框
                        label="Use Tiny VAE for faster decoding",
                        value=True
                    )
                    height = gr.Slider(
                        label="Image Height",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024,
                        info="Image height in pixels. Set to 1024 for the best result"
                    )
                    width = gr.Slider(
                        label="Image Width",
                        minimum=512,
                        maximum=1536,
                        step=64,
                        value=1024,
                        info="Image width in pixels. Set to 1024 for the best result"
                    )
            with gr.Column():  #* 第二列
                #* 创建一个图片展示控件, 用于显示生成的图像
                result = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery", height=1024)
                error_message = gr.Text(label="Job Status")  #* 文本框控件, 显示运行过程中的状态信息

        inputs = [  #* 包含了所有用户输入的控件对象, 会作为输入传给模型
            prompt,
            seed,
            height,
            width,
            num_images,
            fast_vae_decode
        ]
        run_button.click(  #* 这个方法绑定了点击按钮后的行为 \ concurrency_limit=1: 限制每次只有一个任务运行
            fn=model.inference, inputs=inputs, outputs=[result, error_message], concurrency_limit=1
        )
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue(api_open=False)  #* 将推理过程放入任务队列中，确保多个请求可以排队处理
    demo.launch(  #* 启动 Gradio 应用，并通过 share=True 生成一个公开链接，用户可以通过该链接访问该应用
        show_error=True,
        share=True
    )
