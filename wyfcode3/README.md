# RingID latent v4

独立项目：在 Stable Diffusion 2.1 的去噪 latent 上嵌入低频傅里叶环，检测只用密钥、图像和原始 VAE。旋转检测以 10° 为步长搜索全周。

## 流程

1. 密钥文件 `keys/watermark.key`。环模板的随机数是 `SHA-256(密钥 + 标签)` 播种的 NumPy Generator。初始噪声 `z0` 是 `torch.Generator(seed)` 的标准正态。不需要 Rust。
2. `z0` 保持干净高斯，DDIM 去噪。
3. 解码前把密钥环写进 latent 第 4 个通道（保留宿主相位，压低中心亮斑，blend 0.85）。
4. VAE 解码成图。
5. 检测：VAE 编码，在 0° 和每隔 10° 的反旋图上做环相关，取最大峰值。不做 DDIM 反演。

## 第一次运行

在已安装依赖的环境里（本仓库的 `watermark` conda 环境即可）：

```powershell
cd ringid_v4_project
python download_weights.py
python eval_quick.py
```

`download_weights.py` 把 `Manojb/stable-diffusion-2-1-base` 下载到 `models/`。`generate.py` 和 `detect.py` 在权重不存在时也会自动下载。

`eval_quick.py` 会在没有权重时先下载，然后生成同一 seed 的无水印图、水印图和一张随机旋转图，保存三联对比图。不计算 PSNR / SSIM。

没写过阈值时，检测用一组无关高斯 latent 快速估一个阈值，并提示你再跑 `python calibrate.py`（干净 SD 负样本，和正式评测一致）。原先 100 张实验在旧密钥上的阈值是 0.557，换密钥后要重新校准。

## 命令

```powershell
python generate.py --prompt "a photo of a cat" --seed 8 --steps 15
python detect.py --image output/wm_seed8.png --threshold 0.557
```

把已有 PEM 或密钥原文放到 `keys/watermark.key`，环模板由该文件全文决定。
