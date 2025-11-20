# ultralytics_pro - Model Package

[繁體中文](README.md) | English

## User Guide

Supported Version
- Recommended: `ultralytics==8.3.225`
- Install additional dependencies:
  ```powershell
  pip install -r requirements.txt
  ```

This project provides organized and extended ultralytics model configurations and examples, making it convenient for local development or for replacing the ultralytics package configuration in site-packages ("C:\Users\USER\AppData\Local\Programs\Python\Python310\Lib\site-packages\ultralytics").

> [!IMPORTANT]
> Important Note:
> - Please be sure to back up the original folder before replacing system packages, and confirm Python version and dependency compatibility.

> [!TIP]
> ## Usage with YOLO_tools
> You can use the [YOLO_tools](https://github.com/Chriz122/YOLO_tools) toolbox for training, annotation processing, evaluation, and more.

## Model Introduction
### YOLOv3 Series
| Model Name                                                                  |                                                Improved Module / Architecture (Summary) | Improvements over Original YOLO                                  | Strengths & Application Scenarios               |
| --------------------------------------------------------------------- | -----------------------------------------------------------: | ---------------------------------------------- | --------------------- |
| `yolov3.yaml`           |                 Standard YOLOv3  | Standard YOLOv3                                        | General object detection                |
| `yolov3-spp.yaml`           |              Add SPP (Spatial Pyramid Pooling) layer to backbone/neck | Larger receptive field, improved multi-scale feature fusion and small object performance                          | Complex scenes with multi-scale, small object detection      |
| `yolov3-tiny.yaml`           |                                  Simplified backbone and head (tiny structure) | Faster inference, fewer parameters, lower accuracy                                  | Embedded/real-time inference with limited resources        |
| `yolov3-rtdetr.yaml`           | RT-DETR head                                        | Detection head replaced with `RTDETRDecoder` for more stable localization and classification     | Real-time performance with high accuracy           |
| `yolov3-spp-rtdetr.yaml`           |                                        Both SPP and RT-DETR head | Multi-scale fusion + improved decoding, helps with Small objects and boundary accuracy                      | Small objects + high localization demand           |
| `yolov3-tiny-rtdetr.yaml`           |                                        tiny + RT-DETR head | Attempts to improve localization/classification quality in extremely lightweight models                             | Ultra-low resource scenarios needing better accuracy         |

### YOLOv4 Series
| Model Name                                                                        |                                 Improved Module / Architecture (Summary) | Improvements over Original YOLO                | Strengths & Application Scenarios             |
| --------------------------------------------------------------------------- | --------------------------------------------: | ---------------------------- | ------------------- |
| `yolov4-p5.yaml` / `yolov4-p6.yaml` / `yolov4-p7.yaml`                      |                   Adjust output levels (P5/P6/P7 correspond to different pyramid layers) | Supports larger/smaller scale detection needs (P7 is better for large scale)    | General object detection, specialized for small/medium/large objects |
| `yolov4-csp-rtdetr.yaml`           |                             CSP + RT-DETR head | Combines CSP efficiency with RT-DETR-like decoding    | Real-time performance with high accuracy       |
| `yolov4-csp.yaml`           |  CSPDarknet (Scaling Cross Stage Partial Network) backbone/head | Reduces redundant computation, improves parameter efficiency and training stability          | Large model training efficiency and inference balance       |
| `yolov4-mish-rtdetr.yaml`           |                                Mish activation + RT-DETR head | Improved accuracy and better decoding/localization                | Real-time performance with high accuracy             |
| `yolov4-mish.yaml`           |                    Uses Mish activation (smoother than ReLU/Leaky ReLU) | Smoother gradients and better feature expression, common in high-accuracy models      | Accuracy-priority scenarios (higher computation acceptable)     |

### YOLOv5 Series
| Model Name                                                                    | Improved Module / Architecture (Summary)                                     | Improvements over Original YOLO                               | Strengths & Application Scenarios                       |
| ----------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------- | ------------------------------------ |
| `yolov5.yaml`           | Standard YOLOv5                           | Standard YOLOv5                      | General object detection                         |
| `yolov5-p6.yaml` / `yolov5-p7.yaml` / `yolov5-old-p6.yaml` / `yolov5-old.yaml` / `yolov5-p2.yaml` / `yolov5-p34.yaml` / `yolov5-p6.yaml` / `yolov5-p7.yaml` /                                 | Adjust output levels (corresponding to different pyramid layers)               | Supports larger/smaller scale detection needs (P7 is better for large scale)               | General object detection, specialized for small/medium/large objects                   |
|`yolov5-PPLCNet.yaml`           | PPLCNet (PaddlePaddle Lightweight Network) backbone | Lightweight PPLCNet feature extraction                       | Mobile/edge deployment         |
| `yolov5-AIFI.yaml`           | AIFI (Attention-based Intrascale Feature Interaction) backbone | Attention-guided feature extraction                       | Complex backgrounds or scenarios requiring fine features         |
| `yolov5-AKConv.yaml`           | AKConv (Adaptive Kernel Convolution)                                   | Replaces standard convolution, improves adaptability to objects of different sizes and shapes     | Small objects or irregularly shaped objects             |
| `yolov5-BoT3.yaml`           | BoT3 (Bottleneck Transformer)                             | Combines Multi-Head Self-Attention (MHSA) in C3 module               | Scenarios requiring global context capture         |
| `yolov5-CAConv.yaml`           | CAConv (Coordinate Attention Convolution)                        | Integrates coordinate attention, strengthens spatial position and channel relationships             | Fine localization, complex background scenarios               |
| `yolov5-CARAFE.yaml`           | CARAFE (Content-Aware ReAssembly of FEatures)                           | Uses CARAFE module for upsampling, better recovery of detailed features     | Small objects / edge refinement scenarios     |
| `yolov5-CCFM.yaml`           | CCFM (Cross-Channel Feature Fusion Module)                                 | Improves multi-channel cross-layer fusion, enhances representation quality                   | Complex backgrounds, multi-object scenarios               |
| `yolov5-CNeB-neck.yaml`           | CNeB-neck (Cross-Stage Network Block Neck)                             | Adjusts Neck structure to improve feature fusion or lightweighting               | Scenarios seeking balance between efficiency and accuracy             |
| `yolov5-CoordAtt.yaml`           | CoordAtt (Coordinate Attention)                                     | Adds coordinate attention to C3 module, captures direction and position-sensitive information | Small objects, scenarios where positional information is important           |
| `yolov5-CPCA.yaml`           | CPCA (Channel and Position Cross Attention)                               | Enhances interaction between channel and position attention                         | Accuracy improvement in complex scenarios                 |
| `yolov5-CrissCrossAttention.yaml`           | Criss-Cross Attention                      | Enhances spatial attention mechanism     | Tasks requiring strong attention mechanisms               |
| `yolov5-D-LKAAttention.yaml`           | D-LKA (Deformable Large Kernel Attention)                                | Combines large receptive field and deformable attention, improves performance for distant and irregular objects | Distant/irregular object detection                   |
| `yolov5-DAttention.yaml`           |DAttention (Dynamic Attention Module) backbone | Dynamically allocates attention weights, improves robustness to noise                     | Scenarios with heavy background interference                     |    
| `yolov5-DCNv2.yaml`           | DCNv2 (Deformable Convolution v2)                                     | Uses learnable deformable sampling positions, improves adaptability to object shapes   | Non-rigid / deformable object detection                 |
| `yolov5-deconv.yaml`           | Deconv (Deconvolution Upsampling) neck                                     | Uses deconvolution instead of `nn.Upsample`, learnable upsampling parameters   | Scenarios requiring fine feature recovery               |
| `yolov5-Dyample.yaml`           | Dyample (Dynamic Sampling)                                        | Dynamically extracts local features, improves representation ability               | Structurally complex objects                         |
| `yolov5-ECAAttention.yaml`           | ECA (Efficient Channel Attention)                                      | Adds lightweight channel attention in C3 module, improves performance at low cost     | Lightweight scenarios seeking accuracy improvement           |
| `yolov5-EffectiveSE.yaml`           | EffectiveSE (Improved SE Attention)                            | Adds to C3 module, strengthens channel reweighting, low computation cost       | General accuracy improvement                         |
| `yolov5-GAMAttention.yaml`           | GAMAttention (Global Attention Module) backbone | Enhances global attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov5-goldyolo.yaml`           |          goldyolo (Integrated Optimization Design) | Integrates multiple improvement strategies, improves AP and speed ratio      | Scenarios seeking overall performance improvement         |
| `yolov5-hornet-backbone.yaml` / `yolov5-hornet-neck.yaml`           | Hornet (Recursive Gated Convolution) backbone                               | Uses `HorNet` blocks for backbone or neck, improves efficiency and performance   | High-accuracy tasks                     |
| `yolov5-l-mobilenetv3s.yaml` / `yolov5-mobile3s.yaml` / `yolov5-Lite-*` |                   MobileNet / Lite series backbone/variant | Ultra-lightweight, low-compute deployment                   | Mobile/edge deployment                      |
| `yolov5-LeakyReLU.yaml`           |                         Changes activation to LeakyReLU | More conservative activation choice (beneficial for some convergence)         | Training stability adjustment for some datasets                  |
| `yolov5-mobile3s.yaml` / `yolov5-mobilv3l.yaml`           |                   MobileNetv3 backbone/variant               | Ultra-lightweight, low-compute deployment                   | Mobile/edge deployment                      |
| `yolov5-mobileone-backbone.yaml` / `yolov5-MobileOne-Lite-g.yaml` / `yolov5-MobileOne.yaml` | MobileOne series backbone/variant               | Ultra-lightweight, low-compute deployment                   | Mobile/edge deployment                      |
| `yolov5-ODConvNext.yaml`           | ODConvNext (Dynamic Convolution Next)                                | Introduces `ODConv`, a dynamic convolution with multi-dimensional learning | Complex feature demand scenarios                     |
| `yolov5-RepVGG.yaml` / `yolov5-RepVGG-A1-backbone.yaml`           | RepVGG (Re-parameterized VGG)                                     | Multi-branch during training, merged into single convolution for inference, balances accuracy and speed | Scenarios requiring both training accuracy and inference efficiency     |
| `yolov5-rtdetr.yaml`           | RT-DETR head                                        | Detection head replaced with `RTDETRDecoder` for more stable localization and classification     | Real-time performance with high accuracy           |
| `yolov5-scal-zoom.yaml`           |                    scale/zoom augmentation or multi-scale strategy | More robust to different scales                | Multi-scale dataset adaptation                      |
| `yolov5-SEAttention.yaml`           | SEAttention (Squeeze-and-Excitation Attention) backbone | Enhances SE attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov5-SegNextAttention.yaml`           | SegNextAttention backbone | Enhances semantic segmentation attention mechanism                       | Tasks requiring strong semantic segmentation       |
| `yolov5-ShuffleAttention.yaml` / `yolov5-Shufflenetv2.yaml`             | ShuffleNetV2 / ShuffleAttention backbone | Enhances channel and spatial attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov5-SimSPPF.yaml`           | SimSPPF (Simple SPPF) backbone | Simplified SPPF module                       | Tasks requiring efficient feature extraction       |
| `yolov5-SKAttention.yaml`           |  SKAttention (Selective Kernel Attention) backbone | Enhances selective kernel attention mechanism                       | Scenarios with significant multi-scale/shape variation       |
| `yolov5-SPPCSPC.yaml`           | SPPCSPC (SPP + CSP variant) backbone | Combines SPP and CSP, enhances multi-scale pooling and feature representation       | Small object and representation stability improvement               |
| `yolov5-transformer.yaml`           | Transformer module                                           | Adds `TransformerBlock` at the end of backbone, improves global context modeling | Long-range dependency or complex scenarios             |
| `yolov5-TripletAttention.yaml`           | TripletAttention backbone | Enhances triplet attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov5-VanillaNet.yaml`           | VanillaNet (Minimalist Network)                                   | Uses `VanillaNet` as backbone, pursues the simplest architecture       | Scenarios requiring easy deployment/debugging             |

### YOLOv6 Series
| Model Name                                                                                       | Improved Module / Architecture (Summary)                                     | Improvements over Original YOLO                                       | Strengths & Application Scenarios                       |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| `yolov6.yaml`           | Standard YOLOv6  | Standard YOLOv6      | General object detection                   |
| `yolov6-3.0-p2.yaml` / `yolov6-3.0-p34.yaml` / `yolov6-3.0-p6.yaml` / `yolov6-3.0-p7.yaml` | Adjust output levels (P2/P34/P6/P7 correspond to different pyramid layers)               | Supports larger/smaller scale detection needs (P7 is better for large scale)               | General object detection, specialized for small/medium/large objects                   |
| `yolov6-3.0-rtdetr.yaml` / `yolov6-4.0-rtdetr.yaml`           | RT-DETR head                                        | Detection head replaced with `RTDETRDecoder` for more stable localization and classification     | Real-time performance with high accuracy           |
| `yolov6-4.0-CPCA.yaml`           | CPCA (Channel and Position Cross Attention)                               | Enhances interaction between channel and position attention                         | Accuracy improvement in complex scenarios                 |
| `yolov6-4.0-CrissCrossAttention.yaml`           | Criss-Cross Attention                                   | Enhances spatial attention mechanism     | Tasks requiring strong attention mechanisms               |
| `yolov6-4.0-D-LKAAttention.yaml`           | D-LKA (Deformable Large Kernel Attention)                                | Combines large receptive field and deformable attention, improves performance for distant and irregular objects | Distant/irregular object detection                   |
| `yolov6-4.0-DAttention.yaml`           | DAttention (Dynamic Attention Module) backbone | Dynamically allocates attention weights, improves robustness to noise                     | Scenarios with heavy background interference                     |
| `yolov6-4.0-GAMAttention.yaml`           | GAMAttention (Global Attention Module) backbone | Enhances global attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov6-4.0-SEAttention.yaml`           | SEAttention (Squeeze-and-Excitation Attention) backbone | Enhances SE attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov6-4.0-SegNextAttention-obb.yaml`           | SegNeXt Attention + Rotated Box (OBB)                             | Combines SegNeXt attention and rotated box prediction, improves rotated object detection performance      | Rotated object detection (e.g., remote sensing images)           |
| `yolov6-4.0-ShuffleAttention-obb.yaml`           | ShuffleAttention backbone + Rotated Box (OBB)                           | Enhances channel and spatial attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov6-4.0-SKAttention-obb.yaml`           | SKAttention + Rotated Box (OBB)                                | Multi-scale kernel attention combined with rotated box prediction                             | Multi-scale rotated object detection                   |
| `yolov6-4.0-TripletAttention-obb.yaml`           | TripletAttention backbone + Rotated Box (OBB) | Enhances triplet attention mechanism                       | Tasks requiring strong attention mechanisms       |

### YOLOv7 Series
| Model Name                                                                    | Improved Module / Architecture (Summary)                                     | Improvements over Original YOLO                                       | Strengths & Application Scenarios                       |
| ----------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| `yolov7.yaml` / `yolov7-x.yaml` / `yolov7-w6.yaml` /  `yolov7-tiny.yaml` / `yolov7-tiny-silu.yaml` / `yolov7-e6e.yaml` / `yolov7-e6.yaml` / `yolov7-d6.yaml`|  Standard YOLOv7  | Standard YOLOv7  | General object detection, specialized for small/medium/large objects          |
| `yolov7-af-i.yaml`           |         AF-I (Lightweight Module) | Fewer parameters while maintaining representation capability                       | Mobile/edge deployment         |
| `yolov7-af.yaml`           |         AF (Lightweight Module)   | Fewer parameters while maintaining representation capability                       | Mobile/edge deployment         |
| `yolov7-C3C2-CPCA.yaml` / `yolov7-C3C2-CPCA-u6.yaml`           |       C3C2 Module (C3 variant, cross-stage design) | Improves feature flow/fusion efficiency and representation capability          | Medium-sized models / accuracy improvement scenarios         |
| `yolov7-C3C2-CrissCrossAttention.yaml` / `yolov7-C3C2-CrissCrossAttention-u6.yaml`           |       Integrates Criss-Cross Attention in C3C2 architecture                      | Enhances spatial attention mechanism     | Tasks requiring strong attention mechanisms               |
| `yolov7-C3C2-GAMAttention.yaml` / `yolov7-C3C2-GAMAttention-u6.yaml`           |       Integrates GAMAttention (Global Attention Module) backbone in C3C2 architecture | Enhances global attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov7-C3C2-RepVGG.yaml` / `yolov7-C3C2-RepVGG-u6.yaml`           |       Integrates RepVGG (Re-parameterized VGG) module in C3C2 architecture                        | Multi-branch during training, merged into single convolution for inference, balances accuracy and speed | Scenarios requiring both training accuracy and inference efficiency     |
| `yolov7-C3C2-ResNet.yaml` / `yolov7-C3C2-ResNet-u6.yaml`           |       Integrates ResNet module in C3C2 architecture | Selects attention enhancement details based on task            | Targeted task optimization (e.g., Small objects / complex backgrounds) |
| `yolov7-C3C2-SegNextAttention.yaml` / `yolov7-C3C2-SegNextAttention-u6.yaml`           |       Integrates SegNextAttention backbone in C3C2 architecture | Enhances semantic segmentation attention mechanism                       | Tasks requiring strong semantic segmentation capabilities       |
| `yolov7-C3C2.yaml` / `yolov7-C3C2-u6.yaml`           |       C3C2 Module (C3 variant, cross-stage design) | Improves feature flow/fusion efficiency and representation capability          | Medium-sized models / accuracy improvement scenarios         |
| `yolov7-DCNv2.yaml` / `yolov7-DCNv2-u6.yaml`           | DCNv2 (Deformable Convolution v2)                                     | Uses learnable deformable sampling positions, improves adaptability to object shapes   | Non-rigid / deformable object detection                 |
| `yolov7-goldyolo.yaml` / `yolov7-goldyolo-u6.yaml` / `yolov7-goldyolo-simple.yaml`           |          goldyolo (Integrated Optimization Design) | Collection of multiple improvement strategies, improves AP and speed ratio      | Scenarios seeking overall performance improvement         |
| `yolov7-MobileOne.yaml` / `yolov7-MobileOne-u6.yaml` / `yolov7-tiny-MobileOne.yaml`           |    MobileOne lightweight backbone | Inference speed optimization, suitable for mobile/embedded           | Edge devices / mobile            |
| `yolov7-RepNCSPELAN.yaml` / `yolov7-RepNCSPELAN-u6.yaml`           |         RepNCSPELAN (Composite Module) backbone | Combines Rep design with NCSPELAN-like optimization  | Balances training representation and inference efficiency         |
| `yolov7-rtdetr.yaml` / `yolov7-rtdetr-u6.yaml`           | RT-DETR head                                        | Replaces detection head with `RTDETRDecoder`, pursuing more stable localization and classification     | Requires high accuracy while providing real-time performance           |
| `yolov7-simple.yaml`           |          Simplified YOLOv7 | Reduces parameters and computation, improves speed              | Scenarios requiring ultra-fast inference            |
| `yolov7-tiny-AKConv.yaml`           | AKConv (Adaptive Kernel Convolution)                                   | Replaces standard convolution, improves adaptability to objects of different sizes and shapes     | Small objects or irregularly shaped objects             |
| `yolov7-tiny-goldyolo-simple.yaml`           |          goldyolo (Integrated Optimization Design) | Collection of multiple improvement strategies, improves AP and speed ratio      | Scenarios seeking overall performance improvement         |
| `yolov7-tiny-goldyolo.yaml`           |          goldyolo (Integrated Optimization Design) | Collection of multiple improvement strategies, improves AP and speed ratio      | Scenarios seeking overall performance improvement         |
| `yolov7-tiny-MobileNetv3.yaml`           |       MobileNetv3 lightweight backbone | Lightweight design, suitable for mobile/embedded              | Edge devices / mobile            |
| `yolov7-tiny-MobileOne.yaml`           |    MobileOne lightweight backbone | Inference speed optimization, suitable for mobile/embedded           | Edge devices / mobile            |
| `yolov7-tiny-PPLCNet.yaml`           | PPLCNet (PaddlePaddle Lightweight Network) backbone | Lightweight PPLCNet feature extraction                       | Mobile/edge deployment         |
| `yolov7-tiny-RepNCSPELAN.yaml`           |         RepNCSPELAN (Composite Module) backbone | Combines Rep design with NCSPELAN-like optimization  | Balances training representation and inference efficiency         |
| `yolov7-tiny-rtdetr.yaml`           | RT-DETR head                                        | Replaces detection head with `RTDETRDecoder`, pursuing more stable localization and classification     | Requires high accuracy while providing real-time performance           |
| `yolov7-tiny-simple.yaml`           |          Simplified YOLOv7-tiny | Reduces parameters and computation, improves speed              | Scenarios requiring ultra-fast inference            |
| `yolov7-u6.yaml`           |        YOLOv7-u6 (Large Scale Input) | Suitable for high-resolution input, improves small object detection         | High-resolution images / small object detection         |

### YOLOv8 Series
| Model Name                                                                                              |                                           Improved Module / Architecture (Summary) | Improvements over Original YOLO                     | Strengths & Application Scenarios           |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------: | --------------------------------- | ----------------- |
| `yolov8.yaml`           |                    Standard YOLOv8  | Standard YOLOv8       | General object detection                |
|`yoloe-v8.yaml`           |                    YOLOEDetect head  | Open-vocabulary object detection model      | Suitable for open-vocabulary, general object detection    |
| `yolov8-cls-resnet101.yaml` / `yolov8-cls-resnet50.yaml` / `yolov8-cls.yaml`                       |                                 ResNet101/50 backbone | Enhances feature extraction for classification tasks                     | Image classification tasks           |
| `yolov8-ghost.yaml` / `yolov8-ghost-p2.yaml` / `yolov8-ghost-p6.yaml`           |                             GhostModule/backbone (Lightweight Module) | Fewer parameters while maintaining representation capability                       | Mobile/edge deployment         |
| `yolov8-rtdetr.yaml`           | RT-DETR head                                        | Replaces detection head with `RTDETRDecoder`, pursuing more stable localization and classification     | Requires high accuracy while providing real-time performance           |
| `yolov8-world.yaml` / `yolov8-worldv2.yaml`           |                                 WorldDetect head   | Open-vocabulary object detection model                   | Suitable for open-vocabulary, general object detection             |
| `yolov8-AIFI.yaml`           | AIFI (Attention-based Intrascale Feature Interaction) backbone | Attention-guided feature extraction                       | Tasks requiring strong attention mechanisms       |
| `yolov8-AKConv.yaml`           | AKConv (Adaptive Kernel Convolution)                                   | Replaces standard convolution, improves adaptability to objects of different sizes and shapes     | Small objects or irregularly shaped objects             |
| `yolov8-BoT3.yaml`           | BoT3 (Bottleneck Transformer)                             | Combines Multi-Head Self-Attention (MHSA) in C2f module               | Scenarios requiring global context capture         |
| `yolov8-C2f-DAttention.yaml` | C2f + DAttention backbone | Combines C2f with dynamic attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov8-C2f-DRB.yaml` | C2f + DRB (Dynamic Residual Block) backbone | Combines C2f with dynamic residual block                       | Tasks requiring flexible feature extraction       |
| `yolov8-C2f-EMBC.yaml` | C2f + EMBC (Efficient Multi-Branch Convolution) backbone | Combines C2f with efficient multi-branch convolution                       | Tasks requiring efficient feature extraction       |
| `yolov8-C2f-EMSC.yaml` | C2f + EMSC (Efficient Multi-Scale Convolution) backbone | Combines C2f with efficient multi-scale convolution                       | Tasks requiring multi-scale feature extraction       |
| `yolov8-C2f-EMSCP.yaml` | C2f + EMSCP (Efficient Multi-Branch and Multi-Scale Convolution and Pooling) backbone | Combines C2f with efficient multi-scale convolution and pooling                       | Tasks requiring multi-scale feature extraction       |
| `yolov8-C2f-FasterBlock.yaml` | C2f + FasterBlock backbone | Combines C2f with FasterBlock                       | Tasks requiring efficient feature extraction       |
| `yolov8-C2f-GhostModule-DynamicConv.yaml` | C2f + GhostModule + DynamicConv backbone | Combines C2f, GhostModule, and Dynamic Convolution                       | Mobile/edge deployment         |
| `yolov8-C2f-MSBlockv2.yaml` | C2f + MSBlockv2 backbone | Combines C2f with MSBlockv2                       | Tasks requiring multi-scale feature extraction       |
| `yolov8-C2f-OREPA.yaml` | C2f + OREPA (Optimized Re-parameterization) backbone | Combines C2f with OREPA                       | Training-deployment integration optimization       |
| `yolov8-C2f-REPVGGOREPA.yaml` | C2f + RepVGG + OREPA backbone | Combines C2f, RepVGG, and OREPA                       | Training-deployment integration optimization       |
| `yolov8-C2f-RetBlock.yaml` | C2f + RetBlock backbone | Combines C2f with RetBlock                       | Tasks requiring efficient feature extraction       |
| `yolov8-C2f-RVB-EMA.yaml` | C2f + RVB (RepVGG Module) + EMA backbone | Combines C2f, RVB, and EMA                       | Training-deployment integration optimization       |
| `yolov8-C2f-RVB.yaml` | C2f + RVB (RepVGG Module) backbone | Combines C2f with RVB                       | Tasks requiring efficient feature extraction       |
| `yolov8-C2f-Star-CAA.yaml` | C2f + Star-CAA (Criss-Cross Attention) backbone | Combines C2f with Criss-Cross Attention                       | Tasks requiring strong attention mechanisms       |
| `yolov8-C2f-StarNet.yaml` | C2f + StarNet backbone | Combines C2f with StarNet                       | Tasks requiring strong attention mechanisms       |
| `yolov8-C2f-UniRepLKNetBlock.yaml` | C2f + UniRepLKNetBlock backbone | Combines C2f with UniRepLKNetBlock                       | Tasks requiring efficient feature extraction       |
| `yolov8-CAConv.yaml` | CAConv (Coordinate Attention Convolution)                        | Integrates coordinate attention, strengthens spatial position and channel relationships             | Fine localization, complex background scenarios               |
| `yolov8-CNeB-neck.yaml` | CNeB-neck (Cross-Stage Network Block) neck                             | Adjusts Neck structure to improve feature fusion or lightweighting               | Scenarios seeking balance between efficiency and accuracy             |
| `yolov8-CoordAtt.yaml` | CoordAtt (Coordinate Attention)                                     | Adds coordinate attention to C2f module, captures direction and position-sensitive information | Small objects, scenarios where positional information is important               |
| `yolov8-CPAarch.yaml` | CPAarch (Channel and Position Attention Architecture) backbone | Combines channel and position attention mechanisms                       | Tasks requiring strong attention mechanisms       |
| `yolov8-CPCA.yaml` | CPCA (Channel and Position Cross Attention)                               | Enhances interaction between channel and position attention                         | Accuracy improvement in complex scenarios                 |
| `yolov8-CrissCrossAttention.yaml` | Criss-Cross Attention                      | Enhances spatial attention mechanism     | Tasks requiring strong attention mechanisms               |
| `yolov8-D-LKAAttention.yaml` | D-LKA (Deformable Large Kernel Attention)                                | Combines large receptive field and deformable attention, improves performance for distant and irregular objects | Distant/irregular object detection                   |
| `yolov8-DAttention.yaml` | DAttention (Dynamic Attention Module) backbone | Dynamically allocates attention weights, improves robustness to noise                     | Scenarios with heavy background interference                     |    
| `yolov8-DCNv2.yaml` | DCNv2 (Deformable Convolution v2)                                     | Uses learnable deformable sampling positions, improves adaptability to object shapes   | Non-rigid / deformable object detection                 |
| `yolov8-deconv.yaml` | Deconv (Deconvolution Upsampling) neck                                     | Uses deconvolution layer instead of `nn.Upsample`, learnable upsampling parameters   | Scenarios requiring fine feature recovery               |
| `yolov8-DiT-C2f-UIB-FMDI.yaml` | DiT + C2f + UIB (Unified Interaction block) + FMDI (Feature Multi-Dimension Interaction) backbone | Combines multiple modules to enhance representation capability                       | High-accuracy tasks             |
| `yolov8-ECAAttention.yaml` | ECAAttention (Efficient Channel Attention) backbone | Efficient channel attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov8-EffectiveSE.yaml` | EffectiveSE (Improved SE Attention)                            | Adds to C3 module, strengthens channel reweighting, low computation cost       | General accuracy improvement                         |
| `yolov8-Faster-Block-CGLU.yaml` | Faster-Block + CGLU (Convolutional Gated Linear Unit) backbone | Combines Faster-Block with CGLU                       | Tasks requiring efficient feature extraction       |
| `yolov8-Faster-EMA.yaml` | Faster-EMA (Exponential Moving Average) backbone | Combines Faster-Block with EMA                       | Training-deployment integration optimization       |
| `yolov8-GAMAttention.yaml` | GAMAttention (Global Attention Module) backbone | Enhances global attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov8-goldyolo.yaml` | goldyolo (Integrated Optimization Design) | Collection of multiple improvement strategies, improves AP and speed ratio      | Scenarios seeking overall performance improvement         |
| `yolov8-hornet-backbone.yaml` | Hornet (Recursive Gated Convolution) backbone                               | Uses `HorNet` blocks for backbone or neck, improves efficiency and performance   | High-accuracy tasks                     |
| `yolov8-hornet-neck.yaml` | Hornet neck | Neck enhancing feature fusion                       | Tasks requiring strong feature fusion       |
| `yolov8-HWD.yaml` | HWD (Hierarchical Weight Decomposition) backbone | Feature extraction with hierarchical weight decomposition                       | Tasks requiring efficient feature extraction       |
| `yolov8-l-mobilenetv3s.yaml` | Lite MobileNetv3s backbone | Lightweight MobileNetv3s feature extraction                       | Mobile/edge deployment         |
| `yolov8-LCDConv.yaml` | LCDConv (Lightweight Context Decomposition Convolution) backbone | Lightweight context decomposition convolution                       | Mobile/edge deployment         |
| `yolov8-LeakyReLU.yaml` | Changes activation to LeakyReLU | More conservative activation choice (beneficial for some convergence)         | Training stability adjustment for some datasets                  |
| `yolov8-Lite-c.yaml` | Lite-c (Lightweight Variant) backbone | Lightweight variant                       | Mobile/edge deployment         |
| `yolov8-Lite-g.yaml` | Lite-g (Lightweight Variant with Ghost Module) backbone | Lightweight variant combined with Ghost module                       | Mobile/edge deployment         |
| `yolov8-Lite-s.yaml` | Lite-s (Lightweight Small Variant) backbone | Lightweight small variant                       | Mobile/edge deployment         |
| `yolov8-MHSA.yaml` | MHSA (Multi-Head Self-Attention) backbone | Enhances multi-head self-attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov8-mobile3s.yaml` | Mobile3s backbone | Lightweight MobileNetv3 feature extraction                       | Mobile/edge deployment         |
| `yolov8-mobileone-backbone.yaml` | MobileOne backbone | Lightweight MobileOne feature extraction                       | Mobile/edge deployment         |
| `yolov8-MobileOne.yaml` | MobileOne neck | Neck enhancing feature fusion                       | Tasks requiring strong feature fusion       |
| `yolov8-mobilev3l.yaml` | MobileV3-Large backbone | Lightweight MobileNetv3 Large feature extraction                       | Mobile/edge deployment         |
| `yolov8-MSFM.yaml` | MSFM (Multi-Scale Feature Module) backbone | Module enhancing multi-scale features                       | Tasks requiring multi-scale feature extraction       |
| `yolov8-ODConvNext.yaml` | ODConvNext (Dynamic Convolution Next)                                | Introduces `ODConv`, a dynamic convolution with multi-dimensional learning | Complex feature demand scenarios                     |
| `yolov8-p2.yaml`、`yolov8-p34.yaml`、`yolov8-p6.yaml`、 `yolov8-p7.yaml` |Adjust output levels (P2/P34/P6/P7 correspond to different pyramid layers)               | Supports larger/smaller scale detection needs (P7 is better for large scale)               | General object detection, specialized for small/medium/large objects                   |
| `yolov8-PPLCNet.yaml` | PPLCNet (PaddlePaddle Lightweight Network) backbone | Lightweight PPLCNet feature extraction                       | Mobile/edge deployment         |
| `yolov8-RepNCSPELAN.yaml` |         RepNCSPELAN (Composite Module) backbone | Combines Rep design with NCSPELAN-like optimization  | Balances training representation and inference efficiency         |
| `yolov8-RepVGG-A1-backbone.yaml` | RepVGG-A1 backbone | Lightweight RepVGG-A1 feature extraction                       | Mobile/edge deployment         |
| `yolov8-RepVGG.yaml` | RepVGG (Re-parameterized VGG)                                     | Multi-branch during training, merged into single convolution for inference, balances accuracy and speed | Scenarios requiring both training accuracy and inference efficiency     |
| `yolov8-RepViTBlock.yaml` | RepViTBlock backbone | Combines Rep and ViT feature extraction                       | Tasks requiring efficient feature extraction       |
| `yolov8-rtdetr.yaml` | RT-DETR head                                        | Replaces detection head with `RTDETRDecoder`, pursuing more stable localization and classification     | Requires high accuracy while providing real-time performance           |
| `yolov8-SEAttention.yaml` | SEAttention (Squeeze-and-Excitation Attention) backbone | Enhances SE attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov8-SegNextAttention.yaml` | SegNextAttention backbone | Enhances semantic segmentation attention mechanism                       | Tasks requiring strong semantic segmentation capabilities       |
| `yolov8-ShuffleAttention.yaml` | ShuffleAttention backbone | Enhances channel and spatial attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov8-Shufflenetv2.yaml` | ShuffleNetv2 backbone | Lightweight ShuffleNetv2 feature extraction                       | Mobile/edge deployment         |
| `yolov8-SimAM.yaml` | SimAM (Simple Attention Module) backbone | Simplified attention module                       | Tasks requiring strong attention mechanisms       |
| `yolov8-SimSPPF.yaml` | SimSPPF (Simple SPPF) backbone | Simplified SPPF module                       | Tasks requiring efficient feature extraction       |
| `yolov8-SKAttention.yaml` | SKAttention (Selective Kernel Attention) backbone | Enhances selective kernel attention mechanism                       | Scenarios with significant multi-scale/shape variation       |
| `yolov8-SPDConv.yaml` | SPDConv (Spatially Pooled Convolution) backbone | Enhances spatially pooled convolution feature extraction                       | Tasks requiring efficient feature extraction       |
| `yolov8-SPPCSPC.yaml` | SPPCSPC (SPP + CSP variant) backbone | Combines SPP and CSP, enhances multi-scale pooling and feature representation       | Small object and representation stability improvement               |
| `yolov8-StripNet-sn2.yaml` | StripNet-sn2 backbone | Lightweight StripNet-sn2 feature extraction                       | Mobile/edge deployment         |
| `yolov8-SwinTransformer.yaml` | Swin Transformer backbone | Transformer enhancing local and global features                       | Tasks requiring strong context understanding capabilities       |
| `yolov8-TripletAttention.yaml` | TripletAttention backbone | Enhances triplet attention mechanism                       | Tasks requiring strong attention mechanisms       |
| `yolov8-VanillaNet.yaml` | VanillaNet (Minimalist Network)                                   | Uses `VanillaNet` as backbone, pursues the simplest architecture       | Scenarios requiring easy deployment/debugging             |

### YOLOv9 Series
| Model Name                                  | Improved Module / Architecture (Summary)                                               | Improvements over Original YOLO                      | Strengths & Application Scenarios                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov9*.yaml`           | Standard YOLOv9  | Standard YOLOv9          | General object detection                     |
| `gelan-c-AKConv.yaml`           | AKConv (Adaptive Kernel Convolution)                                   | Replaces standard convolution, improves adaptability to objects of different sizes and shapes     | Small objects or irregularly shaped objects             |
| `gelan-c-DCNV3RepNCSPELAN4.yaml`           | DCNv3 + RepNCSPELAN (Deformable Convolution + Re-parameterization/Composite Module)                       | Adapts to deformable objects, training-inference trade-off               | Non-rigid objects / complex shapes                 |
| `gelan-c-DualConv.yaml`           | DualConv (Dual Path Convolution)                                                     | Improves channel/spatial information separation and fusion                 | Complex backgrounds / multi-scale                    |
| `gelan-c-FasterRepNCSPELAN.yaml`           | FasterBlock + RepNCSPELAN                                              | Accelerates while maintaining representation capability                       | Requires high throughput without sacrificing accuracy            |
| `gelan-c-OREPAN.yaml`           | OREPA (Re-parameterized attention/fusion)                                         | Strong training, simplified inference                            | Training-deployment integration optimization                   |
| `gelan-c-PANet.yaml`           | PANet (Path Aggregation Network)                                                | Enhances multi-scale feature fusion                          | Complex backgrounds / Multi-scale objects                 |
| `gelan-c-SCConv.yaml` / `gelan-c-SPDConv.yaml`           | SCConv / SPDConv (Special Convolution Variants)                                         | Improves local/multi-scale feature extraction                     | Multi-scale / structurally variable objects                 |
| `gelan-s-FasterRepNCSPELAN.yaml`           | s-FasterBlock + RepNCSPELAN (Lightweight Version)                                    | Lightweight while maintaining representation capability                     | Mobile / lightweight scenarios                      |
| `gelan-c-dpose.yaml`           | dpose variant (with pose head)                                          | Simultaneous detection and pose estimation                         | Human/animal pose estimation                      |
| `gelan-c-dseg.yaml`           | dseg variant (with segmentation head)                                    | Simultaneous detection and semantic segmentation                         | Semantic segmentation tasks                          |

### YOLOv10 Series
| Model Name                                  | Improved Module / Architecture (Summary)                                               | Improvements over Original YOLO                      | Strengths & Application Scenarios                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov10*.yaml`           | Standard YOLOv10  | Standard YOLOv10      | General object detection                     |
| `yolov10n-ADNet.yaml`           | ADNet (Dedicated attention / decoder module)                                   | Improves classification/localization consistency                        | Accuracy priority while maintaining speed                |
| `yolov10n-ADown.yaml`           | ADown (Adaptive Downsampling Module)                                                 | Reduces computation, improves speed                        | Scenarios requiring speed optimization                  |
| `yolov10n-AIFI.yaml`           | AIFI (Attention-based Intrascale Feature Interaction) backbone | Attention-guided feature extraction                       | Tasks requiring strong attention mechanisms       |
| `yolov10n-AirNet.yaml`           | AirNet (Lightweight backbone / fusion)                                          | Lightweight while maintaining representation capability                       | Mobile / edge deployment                   |
| `yolov10n-ASF.yaml` / `yolov10n-ASFF.yaml`      | ASF / ASFF (Attention Fusion Module)                                              | Improves feature fusion and background suppression                      | Complex backgrounds / low contrast images               |
| `yolov10n-BiFormer.yaml` / `yolov10n-BiFPN.yaml`| BiFormer / BiFPN (Transformer-like / BiFPN)                             | Better context modeling and pyramid fusion                  | Requires large-scale context or multi-scale fusion             |
| `yolov10n-C2f-CSPHet.yaml`           | CSPHet (CSP + Heterogeneous Attention)                                              | Enhances feature extraction and fusion                          | Complex scenarios / Multi-scale objects                 |
| `yolov10n-C2f-CSPPC.yaml`           | CSPPC (CSP + Pixel Attention)                                              | Improves pixel-level feature fusion                      | Scenarios requiring high-resolution output                   |
| `yolov10n-C2f-DLKA.yaml`           | DLKA (Deep Deformable Attention)                                              | Expands receptive field, improves performance for distant and irregular objects           | Distant / irregular objects                       |
| `yolov10n-C2f-DWRSeg.yaml`           | DWRSeg (Deep Deformable Segmentation)                                              | Improves segmentation accuracy and boundary detection                      | Scenarios requiring fine segmentation                     |
| `yolov10n-C2f-GhostModule.yaml`           | GhostModule (Lightweight Module)                                             | Reduces computation, improves speed                        | Scenarios requiring speed optimization                    |
| `yolov10n-C2f-iRMB.yaml`           | iRMB (Enhanced Re-parameterization Module)                                           | Improves feature representation capability                            | Scenarios requiring enhanced feature extraction                   |
| `yolov10n-C2f-MLLABlock.yaml`           | MLLABlock (Multi-Level Lightweight Module)                                        | Reduces computation, improves speed                        | Scenarios requiring speed optimization                    |
| `yolov10n-C2f-MSBlock.yaml`           | MSBlock (Multi-Scale Feature Extraction Module)                                        | Enhances multi-scale feature extraction                          | Complex backgrounds / Multi-scale objects                 |
| `yolov10n-C2f-ODConv.yaml`           | ODConv (Deformable Convolution Module)                                            | Expands receptive field, improves performance for distant and irregular objects           | Distant / irregular objects                       |
| `yolov10n-C2f-OREPA.yaml`           | OREPA (Re-parameterization Module)                                              | Strong training, simplified inference                            | Training-deployment process optimization                      |
| `yolov10n-C2f-RepELAN-high.yaml`           | RepELAN-high (Efficient Re-parameterization Module)                                    | Improves feature representation capability                            | Scenarios requiring enhanced feature extraction                   |
| `yolov10n-C2f-RepELAN-low.yaml`           | RepELAN-low (Lightweight Re-parameterization Module)                                  | Reduces computation, improves speed                        | Scenarios requiring speed optimization                    |
| `yolov10n-C2f-SAConv.yaml`           | SAConv (Spatial Attention Convolution)                                          | Enhances spatial feature extraction                            | Complex backgrounds / Multi-scale objects                 |
| `yolov10n-C2f-ScConv.yaml`           | ScConv (Spatial Convolution)                                                | Improves spatial feature extraction                            | Complex backgrounds / Multi-scale objects                 |
| `yolov10n-C2f-SENetV1.yaml`           | SENetV1 (Channel Attention Network V1)                                      | Improves channel feature extraction                            | Complex backgrounds / Multi-scale objects                 |
| `yolov10n-C2f-SENetV2.yaml`           | SENetV2 (Channel Attention Network V2)                                      | Improves channel feature extraction                            | Complex backgrounds / Multi-scale objects                 |
| `yolov10n-C2f-Triple.yaml`           | Triple (Multiple Feature Fusion Module)                                      | Enhances multiple feature fusion                            | Complex scenarios / Multi-scale objects                 |
| `yolov10n-CCFM.yaml`           | CCFM (Cross-Channel Feature Fusion Module)                                 | Improves multi-channel cross-layer fusion, enhances representation quality                   | Complex backgrounds, multi-object scenarios               |
| `yolov10n-DAT.yaml`           | DAT (Dual Attention Variant)                                                  | Improves feature fusion and background suppression                      | Complex backgrounds / low contrast images               |
| `yolov10n-DLKA.yaml`           | DLKA (Large Kernel + Deformable Attention)                                              | Expands receptive field, improves performance for distant and irregular objects           | Distant / irregular objects                       |
| `yolov10n-DynamicConv.yaml`           | DynamicConv (Dynamic Convolution)                                                   | Adaptive convolution for local features                        | Scenarios requiring adaptive local representation                |
| `yolov10n-EVC.yaml`           | EVC (Efficient Convolution)                                                           | Improves convolution efficiency                              | Scenarios requiring efficient computation                      |
| `yolov10n-FFA.yaml`           | FFA (Feature Fusion Module)                                                      | Enhances feature fusion and representation capability                        | Complex scenarios / Multi-scale objects                   |
| `yolov10n-FocalModulation.yaml`           | FocalModulation                                            | Improves feature extraction for key regions                    | Scenarios requiring emphasis on specific regions                  |
| `yolov10n-HAT.yaml`           | HAT (Efficient Attention Module)                                                  | Reduces computation, improves speed                          | Scenarios requiring speed optimization                      |
| `yolov10n-HGNet-l.yaml`           | HGNet-l (Lightweight High-Order Feature Network)                                          | Reduces computation, improves speed                          | Scenarios requiring speed optimization                      |
| `yolov10n-HGNet-x.yaml`           | HGNet-x (High-Order Feature Network)                                                | Enhances feature extraction and fusion                            | Complex scenarios / Multi-scale objects                   |
| `yolov10n-IAT.yaml`           | IAT (Image Attention Module)                                                  | Improves image feature extraction capability                          | Scenarios requiring emphasis on image features                  |
| `yolov10n-iRMB.yaml`           | iRMB (Lightweight Inverted Module)                                                | Reduces computation, improves speed                          | Scenarios requiring speed optimization                      |
| `yolov10n-Light-HGNet-l.yaml`           | Light-HGNet-l (Lightweight High-Order Feature Network)                                  | Reduces computation, improves speed                          | Scenarios requiring speed optimization                      |
| `yolov10n-Light-HGNet-x.yaml`           | Light-HGNet-x (Lightweight High-Order Feature Network)                                  | Enhances feature extraction and fusion                            | Complex scenarios / Multi-scale objects                   |
| `yolov10n-LSKA.yaml`           | LSKA (Lightweight Spatial Attention)                                              | Reduces computation, improves speed                          | Scenarios requiring speed optimization                      |
| `yolov10n-MBformer.yaml`           | MBformer (Lightweight Transformer)                                             | Reduces computation, improves speed                          | Scenarios requiring speed optimization                      |
| `yolov10n-MultiSEAM.yaml`           | MultiSEAM (Multi-Scale Adaptive Module)                                       | Enhances multi-scale feature extraction                            | Complex scenarios / Multi-scale objects                   |
| `yolov10n-OREPA.yaml`           | OREPA (Object Re-identification and Re-localization Module)                                     | Improves object re-identification and re-localization capability                    | Scenarios requiring emphasis on object identification                  |
| `yolov10n-RCSOSA.yaml`           | RCSOSA (Re-parameterized Cross-Attention Module)                                   | Improves feature fusion and representation quality                        | Complex scenarios requiring strong channel fusion                   |
| `yolov10n-RepGFPN.yaml`           | RepGFPN (Re-parameterized Feature Pyramid Network)                                   | Improves feature pyramid representation capability                      | Complex scenarios / Multi-scale objects                   |
| `yolov10n-RIDNet.yaml`           | RIDNet (Lightweight Re-identification Network)                                         | Reduces computation, improves speed                          | Scenarios requiring speed optimization                      |
| `yolov10n-SEAM.yaml`           | SEAM (Adaptive Feature Fusion Module)                                       | Enhances feature fusion and representation capability                        | Complex scenarios / Multi-scale objects                   |
| `yolov10n-SENetV2.yaml`           | SENetV2 (Improved SENet)                                            | Improves feature extraction and fusion capability                        | Complex scenarios / Multi-scale objects                   |
| `yolov10n-SlimNeck.yaml`           | SlimNeck (Lightweight Neck Network)                                       | Reduces computation, improves speed                          | Scenarios requiring speed optimization                      |
| `yolov10n-SPDConv.yaml`           | SPDConv (Spatial Attention Convolution)                                        | Improves convolution efficiency                              | Scenarios requiring efficient computation                      |
| `yolov10n-SPPELAN.yaml`           | SPPELAN (Spatial Pixel-level Feature Fusion Module)                              | Enhances spatial pixel-level feature fusion                        | Complex scenarios / Multi-scale objects                   |

### YOLOv11 Series
| Model Name                                  | Improved Module / Architecture (Summary)                                               | Improvements over Original YOLO                      | Strengths & Application Scenarios                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov11.yaml`           | Standard YOLOv11                                                   |Standard YOLOv11          | General object detection                     |
|`yoloe-v11.yaml`           |                    YOLOEDetect head  | Open-vocabulary object detection model      | Suitable for open-vocabulary, general object detection    |
| `yolov11-RGBIR.yaml`           | RGB + IR Dual-Branch Backbone (Multiin + C3k2 + C2PSA) fused at P3–P5 | Capable of integrating visible/infrared dual-modal features | Night detection, thermal imaging assistance, multi-spectral tasks |
| `yolov11-cls-resnet18.yaml`           |           ResNet18 backbone | Enhances feature extraction for classification tasks                     | Image classification tasks           |
| `yolov11-ASF.yaml`           | ASF (Adaptive Fusion/Attention)                                                  | Improves multi-scale fusion and background suppression                    | Complex backgrounds / Small objects                      |
| `yolov11-BCN.yaml`           | | BCN (Bidirectional Convolutional Network)                                                | Enhances feature extraction and fusion                          | Complex backgrounds / Multi-scale objects                 |
| `yolov11-BiFPN.yaml`           | BiFPN (Bidirectional Feature Pyramid Network)                                                | Enhances multi-scale feature fusion                         | Multi-scale object detection                        |
| `yolov11-C2PSA-CGA.yaml`           | C2PSA (Channel-Position Self-Attention) combined with CGA (Channel Guided Attention)                       | Enhances channel and position interaction                       | Complex backgrounds, multi-object scenarios                   |
| `yolov11-C2PSA-DAT.yaml`           | C2PSA combined with DAT (Dual Attention Transformer)                                   | Combines local and global attention to improve representation capability             | Complex backgrounds, multi-object scenarios                   |
| `yolov11-C2PSA-DiT-CCFM.yaml` / `yolov11-C2PSA-DiT.yaml`| C2PSA combined with DiT (Dual Transformer) (and CCFM (Cross-Channel Fusion Module)) | Enhances channel/position interaction, improves global context modeling capability | Complex backgrounds, multi-object scenarios                   |
| `yolov11-C2PSA-SENetV2-LightHGNetV2-l.yaml` / `yolov11-C2PSA-SENetV2-LightHGNetV2-l-CCFM.yaml`| C2PSA combined with SENetV2 and LightHGNetV2-l (Lightweight Backbone) (and CCFM (Cross-Channel Fusion Module))         | Lightweight design, enhances channel and position interaction               | Mobile / lightweight scenarios                      |
| `yolov11-C3k2-ConvNeXtV2Block-BiFPN.yaml` / `yolov11-C3k2-ConvNeXtV2Block-BiFPN.yaml`| C3k2 (New block) with ConvNeXtV2 Block and BiFPN                      | Improves representation flow and multi-scale fusion                    | Medium-large models requiring stronger representation capability           |
| `yolov11-C3K2-DiTBlock.yaml` | C3K2 with DiT Block (Dual Transformer Block) | Improves representation flow and global context modeling                | Medium-large models requiring stronger representation capability           |
| `yolov11-C3k2-FasterBlock-OREPA-v10Detect.yaml` | C3k2 with FasterBlock and OREPA (Re-parameterized Attention) | Improves representation flow and inference efficiency                      | Medium-large models requiring stronger representation capability           |
| `yolov11-C3k2-MLLABlock-2-SlimNeck.yaml` / `yolov11-C3k2-MLLABlock-2.yaml` | C3k2 with MLLABlock-2 (and SlimNeck) | Improves representation flow and inference efficiency                      | Medium-large models requiring stronger representation capability           |
| `yolov11-C3k2-OREPA-backbone-v10Detect.yaml` / `yolov11-C3k2-OREPA-backbone.yaml` | C3k2 with OREPA backbone (Re-parameterized Attention) | Improves representation flow and inference efficiency                      | Medium-large models requiring stronger representation capability           |
| `yolov11-C3k2-UIB-CCFM.yaml` / `yolov11-C3k2-UIB-FMDI.yaml` / `yolov11-C3k2-UIB.yaml` | C3k2 with UIB (Unified Interaction Block)| Improves representation flow and channel/position interaction                  | Medium-large models requiring stronger representation capability           |
| `yolov11-C3k2-WTConv.yaml` | C3k2 with WTConv (Weighted Convolution)| Improves representation flow and fusion efficiency                      | Medium-large models requiring stronger representation capability           |
| `yolov11-CAFormer.yaml`    | CAFormer (Channel Attention Transformer)                                           | Combines channel attention and global context modeling               | Complex backgrounds, multi-object scenarios                   |
| `yolov11-CCFM.yaml` / `yolov11-CCFM-C2PSA-DAT.yaml` / `yolov11-CCFM-C2PSA-DAT-v10Detect.yaml`| CCFM combined with C2PSA / DAT etc. composite attention                                           | Enhances channel/position interaction and cross-layer fusion                  | Complex backgrounds, multi-object scenarios                   |
| `yolov11-ConvFormer.yaml`      | ConvFormer (Convolution + Transformer Hybrid)                                       | Combines local convolution and global attention                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-COSNet.yaml`          | COSNet (Channel Attention + Spatial Attention)                                         | Combines channel and spatial attention, improves feature representation capability       | Complex backgrounds, multi-object scenarios                   |
| `yolov11-DecoupleNet.yaml`      | DecoupleNet (Decoupled Head Design)                                                | Separates classification and regression tasks, improves accuracy                  | Scenarios requiring high accuracy                        |
| `yolov11-DiT-C3k2-UIB-CCFM.yaml` | DiT combined with C3k2, UIB and CCFM                                               | Enhances channel/position interaction and global context modeling             | Complex backgrounds, multi-object scenarios                   |
| `yolov11-DiT-C3k2-UIB-FMDI-IDetect.yaml` / `yolov11-DiT-C3k2-UIB-FMDI.yaml`  | DiT combined with C3k2, UIB and FMDI (Feature Multi-Scale Bidirectional Interaction)| Enhances channel/position interaction and global context modeling             | Complex backgrounds, multi-object scenarios                   |
| `yolov11-DiT-C3k2-WTConv-CCFM.yaml` | DiT combined with C3k2, WTConv and CCFM                                               | Enhances channel/position interaction and global context modeling             | Complex backgrounds, multi-object scenarios                   |
| `yolov11-DiT-CCFM-IDetect.yaml` / `yolov11-DiT-CCFM.yaml`  | DiT combined with CCFM (Cross-Channel Fusion Module)                                | Enhances channel/position interaction and global context modeling             | Complex backgrounds, multi-object scenarios                   |
| `yolov11-DiT.yaml`               | DiT (Dual Transformer)                                                   | Improves global context modeling capability                       | Complex backgrounds, multi-object scenarios                   |
| `yolov11-DySnakeConv.yaml`               | DySnakeConv (Dynamic Snake Convolution)                                               | Improves feature representation and flow capability                       | Complex backgrounds, multi-object scenarios                   |
| `yolov11-EfficientNet-CCFM-v10Detect.yaml` | EfficientNet combined with CCFM (Cross-Channel Fusion Module)                    | Enhances channel/position interaction and global context modeling             | Complex backgrounds, multi-object scenarios                   |
| `yolov11-EfficientNet-OREPA-v10Detect.yaml` | EfficientNet combined with OREPA (Re-parameterized Attention)                               | Improves representation flow and inference efficiency                      | Medium-large models requiring stronger representation capability           |
| `yolov11-EfficientNet.yaml`               | EfficientNet                                                               | Improves representation flow and inference efficiency                      | Medium-large models requiring stronger representation capability           |
| `yolov11-EfficientViM.yaml` / `yolov11-EfficientViT_MIT.yaml` | EfficientViM / EfficientViT_MIT (Efficient Transformer)                        | Improves global context modeling and efficiency                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-EMOv2.yaml`               | EMOv2 (Context-Aware Module)                                                   | Combines context awareness to improve representation capability                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-EViT.yaml`               | EViT (Efficient Vision Transformer)                                   | Improves global context modeling and efficiency                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-FasterNet.yaml`               | FasterNet (High-Efficiency Backbone)                                               | Enhances representation flow and inference efficiency                      | Edge devices / mobile terminals            |
| `yolov11-FloraNet.yaml`           | FloraNet (Lightweight Backbone)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-FMDI.yaml`               | FMDI (Feature Multi-Scale Bidirectional Interaction)                                               | Enhances multi-scale feature fusion                         | Multi-scale object detection                        |
| `yolov11-GLNet.yaml`             | GLNet (Global-Local Network)                                                | Combines global and local features                         | Complex backgrounds, multi-object scenarios                   |
| `yolov11-hyper.yaml`             | hyper (Hyperparameter or Special Structure Integration)                                              | Model architecture/training strategy adjustment to improve stability            | Specific dataset optimization                         |
| `yolov11-IdentityFormer.yaml`    | IdentityFormer (Lightweight Transformer)                                      | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-iFormer.yaml`           | iFormer (Hybrid Convolution and Transformer)                                       | Combines local convolution and global attention                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-KW_ResNet.yaml`         | KW_ResNet (Key Re-parameterized ResNet)                                            | Improves representation flow and inference efficiency                      | Medium-large models requiring stronger representation capability           |
| `yolov11-LAE.yaml`               | LAE (Lightweight Attention Enhancement)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-LAUDNet.yaml`           | LAUDNet (Lightweight Backbone)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-LightHGNetV2-l.yaml`    | LightHGNetV2-l (Lightweight Backbone)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-LSNet.yaml`             | LSNet (Lightweight Backbone)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-Mamba-v10Detect.yaml` / `yolov11-Mamba.yaml`  | Mamba (Multi-Scale Attention Module)                                               | Enhances multi-scale feature fusion                         | Multi-scale object detection                        |
| `yolov11-MASF.yaml`              | MASF (Multi-Scale Adaptive Feature)                                               | Enhances multi-scale feature fusion                         | Multi-scale object detection                        |
| `yolov11-MLLA.yaml`             | MLLA (Multi-Level Lightweight Attention)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-MobileNetv4.yaml`      | MobileNetv4 (Lightweight Backbone)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-OverLoCK.yaml`         | OverLoCK (Cross-Level Attention Module)                                               | Enhances cross-level feature fusion                         | Multi-scale object detection                        |
| `yolov11-PKINet.yaml`           | PKINet (Position Key Interaction Network)                                               | Enhances position interaction representation                           | Complex backgrounds, multi-object scenarios                   |
| `yolov11-PoolFormerv2.yaml`    | PoolFormerv2 (Lightweight Pooling Transformer)                                     | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-pst.yaml`              | PST (Pyramid Sparse Transformer) | Improves multi-scale feature fusion and global context modeling efficiency | Complex backgrounds, multi-object scenarios                   |
| `yolov11-QARepVGG.yaml`         | QARepVGG (Quantized Re-parameterized VGG)                                             | Strong training, simplified inference                            | Requires deployment efficiency while maintaining high representation capability            |
| `yolov11-RandFormer.yaml`       | RandFormer (Random Attention Transformer)                                      | Improves global context modeling and efficiency                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-RepLKNet.yaml`         | RepLKNet (Re-parameterized Large Kernel Network)                                             | Strong training, simplified inference                            | Requires deployment efficiency while maintaining high representation capability            |
| `yolov11-ResNet_MoE.yaml`       | ResNet_MoE (Mixture of Experts)                                               | Improves model capacity and representation capability                       | Complex backgrounds, multi-object scenarios                   |
| `yolov11-RFAConv.yaml`          | RFAConv (Re-parameterized Attention Convolution)                                               | Improves representation flow and inference efficiency                      | Medium-large models requiring stronger representation capability           |
| `yolov11-SFSCNet.yaml`          | SFSCNet (Spatial Frequency Selection Convolution Network)                                           | Improves feature representation and flow capability                       | Complex backgrounds, multi-object scenarios                   |
| `yolov11-SGFormer.yaml`         | SGFormer (Sparse Global Attention Transformer)                                   | Improves global context modeling and efficiency                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-SlabPVTv2.yaml`        | SlabPVTv2 (Lightweight Pyramid Vision Transformer)                        | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-SlabSwinTransformer.yaml` | SlabSwinTransformer (Lightweight Swin Transformer)                        | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-SlimNeck.yaml`         | SlimNeck (Lightweight Neck Design)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-SMT.yaml`              | SMT (Sparse Mixed Transformer)                                               | Improves global context modeling and efficiency                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-SoftHGNN.yaml`         | SoftHGNN (Soft Hyper Graph Neural Network)                                               | Models high-level semantic relationships and enhances multi-scale feature reallocation and fusion                         | Complex backgrounds, multi-object scenarios                   |
| `yolov11-SPANet.yaml`           | SPANet (Spatial Attention Network)                                               | Combines spatial attention to improve representation capability                   | Complex backgrounds, multi-object scenarios                   |
| `yolov11-StripMLPNet.yaml`      | StripMLPNet (Strip MLP Network)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-StripNet-sn2.yaml`     | StripNet-sn2 (Strip Backbone Network)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-STViT-CCFM.yaml` | STViT-CCFM (Lightweight Transformer + CCFM) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-TransNeXt.yaml`    | TransNeXt (Aggregated Attention + Convolutional GLU) | Introduces Aggregated Attention to enhance global perception + Convolutional GLU to strengthen local feature modeling   | Complex backgrounds / Multi-object scenes / Multi-scale targets |
| `yolov11-TransXNet.yaml`    | TransXNet (D‑Mixer + MS-FFN)                         | Dynamically captures global + local features (IDConv + OSRA) + multi-scale fusion to enhance representation capability  | Complex backgrounds / Multi-object scenes / Multi-scale targets |
| `yolov11-UniNeXt-CCFM.yaml` | UniNeXt-CCFM (Universal Backbone + CCFM)             | High-dimensional feature fusion + local and global feature modeling + consistently improves performance of various STMs | Complex backgrounds / Multi-object scenes / Multi-scale targets |
| `yolov11-VAN.yaml`              | VAN (Visual Attention Network)                                               | Combines channel attention and global context modeling               | Complex backgrounds, multi-object scenarios                   |
| `yolov11-vHeat.yaml`            | vHeat (Lightweight Backbone)                                               | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-WTConvNeXt.yaml`       | WTConvNeXt (Weighted Convolution + ConvNeXt Hybrid)                                       | Combines local convolution and global attention                     | Complex backgrounds, multi-object scenarios                   |
| `yolov11-C2PSA-DiT-C3k2-WTConv-CCFM-pose.yaml` | C2PSA-DiT-C3k2-WTConv-CCFM (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-CoordConv-BiFPN-pose.yaml` | CoordConv-BiFPN (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-EfficientViM-CCFM-pose.yaml` | EfficientViM-CCFM (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-FasterNet-CCFM-pose.yaml` | FasterNet-CCFM (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-GroupMixFormer-CCFM-pose.yaml` | GroupMixFormer-CCFM (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-GSConv-BiFPN-pose.yaml` | GSConv-BiFPN (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-LightHGNetV2-l-CCFM-pose.yaml` | LightHGNetV2-l-CCFM (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-LSNet-CCFM-pose.yaml` | LSNet-CCFM (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-MobileOne-BiFPN-Lite-g-(i)pose.yaml` | MobileOne-BiFPN-Lite-g (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-SlimNeck-BiFPN-pose.yaml` | SlimNeck-BiFPN (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-SwinTransformer-C2PSA-DAT-pose.yaml` | SwinTransformer-C2PSA-DAT (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-SwinTransformer-DiT-pose.yaml` | SwinTransformer-DiT (Lightweight Pose Estimation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-C3k2-RepVGG-CCFM-seg.yaml` / `yolov11-C3k2-RepVGG-seg.yaml`| C3k2-RepVGG(-CCFM) (Lightweight Semantic Segmentation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-C3k2-SAConv-seg.yaml` | C3k2-SAConv (Lightweight Semantic Segmentation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-C3k2-WTConv-CCFM-seg.yaml` | C3k2-WTConv-CCFM (Lightweight Semantic Segmentation Model) | Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |
| `yolov11-Haar-seg.yaml` | Haar (Lightweight Backbone)| Lightweight design, improves efficiency                         | Mobile / lightweight scenarios                      |

### YOLOv12 Series
| Model Name                                  | Improved Module / Architecture (Summary)                                               | Improvements over Original YOLO                      | Strengths & Application Scenarios                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov12.yaml`           | Standard YOLOv12                                                   |Standard YOLOv12          | General object detection                     |
| `yolov12-ASF.yaml`           | ASF (Adaptive Fusion/Attention)                                                  | Improves multi-scale fusion and background suppression                    | Complex backgrounds / Small objects                      |
| `yolov12-CCFM.yaml`           | CCFM (Cross-Channel Feature Fusion Module)                                 | Improves multi-channel cross-layer fusion, enhances representation quality                   | Complex backgrounds, multi-object scenarios               |
| `yolov12-hyper.yaml`           | hyper (Hyperparameter or Special Structure Integration)                                              | Model architecture/training strategy adjustment to improve stability            | Specific dataset optimization                         |
| `yolov12-ShuffleAttention-CCFM.yaml`           | ShuffleAttention + CCFM                                                   | Lightweight attention enhances channel/position interaction                  | Mobile / lightweight scenarios                      |
| `yolov12-EMOv2-CCFM-pose.yaml`           | EMOv2 + CCFM + pose head                                                  | Combines context awareness and pose estimation                       | Human/animal pose estimation                      |
| `yolov12-TransXNet-CCFM-pose.yaml`           | TransXNet + CCFM + pose head                                              | Combines Transformer and pose estimation                  | High-accuracy pose estimation                         |
| `yolov12-MobileNetv4-CCFM-seg.yaml`           | MobileNetv4 + CCFM + segmentation head                                     | Combines lightweight design and semantic segmentation                         | Mobile / semantic segmentation                       |
| `yolov12-MobileNetv4-ShuffleAttention-seg.yaml` | MobileNetv4 + ShuffleAttention + segmentation head                        | Combines lightweight attention and semantic segmentation                     | Mobile / lightweight semantic segmentation                   |

### YOLOv13 Series
| Model Name                                  | Improved Module / Architecture (Summary)                                               | Improvements over Original YOLO                      | Strengths & Application Scenarios                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov13.yaml`                                   | Standard YOLOv13                                                   | Standard YOLOv13          | General object detection                     |
| `yolov13-sn2.yaml`                                | sn2 / specific block variant                                                     | Adjusts depth/width or block type                    | Choose different depth/performance trade-offs as needed           |
| `yolov13-pose.yaml`                               | pose variant (with pose head)                                            | Simultaneous detection and pose estimation                         | Human/animal pose estimation                      |

## models directory structure
```
ultralytics_pro\ultralytics\cfg\models
│  README.md
│  
├─alss-yolo
│  ├─Detect
│  │      alss-yolo-m.yaml
│  │      alss-yolo-n.yaml
│  │      alss-yolo-s.yaml
│  │      
│  ├─OBB
│  │      alss-yolo-m-obb.yaml
│  │      alss-yolo-n-obb.yaml
│  │      alss-yolo-s-obb.yaml
│  │      
│  ├─Pose
│  │      alss-yolo-m-pose.yaml
│  │      alss-yolo-n-pose.yaml
│  │      alss-yolo-s-pose.yaml
│  │      
│  └─Segment
│          alss-yolo-m-seg.yaml
│          alss-yolo-MSCAM-s-seg.yaml
│          alss-yolo-n-seg.yaml
│          alss-yolo-s-seg.yaml
│          
├─bgf-yolo
│  └─Detect
│          BGF-yolo.yaml
│          
├─cst-yolo
│  └─Detect
│          CST-yolo.yaml
│          
├─damoyolo
│  ├─Detect
│  │      DAMOyolo-b.yaml
│  │      DAMOyolo-l.yaml
│  │      DAMOyolo-m.yaml
│  │      DAMOyolo-n.yaml
│  │      DAMOyolo-s.yaml
│  │      DAMOyolo-t.yaml
│  │      DAMOyolo-x.yaml
│  │      
│  ├─OBB
│  │      DAMOyolo-b-obb.yaml
│  │      DAMOyolo-l-obb.yaml
│  │      DAMOyolo-m-obb.yaml
│  │      DAMOyolo-n-obb.yaml
│  │      DAMOyolo-s-obb.yaml
│  │      DAMOyolo-t-obb.yaml
│  │      DAMOyolo-x-obb.yaml
│  │      
│  ├─Pose
│  │      DAMOyolo-b-pose.yaml
│  │      DAMOyolo-l-pose.yaml
│  │      DAMOyolo-m-pose.yaml
│  │      DAMOyolo-n-pose.yaml
│  │      DAMOyolo-s-pose.yaml
│  │      DAMOyolo-t-pose.yaml
│  │      DAMOyolo-x-pose.yaml
│  │      
│  └─Segment
│          DAMOyolo-b-seg.yaml
│          DAMOyolo-l-seg.yaml
│          DAMOyolo-m-seg.yaml
│          DAMOyolo-n-seg.yaml
│          DAMOyolo-s-seg.yaml
│          DAMOyolo-t-seg.yaml
│          DAMOyolo-x-seg.yaml
│          
├─fbrt-yolo
│  ├─Detect
│  │      FBRT-yolo-C2f-FasterBlock-s.yaml
│  │      FBRT-yolo-l.yaml
│  │      FBRT-yolo-m.yaml
│  │      FBRT-yolo-n.yaml
│  │      FBRT-yolo-s.yaml
│  │      FBRT-yolo-x.yaml
│  │      
│  ├─OBB
│  │      FBRT-yolo-l-obb.yaml
│  │      FBRT-yolo-m-obb.yaml
│  │      FBRT-yolo-n-obb.yaml
│  │      FBRT-yolo-s-obb.yaml
│  │      FBRT-yolo-x-obb.yaml
│  │      
│  ├─Pose
│  │      FBRT-yolo-C2f-FasterBlock-s-pose.yaml
│  │      FBRT-yolo-l-pose.yaml
│  │      FBRT-yolo-m-pose.yaml
│  │      FBRT-yolo-n-pose.yaml
│  │      FBRT-yolo-s-pose.yaml
│  │      FBRT-yolo-x-pose.yaml
│  │      
│  └─Segment
│          FBRT-yolo-l-seg.yaml
│          FBRT-yolo-m-seg.yaml
│          FBRT-yolo-n-seg.yaml
│          FBRT-yolo-s-seg.yaml
│          FBRT-yolo-x-seg.yaml
│          
├─goldyolo
│  ├─Detect
│  │      GOLDYOLO-l.yaml
│  │      GOLDYOLO-m.yaml
│  │      GOLDYOLO-n.yaml
│  │      GOLDYOLO-s.yaml
│  │      GOLDYOLO-t.yaml
│  │      GOLDYOLO-x.yaml
│  │      
│  ├─OBB
│  │      GOLDYOLO-l-obb.yaml
│  │      GOLDYOLO-m-obb.yaml
│  │      GOLDYOLO-n-obb.yaml
│  │      GOLDYOLO-s-obb.yaml
│  │      GOLDYOLO-t-obb.yaml
│  │      GOLDYOLO-x-obb.yaml
│  │      
│  ├─Pose
│  │      GOLDYOLO-l-pose.yaml
│  │      GOLDYOLO-m-pose.yaml
│  │      GOLDYOLO-n-pose.yaml
│  │      GOLDYOLO-s-pose.yaml
│  │      GOLDYOLO-t-pose.yaml
│  │      GOLDYOLO-x-pose.yaml
│  │      
│  └─Segment
│          GOLDYOLO-l-seg.yaml
│          GOLDYOLO-m-seg.yaml
│          GOLDYOLO-n-seg.yaml
│          GOLDYOLO-s-seg.yaml
│          GOLDYOLO-t-seg.yaml
│          GOLDYOLO-x-seg.yaml
│          
├─HEYDet
│  ├─Detect
│  │      HEYDet-l.yaml
│  │      HEYDet-m.yaml
│  │      HEYDet-n.yaml
│  │      HEYDet-s.yaml
│  │      HEYDet-x.yaml
│  │      
│  ├─OBB
│  │      HEYDet-l-obb.yaml
│  │      HEYDet-m-obb.yaml
│  │      HEYDet-n-obb.yaml
│  │      HEYDet-s-obb.yaml
│  │      HEYDet-x-obb.yaml
│  │      
│  ├─Pose
│  │      HEYDet-l-pose.yaml
│  │      HEYDet-m-pose.yaml
│  │      HEYDet-n-pose.yaml
│  │      HEYDet-s-pose.yaml
│  │      HEYDet-x-pose.yaml
│  │      
│  └─Segment
│          HEYDet-l-seg.yaml
│          HEYDet-m-seg.yaml
│          HEYDet-n-seg.yaml
│          HEYDet-s-seg.yaml
│          HEYDet-x-seg.yaml
│          
├─hyper-yolo
│  ├─Classify
│  ├─Detect
│  │      gelan-c-hyper.yaml
│  │      gelan-m-hyper.yaml
│  │      gelan-s-hyper.yaml
│  │      gelan-t-hyper.yaml
│  │      hyper-yolo-l.yaml
│  │      hyper-yolo-m.yaml
│  │      hyper-yolo-n.yaml
│  │      hyper-yolo-s.yaml
│  │      hyper-yolo-t.yaml
│  │      hyper-yolo-x.yaml
│  │      
│  ├─OBB
│  ├─Pose
│  │      gelan-c-hyper-pose.yaml
│  │      hyper-yolo-m-pose.yaml
│  │      
│  └─Segment
│          hyper-yolo-m-seg.yaml
│          hyper-yolo-s-seg.yaml
│          
├─icon
│      10245193.png
│      200px-Tsinghua_University_Logo.svg.ico
│      200px-Tsinghua_University_Logo.svg.png
│      3ymrl-8dk73-001.ico
│      6pzh4-dzva7-001.ico
│      8jfp0-ho2pu-001.ico
│      a1nt0-vyv7i-001.ico
│      a1y74-lzygv-001.ico
│      Academia_Sinica_Emblem.svg.ico
│      Academia_Sinica_Emblem.svg.png
│      aqey5-br9gr-001.ico
│      Arms_of_Monash_University.svg.png
│      Beijing_Institute_of_Technology_Logo.svg.png
│      channels4_profile.ico
│      channels4_profile.jpg
│      channels5_profile.ico
│      channels5_profile.jpg
│      DALL·E 2024-01-26 23.24.57 - A die-cut sticker of a human figure with visual indicators like lines or dots representing pose estimation technology. The figure is stylized and abst.ico
│      DALL·E 2024-01-26 23.24.57 - A die-cut sticker of a human figure with visual indicators like lines or dots representing pose estimation technology. The figure is stylized and abst.png
│      DALL·E 2024-01-26 23.25.02 - A die-cut sticker representing object detection technology. The design features symbolic graphics like a magnifying glass or target symbols on various.ico
│      DALL·E 2024-01-26 23.25.02 - A die-cut sticker representing object detection technology. The design features symbolic graphics like a magnifying glass or target symbols on various.png
│      DALL·E 2024-01-26 23.25.06 - A die-cut sticker illustrating the concept of instance segmentation in image processing. The design features color-coded outlines or distinct segments.ico
│      DALL·E 2024-01-26 23.25.06 - A die-cut sticker illustrating the concept of instance segmentation in image processing. The design features color-coded outlines or distinct segments.png
│      DALL·E 2024-01-26 23.25.10 - A die-cut sticker symbolizing 'Oriented Bounding Boxes Object Detection'. The design showcases objects enclosed in rotated bounding boxes illustratin.ico
│      DALL·E 2024-01-26 23.25.10 - A die-cut sticker symbolizing 'Oriented Bounding Boxes Object Detection'. The design showcases objects enclosed in rotated bounding boxes, illustratin.png
│      DALL·E 2024-01-26 23.25.14 - A die-cut sticker representing the concept of 'Classify'. The design includes symbolic elements like labels categories or sorting visuals illustrat.ico
│      DALL·E 2024-01-26 23.25.14 - A die-cut sticker representing the concept of 'Classify'. The design includes symbolic elements like labels, categories, or sorting visuals, illustrat.png
│      darknet_logo_blue.ico
│      darknet_logo_blue.png
│      eb0tf-fs007-001.ico
│      Guangdong_University_of_Technology_Logo.svg.png
│      Huawei_Standard_logo.svg.ico
│      Huawei_Standard_logo.svg.png
│      sii4h-vwhiq-001.ico
│      unnamed.ico
│      unnamed.png
│      Untitled design.png
│      US Davis.ico
│      US Davis.png
│      world.png
│      ZJUT_seal.svg.png
│      下載.ico
│      下載.png
│      註解 2024-01-26 195508.ico
│      註解 2024-01-26 195508.png
│      註解 2024-07-13 233815.png
│      
├─Leyolo
│  ├─Detect
│  │      Leyolo-l.yaml
│  │      Leyolo-n.yaml
│  │      Leyolo-s.yaml
│  │      
│  ├─OBB
│  │      Leyolo-l-obb.yaml
│  │      Leyolo-n-obb.yaml
│  │      Leyolo-s-obb.yaml
│  │      
│  ├─Pose
│  │      Leyolo-l-pose.yaml
│  │      Leyolo-n-pose.yaml
│  │      Leyolo-s-pose.yaml
│  │      
│  └─Segment
│          Leyolo-l-seg.yaml
│          Leyolo-n-seg.yaml
│          Leyolo-s-seg.yaml
│          
├─maf-yolo
│  ├─Classify
│  ├─Detect
│  │      MAF-YOLOv2-lite-n.yaml
│  │      MAF-YOLOv2-m.yaml
│  │      MAF-YOLOv2-n.yaml
│  │      MAF-YOLOv2-s-C2PSA.yaml
│  │      MAF-YOLOv2-s-DiT.yaml
│  │      MAF-YOLOv2-s.yaml
│  │      
│  ├─OBB
│  ├─Pose
│  └─Segment
├─PicoDet
│  ├─Detect
│  │      PicoDet-l.yaml
│  │      PicoDet-m.yaml
│  │      PicoDet-n.yaml
│  │      PicoDet-s.yaml
│  │      PicoDet-t.yaml
│  │      PicoDet-x.yaml
│  │      
│  ├─OBB
│  │      PicoDet-l-obb.yaml
│  │      PicoDet-m-obb.yaml
│  │      PicoDet-n-obb.yaml
│  │      PicoDet-s-obb.yaml
│  │      PicoDet-t-obb.yaml
│  │      PicoDet-x-obb.yaml
│  │      
│  ├─Pose
│  │      PicoDet-l-pose.yaml
│  │      PicoDet-m-pose.yaml
│  │      PicoDet-n-pose.yaml
│  │      PicoDet-s-pose.yaml
│  │      PicoDet-t-pose.yaml
│  │      PicoDet-x-pose.yaml
│  │      
│  └─Segment
│          PicoDet-l-seg.yaml
│          PicoDet-m-seg.yaml
│          PicoDet-n-seg.yaml
│          PicoDet-s-seg.yaml
│          PicoDet-t-seg.yaml
│          PicoDet-x-seg.yaml
│          
├─ppyoloe
│  ├─Detect
│  │      PPyoloE-l.yaml
│  │      PPyoloE-m.yaml
│  │      PPyoloE-s.yaml
│  │      PPyoloE-x.yaml
│  │      
│  ├─OBB
│  │      PPyoloE-l-obb.yaml
│  │      PPyoloE-m-obb.yaml
│  │      PPyoloE-s-obb.yaml
│  │      PPyoloE-x-obb.yaml
│  │      
│  ├─Pose
│  │      PPyoloE-l-pose.yaml
│  │      PPyoloE-m-pose.yaml
│  │      PPyoloE-s-pose.yaml
│  │      PPyoloE-x-pose.yaml
│  │      
│  └─Segment
│          PPyoloE-l-seg.yaml
│          PPyoloE-m-seg.yaml
│          PPyoloE-s-seg.yaml
│          PPyoloE-x-seg.yaml
│          
├─pst
│  ├─Classify
│  │      r101-cls-pst.yaml
│  │      r18-cls-pst.yaml
│  │      r50-cls-pst.yaml
│  │      
│  └─Detect
│          r101-pst.yaml
│          r18-pst.yaml
│          r50-pst.yaml
│          
├─R
│  ├─Classify
│  │      yoloR-cls-d6.yaml
│  │      yoloR-cls-e6.yaml
│  │      yoloR-cls-p6.yaml
│  │      yoloR-cls-w6.yaml
│  │      yoloR-csp-cls.yaml
│  │      yoloR-csp-x-cls.yaml
│  │      
│  ├─Detect
│  │      r50-csp.yaml
│  │      x50-csp.yaml
│  │      yoloR-csp-rtdetr.yaml
│  │      yoloR-csp-x-rtdetr.yaml
│  │      yoloR-csp-x.yaml
│  │      yoloR-csp.yaml
│  │      yoloR-d6.yaml
│  │      yoloR-e6.yaml
│  │      yoloR-p6.yaml
│  │      yoloR-s2d.yaml
│  │      yoloR-w6.yaml
│  │      
│  ├─OBB
│  │      r50-csp-obb.yaml
│  │      x50-csp-obb.yaml
│  │      yoloR-csp-obb.yaml
│  │      yoloR-csp-x-obb.yaml
│  │      yoloR-obb-d6.yaml
│  │      yoloR-obb-e6.yaml
│  │      yoloR-obb-p6.yaml
│  │      yoloR-obb-s2d.yaml
│  │      yoloR-obb-w6.yaml
│  │      
│  ├─Pose
│  │      r50-csp-pose.yaml
│  │      x50-csp-pose.yaml
│  │      yoloR-csp-pose.yaml
│  │      yoloR-csp-x-pose.yaml
│  │      yoloR-pose-d6.yaml
│  │      yoloR-pose-e6.yaml
│  │      yoloR-pose-p6.yaml
│  │      yoloR-pose-s2d.yaml
│  │      yoloR-pose-w6.yaml
│  │      
│  └─Segment
│          r50-csp-seg.yaml
│          x50-csp-seg.yaml
│          yoloR-csp-seg.yaml
│          yoloR-csp-x-seg.yaml
│          yoloR-seg-d6.yaml
│          yoloR-seg-e6.yaml
│          yoloR-seg-p6.yaml
│          yoloR-seg-s2d.yaml
│          yoloR-seg-w6.yaml
│          
├─rcs-yolo
│  └─Detect
│          RCS-yolo.yaml
│          RCS3-yolo.yaml
│          
├─rt-detr
│  └─Detect
│          rtdetr-l.yaml
│          rtdetr-resnet101.yaml
│          rtdetr-resnet50.yaml
│          rtdetr-x.yaml
│          
├─RTMDet
│  ├─Detect
│  │      RTMDet-l.yaml
│  │      RTMDet-m.yaml
│  │      RTMDet-n.yaml
│  │      RTMDet-s.yaml
│  │      RTMDet-t.yaml
│  │      RTMDet-x.yaml
│  │      
│  ├─OBB
│  │      RTMDet-l-obb.yaml
│  │      RTMDet-m-obb.yaml
│  │      RTMDet-n-obb.yaml
│  │      RTMDet-s-obb.yaml
│  │      RTMDet-t-obb.yaml
│  │      RTMDet-x-obb.yaml
│  │      
│  ├─Pose
│  │      RTMDet-l-pose.yaml
│  │      RTMDet-m-pose.yaml
│  │      RTMDet-n-pose.yaml
│  │      RTMDet-s-pose.yaml
│  │      RTMDet-t-pose.yaml
│  │      RTMDet-x-pose.yaml
│  │      
│  └─Segment
│          RTMDet-l-seg.yaml
│          RTMDet-m-seg.yaml
│          RTMDet-n-seg.yaml
│          RTMDet-s-seg.yaml
│          RTMDet-t-seg.yaml
│          RTMDet-x-seg.yaml
│          
├─syolo
│  └─Detect
│          Syolo-s.yaml
│          Syolo.yaml
│          
├─v10
│  └─Detect
│          yolov10b.yaml
│          yolov10l.yaml
│          yolov10m.yaml
│          yolov10n-ADNet.yaml
│          yolov10n-ADown.yaml
│          yolov10n-AIFI.yaml
│          yolov10n-AirNet.yaml
│          yolov10n-ASF.yaml
│          yolov10n-ASFF.yaml
│          yolov10n-BiFormer.yaml
│          yolov10n-BiFPN.yaml
│          yolov10n-C2f-CSPHet.yaml
│          yolov10n-C2f-CSPPC.yaml
│          yolov10n-C2f-DLKA.yaml
│          yolov10n-C2f-DWRSeg.yaml
│          yolov10n-C2f-GhostModule.yaml
│          yolov10n-C2f-iRMB.yaml
│          yolov10n-C2f-MLLABlock.yaml
│          yolov10n-C2f-MSBlock.yaml
│          yolov10n-C2f-ODConv.yaml
│          yolov10n-C2f-OREPA.yaml
│          yolov10n-C2f-RepELAN-high.yaml
│          yolov10n-C2f-RepELAN-low.yaml
│          yolov10n-C2f-SAConv.yaml
│          yolov10n-C2f-ScConv.yaml
│          yolov10n-C2f-SENetV1.yaml
│          yolov10n-C2f-SENetV2.yaml
│          yolov10n-C2f-Triple.yaml
│          yolov10n-CCFM.yaml
│          yolov10n-DAT.yaml
│          yolov10n-DLKA.yaml
│          yolov10n-DynamicConv.yaml
│          yolov10n-EVC.yaml
│          yolov10n-FFA.yaml
│          yolov10n-FocalModulation.yaml
│          yolov10n-HAT.yaml
│          yolov10n-HGNet-l.yaml
│          yolov10n-HGNet-x.yaml
│          yolov10n-IAT.yaml
│          yolov10n-iRMB.yaml
│          yolov10n-Light-HGNet-l.yaml
│          yolov10n-Light-HGNet-x.yaml
│          yolov10n-LSKA.yaml
│          yolov10n-MBformer.yaml
│          yolov10n-MultiSEAM.yaml
│          yolov10n-OREPA.yaml
│          yolov10n-RCSOSA.yaml
│          yolov10n-RepGFPN.yaml
│          yolov10n-RIDNet.yaml
│          yolov10n-SEAM.yaml
│          yolov10n-SENetV2.yaml
│          yolov10n-SlimNeck.yaml
│          yolov10n-SPDConv.yaml
│          yolov10n-SPPELAN.yaml
│          yolov10n.yaml
│          yolov10s.yaml
│          yolov10x.yaml
│          
├─v11
│  ├─Classify
│  │      yolov11-cls-pst.yaml
│  │      yolov11-cls-resnet18.yaml
│  │      yolov11-cls.yaml
│  │      
│  ├─Detect
│  │      yoloe-v11.yaml
│  │      yolov11-ASF.yaml
│  │      yolov11-BCN.yaml
│  │      yolov11-BiFPN.yaml
│  │      yolov11-C2PSA-CGA.yaml
│  │      yolov11-C2PSA-DAT.yaml
│  │      yolov11-C2PSA-DiT-CCFM.yaml
│  │      yolov11-C2PSA-DiT.yaml
│  │      yolov11-C2PSA-SENetV2-LightHGNetV2-l-CCFM.yaml
│  │      yolov11-C2PSA-SENetV2-LightHGNetV2-l.yaml
│  │      yolov11-C3k2-ConvNeXtV2Block-BiFPN.yaml
│  │      yolov11-C3K2-DiTBlock.yaml
│  │      yolov11-C3k2-FasterBlock-OREPA-v10Detect.yaml
│  │      yolov11-C3k2-MLLABlock-2-SlimNeck.yaml
│  │      yolov11-C3k2-MLLABlock-2.yaml
│  │      yolov11-C3k2-OREPA-backbone-v10Detect.yaml
│  │      yolov11-C3k2-OREPA-backbone.yaml
│  │      yolov11-C3k2-UIB-CCFM.yaml
│  │      yolov11-C3k2-UIB-FMDI.yaml
│  │      yolov11-C3k2-UIB.yaml
│  │      yolov11-C3k2-WTConv.yaml
│  │      yolov11-CAFormer.yaml
│  │      yolov11-CCFM-C2PSA-DAT-v10Detect.yaml
│  │      yolov11-CCFM-C2PSA-DAT.yaml
│  │      yolov11-CCFM.yaml
│  │      yolov11-ConvFormer.yaml
│  │      yolov11-COSNet.yaml
│  │      yolov11-DecoupleNet.yaml
│  │      yolov11-DiT-C3k2-UIB-CCFM.yaml
│  │      yolov11-DiT-C3k2-UIB-FMDI-IDetect.yaml
│  │      yolov11-DiT-C3k2-UIB-FMDI.yaml
│  │      yolov11-DiT-C3k2-WTConv-CCFM.yaml
│  │      yolov11-DiT-CCFM-IDetect.yaml
│  │      yolov11-DiT-CCFM.yaml
│  │      yolov11-DiT.yaml
│  │      yolov11-DySnakeConv.yaml
│  │      yolov11-EfficientNet-CCFM-v10Detect.yaml
│  │      yolov11-EfficientNet-OREPA-v10Detect.yaml
│  │      yolov11-EfficientNet.yaml
│  │      yolov11-EfficientViM.yaml
│  │      yolov11-EfficientViT_MIT.yaml
│  │      yolov11-EMOv2.yaml
│  │      yolov11-EViT.yaml
│  │      yolov11-FasterNet.yaml
│  │      yolov11-FloraNet.yaml
│  │      yolov11-FMDI.yaml
│  │      yolov11-GLNet.yaml
│  │      yolov11-hyper.yaml
│  │      yolov11-IdentityFormer.yaml
│  │      yolov11-iFormer.yaml
│  │      yolov11-KW_ResNet.yaml
│  │      yolov11-LAE.yaml
│  │      yolov11-LAUDNet.yaml
│  │      yolov11-LightHGNetV2-l.yaml
│  │      yolov11-LSNet.yaml
│  │      yolov11-Mamba-v10Detect.yaml
│  │      yolov11-Mamba.yaml
│  │      yolov11-MASF.yaml
│  │      yolov11-MLLA.yaml
│  │      yolov11-MobileNetv4.yaml
│  │      yolov11-OREPA-C2PSA-DAT-v10Detect.yaml
│  │      yolov11-OREPA-v10Detect.yaml
│  │      yolov11-OREPA.yaml
│  │      yolov11-OverLoCK.yaml
│  │      yolov11-PKINet.yaml
│  │      yolov11-PoolFormerv2.yaml
│  │      yolov11-pst.yaml
│  │      yolov11-QARepVGG.yaml
│  │      yolov11-RandFormer.yaml
│  │      yolov11-RepGFPN.yaml
│  │      yolov11-RepLKNet.yaml
│  │      yolov11-ResNet_MoE.yaml
│  │      yolov11-RFAConv.yaml
│  │      yolov11-RGBIR.yaml
│  │      yolov11-SFSCNet.yaml
│  │      yolov11-SGFormer.yaml
│  │      yolov11-SlabPVTv2.yaml
│  │      yolov11-SlabSwinTransformer.yaml
│  │      yolov11-SlimNeck.yaml
│  │      yolov11-SMT.yaml
│  │      yolov11-SoftHGNN.yaml
│  │      yolov11-SPANet.yaml
│  │      yolov11-StripMLPNet.yaml
│  │      yolov11-StripNet-sn2.yaml
│  │      yolov11-STViT-CCFM.yaml
│  │      yolov11-TransNeXt.yaml
│  │      yolov11-TransXNet.yaml
│  │      yolov11-UniNeXt-CCFM.yaml
│  │      yolov11-VAN.yaml
│  │      yolov11-vHeat.yaml
│  │      yolov11-WTConvNeXt.yaml
│  │      yolov11.yaml
│  │      
│  ├─OBB
│  │      yolov11-DecoupleNet-obb.yaml
│  │      yolov11-obb.yaml
│  │      
│  ├─Pose
│  │      yolov11-BiFPN-pose.yaml
│  │      yolov11-C2PSA-DiT-C3k2-WTConv-CCFM-pose.yaml
│  │      yolov11-C3k2-ConvNeXtV2Block-BiFPN-pose.yaml
│  │      yolov11-CCFM-C2PSA-DAT-pose.yaml
│  │      yolov11-CCFM-pose.yaml
│  │      yolov11-CoordConv-BiFPN-pose.yaml
│  │      yolov11-EfficientViM-CCFM-pose.yaml
│  │      yolov11-FasterNet-CCFM.yaml
│  │      yolov11-GroupMixFormer-pose.yaml
│  │      yolov11-GSConv-BiFPN-pose.yaml
│  │      yolov11-hyper-pose.yaml
│  │      yolov11-LightHGNetV2-l-CCFM-pose.yaml
│  │      yolov11-LSNet-CCFM-pose.yaml
│  │      yolov11-LWGANet-CCFM-pose.yaml
│  │      yolov11-MobileOne-BiFPN-Lite-g-ipose.yaml
│  │      yolov11-MobileOne-BiFPN-Lite-g-pose.yaml
│  │      yolov11-pose.yaml
│  │      yolov11-SlimNeck-BiFPN-pose.yaml
│  │      yolov11-SwinTransformer-C2PSA-DAT-pose.yaml
│  │      yolov11-SwinTransformer-DiT-pose.yaml
│  │      
│  └─Segment
│          yoloe-v11-seg.yaml
│          yolov11-C3k2-RepVGG-CCFM-seg.yaml
│          yolov11-C3k2-RepVGG-seg.yaml
│          yolov11-C3k2-SAConv-seg.yaml
│          yolov11-C3k2-WTConv-CCFM-seg.yaml
│          yolov11-C3k2-WTConv-seg.yaml
│          yolov11-CCFM-seg.yaml
│          yolov11-Haar-seg.yaml
│          yolov11-hyper-seg.yaml
│          yolov11-LightHGNetV2-l-seg.yaml
│          yolov11-seg.yaml
│          yolov11-WTConvNeXt-seg.yaml
│          
├─v12
│  ├─Classify
│  │      yolov12-cls.yaml
│  │      
│  ├─Detect
│  │      yolov12-ASF.yaml
│  │      yolov12-CCFM.yaml
│  │      yolov12-hyper.yaml
│  │      yolov12-ShuffleAttention-CCFM.yaml
│  │      yolov12.yaml
│  │      
│  ├─OBB
│  │      yolov12-obb.yaml
│  │      
│  ├─Pose
│  │      yolov12-CCFM-pose.yaml
│  │      yolov12-EMOv2-CCFM-pose.yaml
│  │      yolov12-pose.yaml
│  │      yolov12-TransXNet-CCFM-pose.yaml
│  │      
│  └─Segment
│          yolov12-CCFM-seg.yaml
│          yolov12-MobiloeNetv4-CCFM-seg.yaml
│          yolov12-MobiloeNetv4-ShuffleAttention-seg.yaml
│          yolov12-seg.yaml
│          yolov12-ShuffleAttention-CCFM-seg.yaml
│          
├─v13
│  ├─Detect
│  │      yolov13-sn2.yaml
│  │      yolov13.yaml
│  │      
│  ├─OBB
│  ├─Pose
│  │      yolov13-pose.yaml
│  │      
│  └─Segment
├─v14
├─v3
│  ├─Classify
│  │      yolov3-cls.yaml
│  │      yolov3-spp-cls.yaml
│  │      yolov3-tiny-cls.yaml
│  │      
│  ├─Detect
│  │      yolov3-rtdetr.yaml
│  │      yolov3-spp-rtdetr.yaml
│  │      yolov3-spp.yaml
│  │      yolov3-tiny-rtdetr.yaml
│  │      yolov3-tiny.yaml
│  │      yolov3.yaml
│  │      
│  ├─OBB
│  │      yolov3-obb.yaml
│  │      yolov3-spp-obb.yaml
│  │      yolov3-tiny-obb.yaml
│  │      
│  ├─Pose
│  │      yolov3-pose.yaml
│  │      yolov3-spp-pose.yaml
│  │      yolov3-tiny-pose.yaml
│  │      
│  └─Segment
│          yolov3-seg.yaml
│          yolov3-spp-seg.yaml
│          yolov3-tiny-seg.yaml
│          
├─v4
│  ├─Classify
│  │      yolov4-csp-cls.yaml
│  │      yolov4-mish-cls.yaml
│  │      yolov4-p6-cls.yaml
│  │      
│  ├─Detect
│  │      yolov4-csp-rtdetr.yaml
│  │      yolov4-csp.yaml
│  │      yolov4-mish-rtdetr.yaml
│  │      yolov4-mish.yaml
│  │      yolov4-p5.yaml
│  │      yolov4-p6.yaml
│  │      yolov4-p7.yaml
│  │      
│  ├─OBB
│  │      yolov4-csp-obb.yaml
│  │      yolov4-mish-obb.yaml
│  │      yolov4-obb-p5.yaml
│  │      yolov4-obb-p6.yaml
│  │      yolov4-obb-p7.yaml
│  │      
│  ├─Pose
│  │      yolov4-csp-pose.yaml
│  │      yolov4-mish-pose.yaml
│  │      yolov4-pose-p5.yaml
│  │      yolov4-pose-p6.yaml
│  │      yolov4-pose-p7.yaml
│  │      
│  └─Segment
│          yolov4-csp-seg.yaml
│          yolov4-mish-seg.yaml
│          yolov4-seg-p5.yaml
│          yolov4-seg-p6.yaml
│          yolov4-seg-p7.yaml
│          
├─v5
│  ├─Classify
│  │      yolov5-AIFI-cls.yaml
│  │      yolov5-cls.yaml
│  │      yolov5-p6.yaml
│  │      yolov5-PPLCNet-cls.yaml
│  │      yolov5-RepVGG-cls.yaml
│  │      
│  ├─Detect
│  │      yolov5-AIFI.yaml
│  │      yolov5-AKConv.yaml
│  │      yolov5-BoT3.yaml
│  │      yolov5-CAConv.yaml
│  │      yolov5-CARAFE.yaml
│  │      yolov5-CCFM.yaml
│  │      yolov5-CNeB-neck.yaml
│  │      yolov5-CoordAtt.yaml
│  │      yolov5-CPCA.yaml
│  │      yolov5-CrissCrossAttention.yaml
│  │      yolov5-D-LKAAttention.yaml
│  │      yolov5-DAttention.yaml
│  │      yolov5-DCNv2.yaml
│  │      yolov5-deconv.yaml
│  │      yolov5-Dyample.yaml
│  │      yolov5-ECAAttention.yaml
│  │      yolov5-EffectiveSE.yaml
│  │      yolov5-GAMAttention.yaml
│  │      yolov5-goldyolo.yaml
│  │      yolov5-hornet-backbone.yaml
│  │      yolov5-hornet-neck.yaml
│  │      yolov5-l-mobilenetv3s.yaml
│  │      yolov5-LeakyReLU.yaml
│  │      yolov5-Lite-c.yaml
│  │      yolov5-Lite-e.yaml
│  │      yolov5-Lite-g.yaml
│  │      yolov5-Lite-s.yaml
│  │      yolov5-mobile3s.yaml
│  │      yolov5-mobileone-backbone.yaml
│  │      yolov5-MobileOne-Lite-g.yaml
│  │      yolov5-MobileOne.yaml
│  │      yolov5-mobilev3l.yaml
│  │      yolov5-ODConvNext.yaml
│  │      yolov5-old-p6.yaml
│  │      yolov5-old.yaml
│  │      yolov5-p2.yaml
│  │      yolov5-p34.yaml
│  │      yolov5-p6.yaml
│  │      yolov5-p7.yaml
│  │      yolov5-PPLCNet.yaml
│  │      yolov5-RepVGG-A1-backbone.yaml
│  │      yolov5-RepVGG.yaml
│  │      yolov5-rtdetr.yaml
│  │      yolov5-scal-zoom.yaml
│  │      yolov5-SEAttention.yaml
│  │      yolov5-SegNextAttention.yaml
│  │      yolov5-ShuffleAttention.yaml
│  │      yolov5-Shufflenetv2.yaml
│  │      yolov5-SimSPPF.yaml
│  │      yolov5-SKAttention.yaml
│  │      yolov5-SPPCSPC.yaml
│  │      yolov5-transformer.yaml
│  │      yolov5-TripletAttention.yaml
│  │      yolov5-VanillaNet.yaml
│  │      yolov5.yaml
│  │      
│  ├─OBB
│  │      yolov5-AIFI-obb.yaml
│  │      yolov5-AKConv-obb.yaml
│  │      yolov5-CAConv-obb.yaml
│  │      yolov5-CCFM-obb.yaml
│  │      yolov5-CNeB-neck-obb.yaml
│  │      yolov5-CoordAtt-obb.yaml
│  │      yolov5-CPCA-obb.yaml
│  │      yolov5-CrissCrossAttention-obb.yaml
│  │      yolov5-D-LKAAttention-obb.yaml
│  │      yolov5-DAttention-obb.yaml
│  │      yolov5-DCNv2-obb.yaml
│  │      yolov5-deconv-obb.yaml
│  │      yolov5-ECAAttention-obb.yaml
│  │      yolov5-EffectiveSE-obb.yaml
│  │      yolov5-GAMAttention-obb.yaml
│  │      yolov5-goldyolo-obb.yaml
│  │      yolov5-hornet-backbone-obb.yaml
│  │      yolov5-hornet-neck-obb.yaml
│  │      yolov5-l-mobilenetv3s-obb.yaml
│  │      yolov5-LeakyReLU-obb.yaml
│  │      yolov5-Lite-c-obb.yaml
│  │      yolov5-Lite-e-obb.yaml
│  │      yolov5-Lite-g-obb.yaml
│  │      yolov5-Lite-s-obb.yaml
│  │      yolov5-mobile3s-obb.yaml
│  │      yolov5-mobileone-backbone-obb.yaml
│  │      yolov5-mobilev3l-obb.yaml
│  │      yolov5-obb-p2.yaml
│  │      yolov5-obb-p34.yaml
│  │      yolov5-obb-p6.yaml
│  │      yolov5-obb-p7.yaml
│  │      yolov5-obb.yaml
│  │      yolov5-PPLCNet-obb.yaml
│  │      yolov5-RepVGG-A1-backbone-obb.yaml
│  │      yolov5-RepVGG-obb.yaml
│  │      yolov5-SEAttention-obb.yaml
│  │      yolov5-SegNextAttention-obb.yaml
│  │      yolov5-ShuffleAttention-obb.yaml
│  │      yolov5-Shufflenetv2-obb.yaml
│  │      yolov5-SimSPPF-obb.yaml
│  │      yolov5-SKAttention-obb.yaml
│  │      yolov5-SPPCSPC-obb.yaml
│  │      yolov5-transformer-obb.yaml
│  │      yolov5-TripletAttention-obb.yaml
│  │      
│  ├─Pose
│  │      yolov5-AIFI-pose.yaml
│  │      yolov5-AKConv-pose.yaml
│  │      yolov5-boost-pose.yaml
│  │      yolov5-CAConv-pose.yaml
│  │      yolov5-CCFM-pose.yaml
│  │      yolov5-CNeB-neck-pose.yaml
│  │      yolov5-CoordAtt-pose.yaml
│  │      yolov5-CPCA-pose.yaml
│  │      yolov5-CrissCrossAttention-pose.yaml
│  │      yolov5-D-LKAAttention-pose.yaml
│  │      yolov5-DAttention-pose.yaml
│  │      yolov5-DCNv2-pose.yaml
│  │      yolov5-deconv-pose.yaml
│  │      yolov5-ECAAttention-pose.yaml
│  │      yolov5-EffectiveSE-pose.yaml
│  │      yolov5-GAMAttention-pose.yaml
│  │      yolov5-goldyolo-pose.yaml
│  │      yolov5-hornet-backbone-pose.yaml
│  │      yolov5-hornet-neck-pose.yaml
│  │      yolov5-l-mobilenetv3s-pose.yaml
│  │      yolov5-LeakyReLU-obb.yaml
│  │      yolov5-Lite-c-pose.yaml
│  │      yolov5-Lite-e-pose.yaml
│  │      yolov5-Lite-g-pose.yaml
│  │      yolov5-Lite-s-pose.yaml
│  │      yolov5-mobile3s-pose.yaml
│  │      yolov5-mobileone-backbone-pose.yaml
│  │      yolov5-mobilev3l-pose.yaml
│  │      yolov5-old-pose-p6.yaml
│  │      yolov5-pose-p2.yaml
│  │      yolov5-pose-p34.yaml
│  │      yolov5-pose-p6.yaml
│  │      yolov5-pose-p7.yaml
│  │      yolov5-pose.yaml
│  │      yolov5-PPLCNet-pose.yaml
│  │      yolov5-RepVGG-A1-backbone-pose.yaml
│  │      yolov5-RepVGG-pose.yaml
│  │      yolov5-SEAttention-pose.yaml
│  │      yolov5-SegNextAttention-pose.yaml
│  │      yolov5-ShuffleAttention-pose.yaml
│  │      yolov5-Shufflenetv2-pose.yaml
│  │      yolov5-SimSPPF-pose.yaml
│  │      yolov5-SKAttention-pose.yaml
│  │      yolov5-SPPCSPC-pose.yaml
│  │      yolov5-transformer-pose.yaml
│  │      yolov5-TripletAttention-pose.yaml
│  │      
│  └─Segment
│          yolov5-AIFI-seg.yaml
│          yolov5-AKConv-seg.yaml
│          yolov5-BoT3-seg.yaml
│          yolov5-CAConv-seg.yaml
│          yolov5-CARAFE-seg.yaml
│          yolov5-CCFM-seg.yaml
│          yolov5-CNeB-neck-seg.yaml
│          yolov5-CoordAtt-seg.yaml
│          yolov5-CPCA-seg.yaml
│          yolov5-CrissCrossAttention-seg.yaml
│          yolov5-D-LKAAttention-seg.yaml
│          yolov5-DAttention-seg.yaml
│          yolov5-DCNv2-seg.yaml
│          yolov5-deconv-seg.yaml
│          yolov5-Dyample-seg.yaml
│          yolov5-ECAAttention-seg.yaml
│          yolov5-EffectiveSE-seg.yaml
│          yolov5-GAMAttention-seg.yaml
│          yolov5-goldyolo-seg.yaml
│          yolov5-hornet-backbone-seg.yaml
│          yolov5-hornet-neck-seg.yaml
│          yolov5-l-mobilenetv3s-seg.yaml
│          yolov5-LeakyReLU.yaml
│          yolov5-Lite-c-seg.yaml
│          yolov5-Lite-e-seg.yaml
│          yolov5-Lite-g-seg.yaml
│          yolov5-Lite-s-seg.yaml
│          yolov5-mobile3s-seg.yaml
│          yolov5-mobileone-backbone-seg.yaml
│          yolov5-MobileOne-Lite-g-seg.yaml
│          yolov5-MobileOne-seg.yaml
│          yolov5-mobilev3l-seg.yaml
│          yolov5-PPLCNet-seg.yaml
│          yolov5-RepVGG-A1-backbone-seg.yaml
│          yolov5-RepVGG-seg.yaml
│          yolov5-SEAttention-seg.yaml
│          yolov5-seg-p2.yaml
│          yolov5-seg-p34.yaml
│          yolov5-seg-p6.yaml
│          yolov5-seg-p7.yaml
│          yolov5-seg.yaml
│          yolov5-SegNextAttention-seg.yaml
│          yolov5-ShuffleAttention-seg.yaml
│          yolov5-Shufflenetv2-seg.yaml
│          yolov5-SimSPPF-seg.yaml
│          yolov5-SKAttention-seg.yaml
│          yolov5-SPPCSPC-seg.yaml
│          yolov5-transformer-seg.yaml
│          yolov5-Triplet-D-LKAAttention-seg.yaml
│          yolov5-TripletAttention-seg.yaml
│          
├─v6
│  ├─Classify
│  │      yolov6-3.0-cls-p6.yaml
│  │      yolov6-3.0-cls.yaml
│  │      yolov6-4.0-cls-p6.yaml
│  │      yolov6-4.0-cls.yaml
│  │      
│  ├─Detect
│  │      yolov6-3.0-p2.yaml
│  │      yolov6-3.0-p34.yaml
│  │      yolov6-3.0-p6.yaml
│  │      yolov6-3.0-p7.yaml
│  │      yolov6-3.0-rtdetr.yaml
│  │      yolov6-3.0.yaml
│  │      yolov6-4.0-CPCA.yaml
│  │      yolov6-4.0-CrissCrossAttention.yaml
│  │      yolov6-4.0-D-LKAAttention.yaml
│  │      yolov6-4.0-DAttention.yaml
│  │      yolov6-4.0-GAMAttention.yaml
│  │      yolov6-4.0-p2.yaml
│  │      yolov6-4.0-p34.yaml
│  │      yolov6-4.0-p6.yaml
│  │      yolov6-4.0-p7.yaml
│  │      yolov6-4.0-rtdetr.yaml
│  │      yolov6-4.0-SEAttention.yaml
│  │      yolov6-4.0-SegNextAttention.yaml
│  │      yolov6-4.0-ShuffleAttention.yaml
│  │      yolov6-4.0-SKAttention.yaml
│  │      yolov6-4.0-TripletAttention.yaml
│  │      yolov6-4.0.yaml
│  │      yolov6.yaml
│  │      
│  ├─OBB
│  │      yolov6-3.0-obb-p2.yaml
│  │      yolov6-3.0-obb-p34.yaml
│  │      yolov6-3.0-obb-p6.yaml
│  │      yolov6-3.0-obb-p7.yaml
│  │      yolov6-3.0-obb.yaml
│  │      yolov6-4.0-CPCA-obb.yaml
│  │      yolov6-4.0-CrissCrossAttention-obb.yaml
│  │      yolov6-4.0-D-LKAAttention-obb.yaml
│  │      yolov6-4.0-DAttention-obb.yaml
│  │      yolov6-4.0-GAMAttention-obb.yaml
│  │      yolov6-4.0-obb-p2.yaml
│  │      yolov6-4.0-obb-p34.yaml
│  │      yolov6-4.0-obb-p6.yaml
│  │      yolov6-4.0-obb-p7.yaml
│  │      yolov6-4.0-obb.yaml
│  │      yolov6-4.0-SEAttention-obb.yaml
│  │      yolov6-4.0-SegNextAttention-obb.yaml
│  │      yolov6-4.0-ShuffleAttention-obb.yaml
│  │      yolov6-4.0-SKAttention-obb.yaml
│  │      yolov6-4.0-TripletAttention-obb.yaml
│  │      
│  ├─Pose
│  │      yolov6-3.0-pose-p2.yaml
│  │      yolov6-3.0-pose-p34.yaml
│  │      yolov6-3.0-pose-p6.yaml
│  │      yolov6-3.0-pose-p7.yaml
│  │      yolov6-3.0-pose.yaml
│  │      yolov6-4.0-CPCA-pose.yaml
│  │      yolov6-4.0-CrissCrossAttention-pose.yaml
│  │      yolov6-4.0-D-LKAAttention-pose.yaml
│  │      yolov6-4.0-DAttention-pose.yaml
│  │      yolov6-4.0-GAMAttention-pose.yaml
│  │      yolov6-4.0-pose-p2.yaml
│  │      yolov6-4.0-pose-p34.yaml
│  │      yolov6-4.0-pose-p6.yaml
│  │      yolov6-4.0-pose-p7.yaml
│  │      yolov6-4.0-pose.yaml
│  │      yolov6-4.0-SEAttention-pose.yaml
│  │      yolov6-4.0-SegNextAttention-pose.yaml
│  │      yolov6-4.0-ShuffleAttention-pose.yaml
│  │      yolov6-4.0-SKAttention-pose.yaml
│  │      yolov6-4.0-TripletAttention-pose.yaml
│  │      
│  └─Segment
│          yolov6-3.0-seg-p2.yaml
│          yolov6-3.0-seg-p34.yaml
│          yolov6-3.0-seg-p6.yaml
│          yolov6-3.0-seg-p7.yaml
│          yolov6-3.0-seg.yaml
│          yolov6-4.0-CPCA-seg.yaml
│          yolov6-4.0-CrissCrossAttention-seg.yaml
│          yolov6-4.0-D-LKAAttention-seg.yaml
│          yolov6-4.0-DAttention-seg.yaml
│          yolov6-4.0-GAMAttention.yaml
│          yolov6-4.0-SEAttention-seg.yaml
│          yolov6-4.0-seg-p2.yaml
│          yolov6-4.0-seg-p34.yaml
│          yolov6-4.0-seg-p6.yaml
│          yolov6-4.0-seg-p7.yaml
│          yolov6-4.0-seg.yaml
│          yolov6-4.0-SegNextAttention-seg.yaml
│          yolov6-4.0-ShuffleAttention-seg.yaml
│          yolov6-4.0-SKAttention-seg.yaml
│          yolov6-4.0-TripletAttention-seg.yaml
│          
├─v7
│  ├─Classify
│  │      yolov7-cls-d6.yaml
│  │      yolov7-cls-e6.yaml
│  │      yolov7-cls-e6e.yaml
│  │      yolov7-cls-w6.yaml
│  │      yolov7-cls.yaml
│  │      yolov7-DCNv2-cls.yaml
│  │      yolov7-swin-cls.yaml
│  │      yolov7-tiny-cls.yaml
│  │      yolov7-x-cls.yaml
│  │      
│  ├─Detect
│  │  │  yolov7-af-i.yaml
│  │  │  yolov7-af.yaml
│  │  │  yolov7-C3C2-CPCA.yaml
│  │  │  yolov7-C3C2-CrissCrossAttention.yaml
│  │  │  yolov7-C3C2-GAMAttention.yaml
│  │  │  yolov7-C3C2-RepVGG.yaml
│  │  │  yolov7-C3C2-ResNet.yaml
│  │  │  yolov7-C3C2-SegNextAttention.yaml
│  │  │  yolov7-C3C2.yaml
│  │  │  yolov7-d6.yaml
│  │  │  yolov7-DCNv2.yaml
│  │  │  yolov7-e6.yaml
│  │  │  yolov7-e6e.yaml
│  │  │  yolov7-goldyolo-simple.yaml
│  │  │  yolov7-goldyolo.yaml
│  │  │  yolov7-MobileOne.yaml
│  │  │  yolov7-RepNCSPELAN.yaml
│  │  │  yolov7-rtdetr.yaml
│  │  │  yolov7-simple.yaml
│  │  │  yolov7-tiny-AKConv.yaml
│  │  │  yolov7-tiny-goldyolo-simple.yaml
│  │  │  yolov7-tiny-goldyolo.yaml
│  │  │  yolov7-tiny-MobileNetv3.yaml
│  │  │  yolov7-tiny-MobileOne.yaml
│  │  │  yolov7-tiny-PPLCNet.yaml
│  │  │  yolov7-tiny-RepNCSPELAN.yaml
│  │  │  yolov7-tiny-rtdetr.yaml
│  │  │  yolov7-tiny-SiLU.yaml
│  │  │  yolov7-tiny-simple.yaml
│  │  │  yolov7-tiny.yaml
│  │  │  yolov7-w6.yaml
│  │  │  yolov7-x-rtdetr.yaml
│  │  │  yolov7-x.yaml
│  │  │  yolov7.yaml
│  │  │  
│  │  ├─deploy
│  │  │      yolov7-d6.yaml
│  │  │      yolov7-e6.yaml
│  │  │      yolov7-e6e.yaml
│  │  │      yolov7-tiny-silu.yaml
│  │  │      yolov7-tiny.yaml
│  │  │      yolov7-w6.yaml
│  │  │      yolov7-x.yaml
│  │  │      yolov7.yaml
│  │  │      
│  │  └─u6
│  │          yolov7-C3C2-CPCA-u6.yaml
│  │          yolov7-C3C2-CrissCrossAttention-u6.yaml
│  │          yolov7-C3C2-GAMAttention-u6.yaml
│  │          yolov7-C3C2-RepVGG-u6.yaml
│  │          yolov7-C3C2-ResNet-u6.yaml
│  │          yolov7-C3C2-SegNextAttention-u6.yaml
│  │          yolov7-C3C2-u6.yaml
│  │          yolov7-DCNv2-u6.yaml
│  │          yolov7-goldyolo-u6.yaml
│  │          yolov7-MobileOne-u6.yaml
│  │          yolov7-RepNCSPELAN-u6.yaml
│  │          yolov7-rtdetr-u6.yaml
│  │          yolov7-u6.yaml
│  │          
│  ├─OBB
│  │  │  yolov7-af-Iobb.yaml
│  │  │  yolov7-af-obb.yaml
│  │  │  yolov7-C3C2-CPCA-obb.yaml
│  │  │  yolov7-C3C2-CrissCrossAttention-obb.yaml
│  │  │  yolov7-C3C2-GAMAttention-obb.yaml
│  │  │  yolov7-C3C2-obb.yaml
│  │  │  yolov7-C3C2-RepVGG-obb.yaml
│  │  │  yolov7-C3C2-ResNet-obb.yaml
│  │  │  yolov7-C3C2-SegNextAttention-obb.yaml
│  │  │  yolov7-DCNv2-obb.yaml
│  │  │  yolov7-goldyolo-obb.yaml
│  │  │  yolov7-MobileOne-obb.yaml
│  │  │  yolov7-obb-d6.yaml
│  │  │  yolov7-obb-e6.yaml
│  │  │  yolov7-obb-e6e.yaml
│  │  │  yolov7-obb-w6.yaml
│  │  │  yolov7-obb.yaml
│  │  │  yolov7-RepNCSPELAN-obb.yaml
│  │  │  yolov7-simple-obb.yaml
│  │  │  yolov7-tiny-AKConv-obb.yaml
│  │  │  yolov7-tiny-goldyolo-obb.yaml
│  │  │  yolov7-tiny-MobileNetv3-obb.yaml
│  │  │  yolov7-tiny-MobileOne-obb.yaml
│  │  │  yolov7-tiny-obb.yaml
│  │  │  yolov7-tiny-PPLCNet-obb.yaml
│  │  │  yolov7-tiny-RepNCSPELAN-obb.yaml
│  │  │  yolov7-tiny-SiLU-obb.yaml
│  │  │  yolov7-x-obb.yaml
│  │  │  
│  │  ├─deploy
│  │  │      yolov7-obb-d6.yaml
│  │  │      yolov7-obb-e6.yaml
│  │  │      yolov7-obb-e6e.yaml
│  │  │      yolov7-obb-w6.yaml
│  │  │      yolov7-obb.yaml
│  │  │      yolov7-tiny-obb.yaml
│  │  │      yolov7-tiny-silu-obb.yaml
│  │  │      yolov7-x-obb.yaml
│  │  │      
│  │  └─u6
│  │          yolov7-C3C2-CPCA-obb-u6.yaml
│  │          yolov7-C3C2-CrissCrossAttention-obb-u6.yaml
│  │          yolov7-C3C2-GAMAttention-obb-u6.yaml
│  │          yolov7-C3C2-obb-u6.yaml
│  │          yolov7-C3C2-RepVGG-obb-u6.yaml
│  │          yolov7-C3C2-ResNet-obb-u6.yaml
│  │          yolov7-C3C2-SegNextAttention-obb-u6.yaml
│  │          yolov7-DCNv2-obb-u6.yaml
│  │          yolov7-goldyolo-obb-u6.yaml
│  │          yolov7-MobileOne-obb-u6.yaml
│  │          yolov7-obb-u6.yaml
│  │          
│  ├─Pose
│  │  │  yolov7-af-ipose.yaml
│  │  │  yolov7-af-pose.yaml
│  │  │  yolov7-C3C2-CPCA-pose.yaml
│  │  │  yolov7-C3C2-CrissCrossAttention-pose.yaml
│  │  │  yolov7-C3C2-GAMAttention-pose.yaml
│  │  │  yolov7-C3C2-pose.yaml
│  │  │  yolov7-C3C2-RepVGG-pose.yaml
│  │  │  yolov7-C3C2-ResNet-pose.yaml
│  │  │  yolov7-C3C2-SegNextAttention-pose.yaml
│  │  │  yolov7-DCNv2-pose.yaml
│  │  │  yolov7-goldyolo-pose.yaml
│  │  │  yolov7-MobileOne-pose.yaml
│  │  │  yolov7-pose-d6.yaml
│  │  │  yolov7-pose-e6.yaml
│  │  │  yolov7-pose-e6e.yaml
│  │  │  yolov7-pose-w6.yaml
│  │  │  yolov7-pose.yaml
│  │  │  yolov7-RepNCSPELAN-pose.yaml
│  │  │  yolov7-simple-pose.yaml
│  │  │  yolov7-tiny-AKConv-pose.yaml
│  │  │  yolov7-tiny-goldyolo-pose.yaml
│  │  │  yolov7-tiny-MobileNetv3-pose.yaml
│  │  │  yolov7-tiny-MobileOne-pose.yaml
│  │  │  yolov7-tiny-pose.yaml
│  │  │  yolov7-tiny-PPLCNet-pose.yaml
│  │  │  yolov7-tiny-RepNCSPELAN-pose.yaml
│  │  │  yolov7-tiny-SiLU-pose.yaml
│  │  │  yolov7-x-pose.yaml
│  │  │  
│  │  ├─deploy
│  │  │      yolov7-pose-d6.yaml
│  │  │      yolov7-pose-e6.yaml
│  │  │      yolov7-pose-e6e.yaml
│  │  │      yolov7-pose-w6.yaml
│  │  │      yolov7-pose.yaml
│  │  │      yolov7-tiny-pose.yaml
│  │  │      yolov7-tiny-silu-pose.yaml
│  │  │      yolov7-x-pose.yaml
│  │  │      
│  │  └─u6
│  │          yolov7-C3C2-CPCA-pose-u6.yaml
│  │          yolov7-C3C2-CrissCrossAttention-pose-u6.yaml
│  │          yolov7-C3C2-GAMAttention-pose-u6.yaml
│  │          yolov7-C3C2-pose-u6.yaml
│  │          yolov7-C3C2-RepVGG-pose-u6.yaml
│  │          yolov7-C3C2-ResNet-pose-u6.yaml
│  │          yolov7-C3C2-SegNextAttention-pose-u6.yaml
│  │          yolov7-DCNv2-pose-u6.yaml
│  │          yolov7-goldyolo-pose-u6.yaml
│  │          yolov7-MobileOne-pose-u6.yaml
│  │          yolov7-pose-u6.yaml
│  │          
│  └─Segment
│      │  yolov7-af-iseg.yaml
│      │  yolov7-af-seg.yaml
│      │  yolov7-C3C2-CPCA-seg.yaml
│      │  yolov7-C3C2-CrissCrossAttention-seg.yaml
│      │  yolov7-C3C2-GAMAttention-seg.yaml
│      │  yolov7-C3C2-RepVGG-seg.yaml
│      │  yolov7-C3C2-ResNet-seg.yaml
│      │  yolov7-C3C2-seg.yaml
│      │  yolov7-C3C2-SegNextAttention-seg.yaml
│      │  yolov7-DCNv2-seg.yaml
│      │  yolov7-goldyolo-seg.yaml
│      │  yolov7-goldyolo-simple-seg.yaml
│      │  yolov7-MobileOne-seg.yaml
│      │  yolov7-RepNCSPELAN-seg.yaml
│      │  yolov7-seg-d6.yaml
│      │  yolov7-seg-e6.yaml
│      │  yolov7-seg-e6e.yaml
│      │  yolov7-seg-w6.yaml
│      │  yolov7-seg.yaml
│      │  yolov7-simple-seg.yaml
│      │  yolov7-tiny-AKConv-seg.yaml
│      │  yolov7-tiny-goldyolo-seg.yaml
│      │  yolov7-tiny-goldyolo-simple-seg.yaml
│      │  yolov7-tiny-MobileNetv3-seg.yaml
│      │  yolov7-tiny-MobileOne-seg.yaml
│      │  yolov7-tiny-PPLCNet-seg.yaml
│      │  yolov7-tiny-RepNCSPELAN-seg.yaml
│      │  yolov7-tiny-seg.yaml
│      │  yolov7-tiny-SiLU-seg.yaml
│      │  yolov7-tiny-simple-seg.yaml
│      │  yolov7-x-seg.yaml
│      │  
│      ├─deploy
│      │      yolov7-seg-d6.yaml
│      │      yolov7-seg-e6.yaml
│      │      yolov7-seg-e6e.yaml
│      │      yolov7-seg-w6.yaml
│      │      yolov7-seg.yaml
│      │      yolov7-tiny-seg.yaml
│      │      yolov7-tiny-silu-seg.yaml
│      │      yolov7-x-seg.yaml
│      │      
│      └─u6
│              yolov7-C3C2-CPCA-seg-u6.yaml
│              yolov7-C3C2-CrissCrossAttention-seg-u6.yaml
│              yolov7-C3C2-GAMAttention-seg-u6.yaml
│              yolov7-C3C2-RepVGG-seg-u6.yaml
│              yolov7-C3C2-ResNet-seg-u6.yaml
│              yolov7-C3C2-seg-u6.yaml
│              yolov7-C3C2-SegNextAttention-seg-u6.yaml
│              yolov7-DCNv2-seg-u6.yaml
│              yolov7-goldyolo-seg-u6.yaml
│              yolov7-MobileOne-seg-u6.yaml
│              yolov7-seg-u6.yaml
│              
├─v8
│  ├─Classify
│  │      yolov8-AIFI-cls.yaml
│  │      yolov8-cls-p2.yaml
│  │      yolov8-cls-p6.yaml
│  │      yolov8-cls.yaml
│  │      yolov8-RepVGG-cls.yaml
│  │      
│  ├─Detect
│  │      yoloe-v8.yaml
│  │      yolov8-AIFI.yaml
│  │      yolov8-AKConv.yaml
│  │      yolov8-BoT3.yaml
│  │      yolov8-C2f-DAttention.yaml
│  │      yolov8-C2f-DRB.yaml
│  │      yolov8-C2f-EMBC.yaml
│  │      yolov8-C2f-EMSC.yaml
│  │      yolov8-C2f-EMSCP.yaml
│  │      yolov8-C2f-FasterBlock.yaml
│  │      yolov8-C2f-GhostModule-DynamicConv.yaml
│  │      yolov8-C2f-MSBlockv2.yaml
│  │      yolov8-C2f-OREPA.yaml
│  │      yolov8-C2f-REPVGGOREPA.yaml
│  │      yolov8-C2f-RetBlock.yaml
│  │      yolov8-C2f-RVB-EMA.yaml
│  │      yolov8-C2f-RVB.yaml
│  │      yolov8-C2f-Star-CAA.yaml
│  │      yolov8-C2f-StarNet.yaml
│  │      yolov8-C2f-UniRepLKNetBlock.yaml
│  │      yolov8-CAConv.yaml
│  │      yolov8-CNeB-neck.yaml
│  │      yolov8-CoordAtt.yaml
│  │      yolov8-CPAarch.yaml
│  │      yolov8-CPCA.yaml
│  │      yolov8-CrissCrossAttention.yaml
│  │      yolov8-D-LKAAttention.yaml
│  │      yolov8-DAttention.yaml
│  │      yolov8-DCNv2.yaml
│  │      yolov8-deconv.yaml
│  │      yolov8-DiT-C2f-UIB-FMDI.yaml
│  │      yolov8-ECAAttention.yaml
│  │      yolov8-EffectiveSE.yaml
│  │      yolov8-Faster-Block-CGLU.yaml
│  │      yolov8-Faster-EMA.yaml
│  │      yolov8-GAMAttention.yaml
│  │      yolov8-goldyolo.yaml
│  │      yolov8-hornet-backbone.yaml
│  │      yolov8-hornet-neck.yaml
│  │      yolov8-HWD.yaml
│  │      yolov8-l-mobilenetv3s.yaml
│  │      yolov8-LCDConv.yaml
│  │      yolov8-LeakyReLU.yaml
│  │      yolov8-Lite-c.yaml
│  │      yolov8-Lite-g.yaml
│  │      yolov8-Lite-s.yaml
│  │      yolov8-MHSA.yaml
│  │      yolov8-mobile3s.yaml
│  │      yolov8-mobileone-backbone.yaml
│  │      yolov8-MobileOne.yaml
│  │      yolov8-mobilev3l.yaml
│  │      yolov8-MSFM.yaml
│  │      yolov8-ODConvNext.yaml
│  │      yolov8-p2.yaml
│  │      yolov8-p34.yaml
│  │      yolov8-p6.yaml
│  │      yolov8-p7.yaml
│  │      yolov8-PPLCNet.yaml
│  │      yolov8-RepNCSPELAN.yaml
│  │      yolov8-RepVGG-A1-backbone.yaml
│  │      yolov8-RepVGG.yaml
│  │      yolov8-RepViTBlock.yaml
│  │      yolov8-rtdetr.yaml
│  │      yolov8-SEAttention.yaml
│  │      yolov8-SegNextAttention.yaml
│  │      yolov8-ShuffleAttention.yaml
│  │      yolov8-Shufflenetv2.yaml
│  │      yolov8-SimAM.yaml
│  │      yolov8-SimSPPF.yaml
│  │      yolov8-SKAttention.yaml
│  │      yolov8-SPDConv.yaml
│  │      yolov8-SPPCSPC.yaml
│  │      yolov8-StripNet-sn2.yaml
│  │      yolov8-SwinTransformer.yaml
│  │      yolov8-TripletAttention.yaml
│  │      yolov8-VanillaNet.yaml
│  │      yolov8.yaml
│  │      
│  ├─OBB
│  │      yolov8-AIFI-obb.yaml
│  │      yolov8-AKConv-obb.yaml
│  │      yolov8-CAConv-obb.yaml
│  │      yolov8-CNeB-neck-obb.yaml
│  │      yolov8-CPCA-obb.yaml
│  │      yolov8-CrissCrossAttention-obb.yaml
│  │      yolov8-D-LKAAttention-obb.yaml
│  │      yolov8-DAttention-obb.yaml
│  │      yolov8-DCNv2-obb.yaml
│  │      yolov8-deconv-obb.yaml
│  │      yolov8-ECAAttention-obb.yaml
│  │      yolov8-EffectiveSE-obb.yaml
│  │      yolov8-GAMAttention-obb.yaml
│  │      yolov8-goldyolo-obb.yaml
│  │      yolov8-hornet-backbone-obb.yaml
│  │      yolov8-hornet-neck-obb.yaml
│  │      yolov8-l-mobilenetv3s-obb.yaml
│  │      yolov8-LeakyReLU-obb.yaml
│  │      yolov8-Lite-c-obb.yaml
│  │      yolov8-Lite-g-obb.yaml
│  │      yolov8-Lite-s-obb.yaml
│  │      yolov8-mobile3s-obb.yaml
│  │      yolov8-mobileone-backbone-obb.yaml
│  │      yolov8-MobileOne-obb.yaml
│  │      yolov8-mobilev3l-obb.yaml
│  │      yolov8-obb-p2.yaml
│  │      yolov8-obb-p34.yaml
│  │      yolov8-obb-p6.yaml
│  │      yolov8-obb-p7.yaml
│  │      yolov8-obb.yaml
│  │      yolov8-PPLCNet-obb.yaml
│  │      yolov8-RepNCSPELAN-obb.yaml
│  │      yolov8-RepVGG-A1-backbone-obb.yaml
│  │      yolov8-RepVGG-obb.yaml
│  │      yolov8-SEAttention-obb.yaml
│  │      yolov8-SegNextAttention-obb.yaml
│  │      yolov8-ShuffleAttention-obb.yaml
│  │      yolov8-Shufflenetv2-obb.yaml
│  │      yolov8-SimAM-obb.yaml
│  │      yolov8-SimSPPF-obb.yaml
│  │      yolov8-SKAttention-obb.yaml
│  │      yolov8-SPPCSPC-obb.yaml
│  │      yolov8-TripletAttention-obb.yaml
│  │      
│  ├─Pose
│  │      yolov8-AIFI-pose.yaml
│  │      yolov8-AKConv-pose.yaml
│  │      yolov8-CAConv-pose.yaml
│  │      yolov8-CNeB-neck-pose.yaml
│  │      yolov8-CoordAtt-pose.yaml
│  │      yolov8-CPCA-pose.yaml
│  │      yolov8-CrissCrossAttention-pose.yaml
│  │      yolov8-D-LKAAttention-pose.yaml
│  │      yolov8-DAttention-pose.yaml
│  │      yolov8-DCNv2-pose.yaml
│  │      yolov8-deconv-pose.yaml
│  │      yolov8-ECAAttention-pose.yaml
│  │      yolov8-EffectiveSE-pose.yaml
│  │      yolov8-GAMAttention-pose.yaml
│  │      yolov8-goldyolo-pose.yaml
│  │      yolov8-hornet-backbone-obb.yaml
│  │      yolov8-hornet-neck-obb.yaml
│  │      yolov8-l-mobilenetv3s-pose.yaml
│  │      yolov8-LeakyReLU-pose.yaml
│  │      yolov8-Lite-c-pose.yaml
│  │      yolov8-Lite-g-pose.yaml
│  │      yolov8-Lite-s-pose.yaml
│  │      yolov8-mobile3s-pose.yaml
│  │      yolov8-mobileone-backbone-pose.yaml
│  │      yolov8-MobileOne-pose.yaml
│  │      yolov8-mobilev3l-pose.yaml
│  │      yolov8-pose-p2.yaml
│  │      yolov8-pose-p34.yaml
│  │      yolov8-pose-p6.yaml
│  │      yolov8-pose-p7.yaml
│  │      yolov8-pose.yaml
│  │      yolov8-PPLCNet-pose.yaml
│  │      yolov8-RepNCSPELAN.yaml
│  │      yolov8-RepVGG-A1-backbone-obb.yaml
│  │      yolov8-RepVGG-pose.yaml
│  │      yolov8-SEAttention-pose.yaml
│  │      yolov8-SegNextAttention-pose.yaml
│  │      yolov8-ShuffleAttention-pose.yaml
│  │      yolov8-Shufflenetv2-pose.yaml
│  │      yolov8-SimAM-pose.yaml
│  │      yolov8-SimSPPF-pose.yaml
│  │      yolov8-SKAttention-pose.yaml
│  │      yolov8-SPPCSPC-pose.yaml
│  │      yolov8-TripletAttention-pose.yaml
│  │      
│  └─Segment
│          yoloe-v8-seg.yaml
│          yolov8-AIFI-seg.yaml
│          yolov8-AKConv-seg.yaml
│          yolov8-BoT3-seg.yaml
│          yolov8-CAConv-seg.yaml
│          yolov8-CNeB-neck-seg.yaml
│          yolov8-CPCA-seg.yaml
│          yolov8-CrissCrossAttention-seg.yaml
│          yolov8-D-LKAAttention-seg.yaml
│          yolov8-DAttention-seg.yaml
│          yolov8-DCNv2-seg.yaml
│          yolov8-deconv-seg.yaml
│          yolov8-ECAAttention-seg.yaml
│          yolov8-EffectiveSE-seg.yaml
│          yolov8-GAMAttention-seg.yaml
│          yolov8-goldyolo-seg.yaml
│          yolov8-hornet-backbone-seg.yaml
│          yolov8-hornet-neck-seg.yaml
│          yolov8-l-mobilenetv3s-seg.yaml
│          yolov8-LeakyReLU-seg.yaml
│          yolov8-Lite-c-seg.yaml
│          yolov8-Lite-g-seg.yaml
│          yolov8-Lite-s-seg.yaml
│          yolov8-mobile3s-seg.yaml
│          yolov8-mobileone-backbone-seg.yaml
│          yolov8-MobileOne-seg.yaml
│          yolov8-mobilev3l-seg.yaml
│          yolov8-PPLCNet-seg.yaml
│          yolov8-RepNCSPELAN.yaml
│          yolov8-RepVGG-A1-backbone-seg.yaml
│          yolov8-RepVGG-seg.yaml
│          yolov8-SEAttention-seg.yaml
│          yolov8-seg-p2.yaml
│          yolov8-seg-p34.yaml
│          yolov8-seg-p6.yaml
│          yolov8-seg-p7.yaml
│          yolov8-seg.yaml
│          yolov8-SegNextAttention.yaml
│          yolov8-ShuffleAttention-seg.yaml
│          yolov8-Shufflenetv2-seg.yaml
│          yolov8-SimAM-seg.yaml
│          yolov8-SimSPPF-seg.yaml
│          yolov8-SKAttention-seg.yaml
│          yolov8-SPPCSPC-seg.yaml
│          yolov8-TripletAttention-seg.yaml
│          
├─v9
│  ├─Detect
│  │  │  gelan-c-AKConv.yaml
│  │  │  gelan-c-DCNV3RepNCSPELAN4.yaml
│  │  │  gelan-c-DualConv.yaml
│  │  │  gelan-c-FasterRepNCSPELAN.yaml
│  │  │  gelan-c-KANRepNCSPELAN4.yaml
│  │  │  gelan-c-OREPAN.yaml
│  │  │  gelan-c-p2.yaml
│  │  │  gelan-c-p34.yaml
│  │  │  gelan-c-p6.yaml
│  │  │  gelan-c-SCConv.yaml
│  │  │  gelan-c-SPDConv.yaml
│  │  │  gelan-c.yaml
│  │  │  gelan-e.yaml
│  │  │  gelan-m.yaml
│  │  │  gelan-s-DySnakeRepNCSPELAN.yaml
│  │  │  gelan-s-FasterRepNCSPELAN.yaml
│  │  │  gelan-s.yaml
│  │  │  gelan-t.yaml
│  │  │  gelan.yaml
│  │  │  
│  │  └─u
│  │          yolov9c.yaml
│  │          yolov9e.yaml
│  │          yolov9m.yaml
│  │          yolov9s.yaml
│  │          yolov9t.yaml
│  │          
│  ├─OBB
│  │  │  gelan-c-AKConv-obb.yaml
│  │  │  gelan-c-dobb.yaml
│  │  │  gelan-c-obb-p2.yaml
│  │  │  gelan-c-obb-p34.yaml
│  │  │  gelan-c-obb-p6.yaml
│  │  │  gelan-c-obb.yaml
│  │  │  gelan-c-OREPAN-obb.yaml
│  │  │  gelan-e-obb.yaml
│  │  │  gelan-m-obb.yaml
│  │  │  gelan-obb.yaml
│  │  │  gelan-s-obb.yaml
│  │  │  gelan-t-obb.yaml
│  │  │  
│  │  └─u
│  │          yolov9-c-obb.yaml
│  │          yolov9-e-obb.yaml
│  │          yolov9-obb.yaml
│  │          
│  ├─Pose
│  │  │  gelan-c-AKConv-pose.yaml
│  │  │  gelan-c-dpose.yaml
│  │  │  gelan-c-OREPAN-pose.yaml
│  │  │  gelan-c-pose-p2.yaml
│  │  │  gelan-c-pose-p34.yaml
│  │  │  gelan-c-pose-p6.yaml
│  │  │  gelan-c-pose.yaml
│  │  │  gelan-c-SPDConv-pose.yaml
│  │  │  gelan-e-pose.yaml
│  │  │  gelan-m-pose.yaml
│  │  │  gelan-pose.yaml
│  │  │  gelan-s-pose.yaml
│  │  │  gelan-t-pose.yaml
│  │  │  
│  │  └─u
│  │          yolov9-c-pose.yaml
│  │          yolov9-e-pose.yaml
│  │          yolov9-pose.yaml
│  │          
│  └─Segment
│      │  gelan-c-AKConv-dseg.yaml
│      │  gelan-c-AKConv-seg.yaml
│      │  gelan-c-dseg.yaml
│      │  gelan-c-OREPAN-dseg.yaml
│      │  gelan-c-OREPAN-seg.yaml
│      │  gelan-c-SCConv-dseg.yaml
│      │  gelan-c-seg-p2.yaml
│      │  gelan-c-seg-p34.yaml
│      │  gelan-c-seg-p6.yaml
│      │  gelan-c-seg.yaml
│      │  gelan-e-dseg.yaml
│      │  gelan-e-seg.yaml
│      │  gelan-m-seg.yaml
│      │  gelan-s-DySnakeRepNCSPELAN4-seg.yaml
│      │  gelan-s-seg.yaml
│      │  gelan-seg.yaml
│      │  gelan-t-seg.yaml
│      │  
│      └─u
│              yolov9-c-seg.yaml
│              yolov9-e-seg.yaml
│              yolov9-seg.yaml
│              
├─X
│  ├─Detect
│  │      yoloX-l-lite-c.yaml
│  │      yoloX-l-lite-g.yaml
│  │      yoloX-l-p6.yaml
│  │      yoloX-l-rtdetr.yaml
│  │      yoloX-l.yaml
│  │      yoloX-m-lite-c.yaml
│  │      yoloX-m-lite-g.yaml
│  │      yoloX-m-p6.yaml
│  │      yoloX-m-rtdetr.yaml
│  │      yoloX-m.yaml
│  │      yoloX-n-lite-c.yaml
│  │      yoloX-n-lite-g.yaml
│  │      yoloX-n-p6.yaml
│  │      yoloX-n-rtdetr.yaml
│  │      yoloX-n.yaml
│  │      yoloX-s-lite-c.yaml
│  │      yoloX-s-lite-g.yaml
│  │      yoloX-s-p6.yaml
│  │      yoloX-s-rtdetr.yaml
│  │      yoloX-s.yaml
│  │      yoloX-t-lite-c.yaml
│  │      yoloX-t-lite-g.yaml
│  │      yoloX-t-p6.yaml
│  │      yoloX-t-rtdetr.yaml
│  │      yoloX-t.yaml
│  │      yoloX-x-lite-c.yaml
│  │      yoloX-x-lite-g.yaml
│  │      yoloX-x-p6.yaml
│  │      yoloX-x-rtdetr.yaml
│  │      yoloX-x.yaml
│  │      yoloXnano-Lite-e.yaml
│  │      
│  ├─OBB
│  │      yoloX-l-lite-c-obb.yaml
│  │      yoloX-l-lite-g-obb.yaml
│  │      yoloX-l-obb-p6.yaml
│  │      yoloX-l-obb.yaml
│  │      yoloX-m-lite-c-obb.yaml
│  │      yoloX-m-lite-g-obb.yaml
│  │      yoloX-m-obb-p6.yaml
│  │      yoloX-m-obb.yaml
│  │      yoloX-n-lite-c-obb.yaml
│  │      yoloX-n-lite-g-obb.yaml
│  │      yoloX-n-obb-p6.yaml
│  │      yoloX-n-obb.yaml
│  │      yoloX-s-lite-c-obb.yaml
│  │      yoloX-s-lite-g-obb.yaml
│  │      yoloX-s-obb-p6.yaml
│  │      yoloX-s-obb.yaml
│  │      yoloX-t-lite-c-obb.yaml
│  │      yoloX-t-lite-g-obb.yaml
│  │      yoloX-t-obb-p6.yaml
│  │      yoloX-t-obb.yaml
│  │      yoloX-x-lite-c-obb.yaml
│  │      yoloX-x-lite-g-obb.yaml
│  │      yoloX-x-obb-p6.yaml
│  │      yoloX-x-obb.yaml
│  │      
│  ├─Pose
│  │      yoloX-l-lite-c-pose.yaml
│  │      yoloX-l-lite-g-pose.yaml
│  │      yoloX-l-pose-p6.yaml
│  │      yoloX-l-pose.yaml
│  │      yoloX-m-lite-c-pose.yaml
│  │      yoloX-m-lite-g-pose.yaml
│  │      yoloX-m-pose-p6.yaml
│  │      yoloX-m-pose.yaml
│  │      yoloX-n-lite-c-pose.yaml
│  │      yoloX-n-lite-g-pose.yaml
│  │      yoloX-n-pose-p6.yaml
│  │      yoloX-n-pose.yaml
│  │      yoloX-s-lite-c-pose.yaml
│  │      yoloX-s-lite-g-pose.yaml
│  │      yoloX-s-pose-p6.yaml
│  │      yoloX-s-pose.yaml
│  │      yoloX-t-lite-c-pose.yaml
│  │      yoloX-t-lite-g-pose.yaml
│  │      yoloX-t-pose-p6.yaml
│  │      yoloX-t-pose.yaml
│  │      yoloX-x-lite-c-pose.yaml
│  │      yoloX-x-lite-g-pose.yaml
│  │      yoloX-x-pose-p6.yaml
│  │      yoloX-x-pose.yaml
│  │      
│  └─Segment
│          yoloX-l-lite-c-seg.yaml
│          yoloX-l-lite-g-seg.yaml
│          yoloX-l-seg-p6.yaml
│          yoloX-l-seg.yaml
│          yoloX-m-lite-c-seg.yaml
│          yoloX-m-lite-g-seg.yaml
│          yoloX-m-seg-p6.yaml
│          yoloX-m-seg.yaml
│          yoloX-n-lite-c-seg.yaml
│          yoloX-n-lite-g-seg.yaml
│          yoloX-n-seg-p6.yaml
│          yoloX-n-seg.yaml
│          yoloX-s-lite-c-seg.yaml
│          yoloX-s-lite-g-seg.yaml
│          yoloX-s-seg-p6.yaml
│          yoloX-s-seg.yaml
│          yoloX-t-lite-c-seg.yaml
│          yoloX-t-lite-g-seg.yaml
│          yoloX-t-seg-p6.yaml
│          yoloX-t-seg.yaml
│          yoloX-x-lite-c-seg.yaml
│          yoloX-x-lite-g-seg.yaml
│          yoloX-x-seg-p6.yaml
│          yoloX-x-seg.yaml
│          
├─yolact
│  └─Segment
│          yolact-cspdarknet53.yaml
│          yolact-resnet101.yaml
│          yolact-resnet50.yaml
│          
├─yolo-world
│      yolov8-world.yaml
│      yolov8-worldv2.yaml
│      
└─yoloe
    ├─Detect
    │      yoloe-v11.yaml
    │      yoloe-v8.yaml
    │      
    └─Segment
            yoloe-v11-seg.yaml
            yoloe-v8-seg.yaml
            
```