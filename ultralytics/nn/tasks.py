# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    MP,
    SP,
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    CSPNeXtBlock,
    BottleneckCSP,
    BottleneckCSP2,
    BottleneckCSPA,
    BottleneckCSPB, 
    BottleneckCSPC,
    C2f,
    RTMBlock,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv_BCN,
    FDConv_cfg,
    DualConv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
    SPPCSPC,
    SPPFCSPC,
    DepthSepConv,
    C2f_DCN, 
    C3_DCN, 
    DCNv2,
    SimFusion_4in, 
    SimFusion_3in, 
    InjectionMultiSum_Auto_pool,
    PyramidPoolAgg,
    TopBasicLayer, 
    AdvPoolFusion, 
    IFM, 
    Shortcut, 
    DownC,
    RepConv_v7,
    ReOrg,
    SimSPPF,
    SimConv,
    RepBlock,
    RepVGGBlock,
    Transpose,
    CBH,
    LC_Block,
    Dense,
    conv_bn_relu_maxpool,
    Shuffle_Block,
    DWConvblock,
    CPCAChannelAttention,
    C3C2,
    LayerNorm_s,
    ConvNextBlock,
    CNeB,
    GAMAttention,
    Res,
    ResCSPA,
    ResCSPB,
    ResCSPC,
    ResXCSPA,
    ResXCSPB,
    ResXCSPC,
    SegNext_Attention,
    LayerNormProxy,
    DAttention,
    ShuffleAttention,
    ES_SEModule,
    ES_Bottleneck, 
    DepthWiseConv,
    PointWiseConv,
    MobileOneBlock,
    MobileOne,
    HorLayerNorm,
    gnconv,
    HorBlock,
    CrissCrossAttention,
    SEAttention,
    SKAttention,
    ECAAttention,
    EffectiveSE,
    CAConv,
    BasicConv,
    ZPool,
    AttentionGate,
    TripletAttention,
    Bottleneck_TripletAt,
    C2f_TripletAt,
    C3_TripletAt,
    DeformConv,
    deformable_LKA,
    deformable_LKA_Attention,
    Bottleneck_DLKA,
    C2f_DLKA,
    C3_DLKA,
    SELayer,
    conv_bn_hswish,
    MobileNetV3_InvertedResidual,
    mobilev3_bneck,
    SimAM,
    IDetect,
    ISegment,
    IOBB,
    IPose,
    ImplicitA,
    ImplicitM,
    RepConvN,
    RepNCSPELAN4, 
    OREPANBottleneck,
    OREPANCSP,
    OREPANCSPELAN4,
    DDetect,
    DOBB,
    DSegment,
    DPose,
    ResSPP,
    CSPResNet,
    CSPResNet_CBS,
    ConvBNLayer,
    CSPResStage,
    RepVggBlock,
    EffectiveSELayer,
    BasicBlock,
    AKConv,
    RepNCSP_AKConv,
    RepNCSPELAN4AKConv, 
    KANRepNCSPELAN4,
    FasterRepNCSPELAN4,
    PRepNCSPELAN4,
    PConv,
    DCNV3_YoLo,
    DCNV3RepNCSPELAN4,
    Bottleneck_DCNV3,
    C2f_DCNV3,
    ODConv2d_3rd,
    ODConv_3rd,
    ConvNextBlock,
    LayerNorm_s,
    DropPath,
    SCConv,
    Yolov7_Tiny_E_ELAN,
    Yolov7_Tiny_SPP,
    Yolov7_Tiny_E_ELANMO,
    Yolov7_E_ELAN,
    V7DownSampling,
    MobileOneBlock_origin,
    MobileOne_origin,
    MHSA,
    BottleneckTransformer,
    BoT3,
    ADD,
    ELAN1,
    CSPStage,
    CoordAtt,
    DySample,
    CARAFE,
    CoordConv,
    RFAConv,
    RFCBAMConv,
    RFCAConv,
    RepViTBlock,
    LSKblock,
    PSAFLA,
    C2f_FLA,
    C2f_Context,
    ContextGuidedBlock_Down,
    MultiDilatelocalAttention, 
    PSAMSDA,
    ASPP,
    C2f_Dual,
    C2f_WT,
    BasicRFB,
    BiLevelRoutingAttention,
    DiverseBranchBlock, 
    C2f_DBB,
    SEAM, 
    MultiSEAM,
    FFA,
    MB_TaylorFormer,
    CPA_arch,
    iRMB, 
    C2f_iRMB,
    HAT,
    Down_wt,
    RetinexFormer,
    C2f_SCConv,
    C2f_GhostModule_DynamicConv,
    C2f_ODConv, 
    ODConv2d,
    C2f_DWRSeg,
    SAConv2d, 
    C2f_SAConv,
    C2f_MSBlock,
    C2f_MSBlockv2,
    OREPA, 
    C3_OREPA,
    C2f_OREPA,
    RepNCSPELAN4_low,
    RepNCSPELAN4_high,
    DynamicConv, 
    C2f_DynamicConv,
    IAT,
    SPDConv,
    EMA,
    C2f_FasterBlock,
    C2f_FasterBlock_EMA,
    C3_Faster_CGLU, 
    C2f_Faster_CGLU,
    C3k2_FasterBlock,
    C2f_SENetV1, 
    SELayerV1,
    C2f_SENetV2, 
    SELayerV2, 
    PSASENetV2,
    Light_HGBlock,
    GSConv,
    VoVGSCSP,
    LSKA,
    RCSOSA,
    RepVGG,
    Bi_FPN,
    RIDNET,
    ADNet,
    mn_conv, 
    InvertedBottleneck, 
    MobileNetV3_BLOCK,
    CSPHet,
    CSPPC,
    FocalModulation,
    Zoom_cat, 
    Add, 
    ScalSeq, 
    attention_model,
    EVCBlock,
    DSConv2D,
    C2f_DSConv,
    CBAM,
    VanillaStem, 
    VanillaBlock,
    C3_Star, 
    C3_Star_CAA,
    C2f_Star, 
    C2f_Star_CAA,
    C3_EMBC, 
    C2f_EMBC,
    C3_EMSC, 
    C2f_EMSC,
    C3_EMSCP, 
    C2f_EMSCP,
    C3_UniRepLKNetBlock, 
    C2f_UniRepLKNetBlock, 
    C3_DRB, 
    C2f_DRB,
    C2f_DAttention,
    DilatedReparamBlock, 
    UniRepLKNetBlock,
    MobileOneBlockv5,
    MobileOnev5,
    C3_RetBlock, 
    C2f_RetBlock,
    C3_REPVGGOREPA, 
    C2f_REPVGGOREPA,
    C3_RFAConv, 
    C2f_RFAConv, 
    C3_RFCBAMConv, 
    C2f_RFCBAMConv, 
    C3_RFCAConv, 
    C2f_RFCAConv,
    C3_RVB, 
    C2f_RVB,
    C3_RVB_EMA, 
    C2f_RVB_EMA,
    C2f_UIB,
    PatchMerging, 
    PatchEmbed, 
    SwinStage,
    Concat_BiFPN,
    C3k2_ConvNeXtV2Block, 
    C3k_ConvNeXtV2Block,
    C3k2_WTConv,
    C3k2_SAConv,
    C3k2_RepVGG,
    C2PSA_DAT,
    DAttentionBaseline,
    LAE, 
    C2PSA_SENetV2, 
    SPPFSENetV2,
    RBFKANConv2d,
    ReLUKANConv2d, 
    KANConv2d, 
    FasterKANConv2d, 
    WavKANConv2d, 
    ChebyKANConv2d, 
    JacobiKANConv2d, 
    FastKANConv2d, 
    GRAMKANConv2d,
    C2PSA_MSDA,
    OREPA_2, 
    C3k2_OREPA_backbone, 
    C3k2_OREPA_neck,
    stem, 
    MBConvBlock,
    C2PSA_CGA, 
    LocalWindowAttention,
    C3k2_MLLABlock1, 
    C3k2_MLLABlock2,
    C2PSAMLLA,
    DiTBlock,
    C2PSA_DiTBlock,
    C3k2_DiTBlock,
    C3k2_UIB,
    RepHDW,
    AVG,
    RepHMS,
    ConvMS,
    UniRepLKNetBlock_pro,
    SDFM,
    MANet,
    HyperComputeModule,
    HyperComputeModule_11,
    ALSS,
    LCA,
    EMCAD_block,
    AAttn,
    ABlock,
    A2C2f,
    MSCAM, 
    MSCAMv2, 
    MSCAMv3, 
    MSCAMv4, 
    MSCAMv5,
    LDConv,
    pvt_v2_b0, 
    pvt_v2_b1,
    pvt_v2_b2, 
    pvt_v2_b3, 
    pvt_v2_b4, 
    pvt_v2_b5,
    MobileNetV1,
    MobileNetV2_n,
    MobileNetV2_s,
    MobileNetV2_m,
    MobileNetV3_large_n,
    MobileNetV3_large_s,
    MobileNetV3_large_m,
    MobileNetV3_small_n,
    MobileNetV3_small_s,
    MobileNetV3_small_m,
    MobileNetV4ConvLarge, 
    MobileNetV4ConvSmall, 
    MobileNetV4ConvMedium, 
    MobileNetV4HybridMedium, 
    MobileNetV4HybridLarge,
    mobile_vit_xx_small, 
    mobile_vit_x_small, 
    mobile_vit_small,
    mobile_vit2_xx_small,
    efficient,
    efficientnet_v2,
    Ghostnetv1,
    GhostNetV2,
    GhostNet_1_0,
    convnextv2_atto, 
    convnextv2_femto, 
    convnext_tiny, 
    convnext_small,
    convnext_base, 
    convnext_large, 
    convnext_xlarge,
    convnext_pico, 
    convnextv2_nano, 
    convnextv2_tiny, 
    convnextv2_base, 
    convnextv2_large, 
    convnextv2_huge,
    EfficientViT_M0, 
    EfficientViT_M1, 
    EfficientViT_M2, 
    EfficientViT_M3, 
    EfficientViT_M4, 
    EfficientViT_M5,
    efficientvit_backbone_b0, 
    efficientvit_backbone_b1, 
    efficientvit_backbone_b2, 
    efficientvit_backbone_b3,
    repvit_m0_6, 
    repvit_m0_9, 
    repvit_m1_0, 
    repvit_m1_1, 
    repvit_m1_5, 
    repvit_m2_3,
    starnet_s050, 
    starnet_s100,
    starnet_s150, 
    starnet_s1, 
    starnet_s2, 
    starnet_s3, 
    starnet_s4,
    fasternet_t0, 
    fasternet_t1, 
    fasternet_t2, 
    fasternet_s, 
    fasternet_m, 
    fasternet_l,
    RepLKNet31B, 
    RepLKNet31L, 
    RepLKNetXL,
    unireplknet_a, 
    unireplknet_f, 
    unireplknet_p,
    unireplknet_n, 
    unireplknet_t, 
    unireplknet_s, 
    unireplknet_b, 
    unireplknet_l, 
    unireplknet_xl,
    LSKNET_T, 
    LSKNET_S,
    moganet_xtiny,
    moganet_tiny, 
    moganet_small, 
    moganet_base, 
    moganet_large, 
    moganet_xlarge,
    vanillanet_5, 
    vanillanet_6, 
    vanillanet_7, 
    vanillanet_8, 
    vanillanet_9, 
    vanillanet_10, 
    vanillanet_11, 
    vanillanet_12, 
    vanillanet_13, 
    vanillanet_13_x1_5, 
    vanillanet_13_x1_5_ada_pool,
    mambaout_femto, 
    mambaout_kobe, 
    mambaout_tiny, 
    mambaout_small, 
    mambaout_base,
    RMT_T, 
    RMT_S, 
    RMT_B, 
    RMT_L,
    revcol_tiny, 
    revcol_small, 
    revcol_base, 
    revcol_large, 
    revcol_xlarge,
    SwinTransformer_Tiny, 
    SwinTransformer_Tiny_c24, 
    SwinTransformer_Small, 
    SwinTransformer_Base, 
    SwinTransformer_Large,
    SwinTransformer_mona_Tiny,
    SwinTransformer_mona_Small, 
    SwinTransformer_mona_Base, 
    SwinTransformer_mona_Large,
    swin_transformer_v2_t,
    swin_transformer_v2_s, 
    swin_transformer_v2_b, 
    swin_transformer_v2_l, 
    swin_transformer_v2_h, 
    swin_transformer_v2_g,
    SlabSwinTransformer_T, 
    SlabSwinTransformer_S, 
    SlabSwinTransformer_B,
    EMO_1M, 
    EMO_2M, 
    EMO_5M, 
    EMO_6M,
    EMO2_1M_k5_hybrid, 
    EMO2_1M_k5_hybrid_256, 
    EMO2_1M_k5_hybrid_512, 
    EMO2_2M_k5_hybrid, 
    EMO2_2M_k5_hybrid_256, 
    EMO2_2M_k5_hybrid_512, 
    EMO2_5M_k5_hybrid, 
    EMO2_5M_k5_hybrid_256, 
    EMO2_5M_k5_hybrid_512, 
    EMO2_20M_k5_hybrid, 
    EMO2_20M_k5_hybrid_256, 
    EMO2_50M_k5_hybrid,
    ShuffleNetG1, 
    ShuffleNetG2, 
    ShuffleNetG3, 
    ShuffleNetG4, 
    ShuffleNetG8,
    shufflenetv2_05, 
    shufflenetv2_10, 
    shufflenetv2_15, 
    shufflenetv2_20,
    VGG11, 
    VGG13, 
    VGG16,
    VGG19,
    ResNet18, 
    ResNet34, 
    ResNet50, 
    ResNet101, 
    ResNet152,
    resnet18_moe,
    resnet34_moe, 
    resnet50_moe, 
    resnet101_moe, 
    resnet152_moe,
    uni_resnet50, 
    uni_resnet101,
    orthonet34, 
    orthonet50, 
    orthonet101, 
    orthonet152,
    sa_resnet50, 
    sa_resnet101, 
    sa_resnet152,
    epsanet50, 
    epsanet101, 
    mspanet50,
    mspanet101,
    kw_resnet18, 
    kw_resnet50,
    overlock_xt, 
    overlock_t, 
    overlock_s, 
    overlock_b,
    rdnet_tiny, 
    rdnet_small, 
    rdnet_base, 
    rdnet_large,
    smt_t, 
    smt_s, 
    smt_b, 
    smt_l,
    GroupMixFormerMiny,
    GroupMixFormerTiny, 
    GroupMixFormerSmall, 
    GroupMixFormerBase,
    GroupMixFormerLarge,
    pola_pvt_tiny, 
    pola_pvt_small, 
    pola_pvt_medium,
    pola_pvt_large,
    nextvit_small, 
    nextvit_base, 
    nextvit_large, 
    focalnet_tiny_srf, 
    focalnet_tiny_lrf, 
    focalnet_small_srf, 
    focalnet_small_lrf, 
    focalnet_base_srf, 
    focalnet_base_lrf, 
    focalnet_large_fl3, 
    focalnet_large_fl4, 
    focalnet_xlarge_fl3, 
    focalnet_xlarge_fl4, 
    focalnet_huge_fl3, 
    focalnet_huge_fl4,
    poolformer_s12, 
    poolformer_s24, 
    poolformer_s36, 
    poolformer_m48, 
    poolformer_m36,
    inceptionnext_tiny, 
    inceptionnext_small, 
    inceptionnext_base, 
    inceptionnext_base_384,
    fastvit_t8, 
    fastvit_t12, 
    fastvit_s12, 
    fastvit_sa12, 
    fastvit_sa24, 
    fastvit_sa36, 
    fastvit_ma36,
    NFNetF0, 
    NFNetF1, 
    NFNetF2, 
    NFNetF3, 
    NFNetF4, 
    NFNetF5, 
    NFNetF6, 
    NFNetF7, 
    DFormerv2_S, 
    DFormerv2_B, 
    DFormerv2_L,
    dfformer_s18,
    dfformer_s36, 
    dfformer_m36, 
    dfformer_b36, 
    gfformer_s18, 
    cdfformer_s18, 
    cdfformer_s36, 
    cdfformer_m36, 
    cdfformer_b36, 
    dfformer_s18_gelu, 
    dfformer_s18_relu, 
    dfformer_s18_k2, 
    dfformer_s18_d8, 
    dfformer_s18_afno,
    GhostNet_Reparam,
    efficientformerv2_s0, 
    efficientformerv2_s1, 
    efficientformerv2_s2, 
    efficientformerv2_l,
    EdgeVitXXS, 
    EdgeVitXS, 
    EdgeVitS,
    GreedyViG_S, 
    GreedyViG_M, 
    GreedyViG_B,
    mobilevigv2_ti, 
    mobilevigv2_s, 
    mobilevigv2_m, 
    mobilevigv2_b,
    uniformer_light_xxs, 
    uniformer_light_xs,
    SwiftFormer_XS,
    SwiftFormer_S, 
    SwiftFormer_L1,
    SwiftFormer_L3,
    pvtv2_b0, 
    pvtv2_b1, 
    pvtv2_b2, 
    pvtv2_b2_li, 
    pvtv2_b3, 
    pvtv2_b4, 
    pvtv2_b5,
    slab_pvt_v2_b0, 
    slab_pvt_v2_b1, 
    slab_pvt_v2_b2, 
    slab_pvt_v2_b2_li, 
    slab_pvt_v2_b3, 
    slab_pvt_v2_b4, 
    slab_pvt_v2_b5,
    conv2former_n, 
    conv2former_t, 
    conv2former_s, 
    conv2former_b, 
    conv2former_b_22k, 
    conv2former_l,
    LWGANet_L0_1242_e32_k11_GELU, 
    LWGANet_L1_1242_e64_k11_GELU, 
    LWGANet_L2_1442_e96_k11_ReLU,
    hornet_tiny_7x7, 
    hornet_tiny_gf, 
    hornet_small_7x7, 
    hornet_small_gf, 
    hornet_base_7x7, 
    hornet_base_gf, 
    hornet_base_gf_img384,
    hornet_large_7x7, 
    hornet_large_gf, 
    hornet_large_gf_img384,
    EfficientViM_M1,
    EfficientViM_M2, 
    EfficientViM_M3, 
    EfficientViM_M4,
    EfficientViM_M5,
    SHViT_S1, 
    SHViT_S2, 
    SHViT_S3, 
    SHViT_S4,
    RCViT_XS, 
    RCViT_S, 
    RCViT_M, 
    RCViT_T,
    gc_vit_xxtiny, 
    gc_vit_xtiny, 
    gc_vit_tiny, 
    gc_vit_tiny2, 
    gc_vit_small, 
    gc_vit_small2, 
    gc_vit_base, 
    gc_vit_large, 
    gc_vit_large_224_21k, 
    gc_vit_large_384_21k, 
    gc_vit_large_512_21k,
    convit_tiny_backbone, 
    convit_small_backbone, 
    convit_base_backbone,
    RepVGG_A0, 
    RepVGG_A1, 
    RepVGG_A2, 
    RepVGG_B0, 
    RepVGG_B1, 
    RepVGG_B1g2, 
    RepVGG_B1g4, 
    RepVGG_B2, 
    RepVGG_B2g2, 
    RepVGG_B2g4, 
    RepVGG_B3, 
    RepVGG_B3g2, 
    RepVGG_B3g4, 
    RepVGG_D2se,
    QARepVGG_A0, 
    QARepVGGV1_A0, 
    QARepVGGV2_A0, 
    QARepVGGV2_A0_d01,
    QARepVGGV2_A0_DW, 
    QARepVGGV6_A0, 
    QARepVGG_A0_ReLU6, 
    QARepVGGV2_A0_PReLU, 
    QARepVGGV2_A1, 
    QARepVGGV2_A2, 
    QARepVGGV2_B0, 
    QARepVGGV2_B1, 
    QARepVGGV2_B1g2, 
    QARepVGGV2_B1g4, 
    QARepVGGV2_D2se,
    decouplenet_d0, 
    decouplenet_d1, 
    decouplenet_d2,
    sbcformer_xs, 
    sbcformer_s, 
    sbcformer_b, 
    sbcformer_l,
    fanet_tiny, 
    fanet_small,
    cosnet_tiny, 
    cosnet_small, 
    cosnet_base,
    wtconvnext_tiny, 
    wtconvnext_small, 
    wtconvnext_base,
    wtconvnext_large, 
    wtconvnext_xlarge,
    MLLA_Tiny, 
    MLLA_Small, 
    MLLA_Base,
    pkinet_t, 
    pkinet_s, 
    pkinet_b,
    glnet_stl, 
    glnet_stl_paramslot, 
    glnet_4g, 
    glnet_9g, 
    glnet_16g,
    RAVLT_T, 
    RAVLT_S, 
    RAVLT_B, 
    RAVLT_L,
    slak_tiny, 
    slak_small, 
    slak_base, 
    slak_large,
    svt_s, 
    svt_b, 
    svt_l,
    EViT_Tiny, 
    EViT_Small, 
    EViT_Base, 
    EViT_Large,
    sgformer_s, 
    sgformer_m, 
    sgformer_b,
    spanet_s, 
    spanet_m, 
    spanet_mx, 
    spanet_b, 
    spanet_bx,
    StripMLPNet_LightTiny, 
    StripMLPNet_Tiny, 
    StripMLPNet_Small, 
    StripMLPNet_Base,
    identityformer_s12, 
    identityformer_s24, 
    identityformer_s36, 
    identityformer_m36, 
    identityformer_m48, 
    randformer_s12, 
    randformer_s24, 
    randformer_s36, 
    randformer_m36, 
    randformer_m48, 
    poolformerv2_s12, 
    poolformerv2_s24, 
    poolformerv2_s36, 
    poolformerv2_m36, 
    poolformerv2_m48, 
    convformer_s18, 
    convformer_s36, 
    convformer_m36, 
    convformer_b36, 
    caformer_s18, 
    caformer_s36, 
    caformer_m36, 
    caformer_b36,
    iformer_small, 
    iformer_base, 
    iformer_large, 
    van_b0, 
    van_b1, 
    van_b2, 
    van_b3, 
    van_b4, 
    van_b5, 
    van_b6,
    vheat_tiny, 
    vheat_small, 
    vheat_base,
    LSNet_T,
    LSNet_S,
    LSNet_B,
    StripNet_tiny, 
    StripNet_small,
    transxnet_tiny, 
    transxnet_small, 
    transxnet_base,
    parcnetv2_xt, 
    parcnetv2_tiny, 
    parcnetv2_small, 
    parcnetv2_base,
    MALA_T, 
    MALA_S, 
    MALA_B, 
    MALA_L,
    mpvit_tiny, 
    mpvit_xsmall, 
    mpvit_small, 
    mpvit_base,
    uninext_t, 
    uninext_s, 
    uninext_b,
    stvit_small, 
    stvit_base, 
    stvit_large,
    fat_b0, 
    fat_b1, 
    fat_b2, 
    fat_b3,
    debi_tiny, 
    debi_small, 
    debi_base,
    maxvit_tiny, 
    maxvit_small, 
    maxvit_base, 
    maxvit_large,
    scalable_vit_s, 
    scalable_vit_b, 
    scalable_vit_l,
    rest_lite, 
    rest_small, 
    rest_base, 
    rest_large, 
    restv2_tiny, 
    restv2_small, 
    restv2_base, 
    restv2_large,
    VanillaNet,
    UniRepLKNet,
    OverLoCK,
    RepViT,
    FastViT,
    MobileViG,
    RepVGGBlock,
    QARepVGGBlock,
    C2PSA_HV_LCA,
    C2PSA_HV_LCA_DynamicTanh,
    LCA_Concat,
    LCA_DynamicTanh_Concat,
    Conv_DynamicTanh,
    C2f_MultiOGA, 
    ChannelAggregationFFN, 
    MultiOrderGatedAggregation,
    C2PSA_Agent,
    C2PSA_KS,
    MAFOBB,
    MAFDetect,
    MAFSegment,
    MAFPose,
    FCM, 
    Pzconv,  
    FCM_3,
    FCM_2, 
    FCM_1, 
    Down,
    CSP_EIMS,
    HRIF,
    MyConcat4, 
    MyConcat6, 
    CST, 
    MCS,
    DySnakeConv,
    C3k2_DSConv,
    DySnakeRepNCSPELAN4,
    SFS_Conv,
    SNI, 
    GSConvE,
    GSConvE2, 
    ESD, 
    ESD2,
    DSConv,
    HyperACE,
    DownsampleConv,
    FullPAD_Tunnel,
    DSC3k2,
    F2SoftHG, 
    ShapeAlignConv, 
    MergeConv,
    Index,
    LRPCHead,
    TorchVision,
    YOLOEDetect,
    YOLOESegment,
    MFAM,
    IEMA,
    DASI,
    PST,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


class BaseModel(torch.nn.Module):
    """Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion, information
    display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Module): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, *args, **kwargs):
        """Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if hasattr(m, 'backbone'):
                x = m(x)
                if len(x) != 5:  # 0 - 5
                    x.insert(0, None)
                for index, i in enumerate(x):
                    if index in self.save:
                        y.append(i)
                    else:
                        y.append(None)
                x = x[-1]  # 最后一个输出传给下一层
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer for improved computation
        efficiency.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, FDConv_cfg, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, (RepConvN)) and hasattr(m, 'fuse_convs'):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv_v7):
                    m.fuse_repvgg_block()
                if isinstance(m, (MobileOneBlock_origin, MobileOne_origin, MobileOneBlockv5, MobileOnev5, FastViT, MobileViG)) and hasattr(m, 'reparameterize'):
                    m.reparameterize()
                if isinstance(m, (UniRepLKNet)) and hasattr(m, 'reparameterize'):
                    m.switch_to_deploy()
                if isinstance(m, (RepVGGBlock)) and hasattr(m, 'rbr_reparam'):
                    m.switch_to_deploy()
                if isinstance(m, (QARepVGGBlock)) and hasattr(m, 'rbr_reparam'):
                    m.switch_to_deploy()
                if isinstance(m, (OverLoCK)):
                    m.reparm()
                if isinstance(m, (OREPA, OREPA_2, VanillaStem, VanillaBlock, DilatedReparamBlock, UniRepLKNetBlock, RepViT, VanillaNet)):
                    m.switch_to_deploy()
                if isinstance(m, UniRepLKNetBlock_pro):
                    m.reparameterize()
                    LOGGER.info("Switch model to UniRepLKNetBlock")
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, v10Detect):
                    m.fuse()  # remove one2many head
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image that the model will be trained on.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, (Detect, MAFDetect, IDetect, DDetect)
        ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented
    inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, MAFDetect, IDetect, DDetect)):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, YOLOESegment, Pose, OBB,
                                                            ISegment, IPose, IOBB,
                                                            MAFSegment, MAFPose, MAFOBB,
                                                            DOBB, DSegment, DPose)) else self.forward(x)

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Augmented inference output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int): Flip type (0=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails.

        Args:
            y (list[torch.Tensor]): List of detection tensors.

        Returns:
            (list[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss
    computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo11n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for
    pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for
    keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo11n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks, providing model
    initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo11n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo11n-cls.yaml", ch=3, nc=None, verbose=True):
        """Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _apply(self, fn):
        """Apply a function to all tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (RTDETRDetectionModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]
        m.anchors = fn(m.anchors)
        m.valid_mask = fn(m.valid_mask)
        return self

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions.

        Returns:
            loss_sum (torch.Tensor): Total loss value.
            loss_items (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class
    specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting
    both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings cooperated with reprta module.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("mobileclip:blt", device=device)

        model = self.clip_model if cache_clip_model else build_text_model("mobileclip:blt", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run auxiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """Get visual embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (list[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2)
            for i, (cls, pf, loc) in enumerate(zip(vocab, head.cv3, head.cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """Get fused vocabulary layer from the model.

        Args:
            names (list): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        vocab = nn.ModuleList()
        for cls_head in head.cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (list[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """Get class positional embeddings.

        Args:
            tpe (torch.Tensor, optional): Text positional embeddings.
            vpe (torch.Tensor, optional): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x = m(x, cls_pe)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPDetectLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing
    specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = TVPSegmentLoss(self) if visual_prompt else self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging
    or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Generate the YOLO network's final layer.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            y (torch.Tensor): Concatenated predictions from all models.
            train_out (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've
    moved a module from one location to another, but you still want to support the old import paths for backwards
    compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Notes:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches
    the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        ckpt (dict): The loaded model checkpoint.
        file (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo11n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """Load a single model weights.

    Args:
        weight (str | Path): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        model (torch.nn.Module): Loaded model.
        ckpt (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model

    # Model compatibility updates
    model.args = args  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        model (torch.nn.Sequential): PyTorch model.
        save (list): Sorted list of output layers.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels, threshold = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    backbone = False
    
    base_modules = frozenset(
        {
            Classify,
            Conv,
            Conv_BCN,
            FDConv_cfg,
            DualConv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            BottleneckCSP,
            BottleneckCSP2,
            BottleneckCSPA, 
            BottleneckCSPB, 
            BottleneckCSPC,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            C1,
            C2,
            C2f,
            RTMBlock,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            RepConv_v7, 
            SPPCSPC,
            SPPFCSPC,
            DepthSepConv,
            C2f_DCN, 
            C3_DCN,
            DCNv2,
            DownC,
            RepBlock,
            SimConv,
            SimSPPF,
            RepVGGBlock,
            CBH,
            LC_Block,
            Dense,
            conv_bn_relu_maxpool,
            Shuffle_Block,
            DWConvblock,
            CPCAChannelAttention,
            C3C2,
            CNeB,
            GAMAttention,
            ResCSPA,
            ResCSPB,
            ResCSPC,
            ResXCSPA,
            ResXCSPB,
            ResXCSPC,
            ES_Bottleneck,
            MobileOneBlock,
            HorBlock,
            EffectiveSE,
            CAConv,
            C2f_TripletAt,
            C3_TripletAt,
            C2f_DLKA,
            C3_DLKA,
            conv_bn_hswish, 
            MobileNetV3_InvertedResidual,
            mobilev3_bneck,
            RepNCSPELAN4, 
            RepNCSPELAN4_low,
            RepNCSPELAN4_high,
            OREPANCSPELAN4,
            SimAM,
            AKConv,
            RepNCSP_AKConv,
            RepNCSPELAN4AKConv,
            KANRepNCSPELAN4,
            FasterRepNCSPELAN4,
            PRepNCSPELAN4,
            PConv,
            DCNV3_YoLo,
            DCNV3RepNCSPELAN4,
            Bottleneck_DCNV3,
            C2f_DCNV3,
            ODConv_3rd,
            ConvNextBlock,
            Yolov7_Tiny_E_ELAN,
            Yolov7_Tiny_SPP,
            Yolov7_Tiny_E_ELANMO,
            Yolov7_E_ELAN,
            V7DownSampling,
            MobileOneBlock_origin,
            BoT3,
            ELAN1,
            CSPStage,
            CoordAtt,
            CARAFE,
            CoordConv,
            RFAConv,
            RFCBAMConv,
            RFCAConv,
            RepViTBlock,
            PSAFLA,
            C2f_FLA,
            C2f_Context,
            PSAMSDA,
            ASPP,
            C2f_Dual,
            C2f_WT,
            BasicRFB,
            DiverseBranchBlock,
            C2f_DBB,
            C2f_iRMB,
            Down_wt,
            C2f_GhostModule_DynamicConv,
            ODConv2d,
            C2f_ODConv,
            SAConv2d,
            C2f_SAConv,
            C2f_MSBlock,
            C2f_MSBlockv2,
            C3_OREPA,
            C2f_OREPA,
            DynamicConv, 
            C2f_DynamicConv,
            SPDConv,
            C2f_FasterBlock,
            C2f_FasterBlock_EMA,
            C3_Faster_CGLU, 
            C2f_Faster_CGLU,
            C3k2_FasterBlock,
            C2f_SENetV1,
            C2f_SENetV2,  
            PSASENetV2,
            C2f_DWRSeg,
            GSConv,
            VoVGSCSP,
            RCSOSA,
            RepVGG,
            mn_conv, 
            InvertedBottleneck, 
            MobileNetV3_BLOCK,
            CSPHet,
            CSPPC,
            EVCBlock,
            DSConv2D,
            C2f_DSConv,
            VanillaStem, 
            VanillaBlock,
            C3_Star, 
            C3_Star_CAA,
            C2f_Star, 
            C2f_Star_CAA,
            C3_EMBC, 
            C2f_EMBC,
            C3_EMSC, 
            C2f_EMSC,
            C3_EMSCP, 
            C2f_EMSCP,
            C3_UniRepLKNetBlock, 
            C2f_UniRepLKNetBlock, 
            C3_DRB, 
            C2f_DRB,
            C2f_DAttention,
            MobileOnev5,
            C3_RetBlock, 
            C2f_RetBlock,
            C3_REPVGGOREPA, 
            C2f_REPVGGOREPA,
            C3_RFAConv, 
            C2f_RFAConv, 
            C3_RFCBAMConv, 
            C2f_RFCBAMConv, 
            C3_RFCAConv, 
            C2f_RFCAConv,
            C3_RVB, 
            C2f_RVB,
            C3_RVB_EMA, 
            C2f_RVB_EMA,
            C2f_UIB,
            PatchMerging, 
            PatchEmbed, 
            SwinStage,
            C3k2_ConvNeXtV2Block, 
            C3k2_WTConv,
            C3k2_SAConv,
            C3k2_RepVGG,
            C2PSA_DAT,
            C2PSA_SENetV2, 
            SPPFSENetV2,
            RBFKANConv2d, 
            ReLUKANConv2d, 
            KANConv2d, 
            FasterKANConv2d, 
            WavKANConv2d, 
            ChebyKANConv2d, 
            JacobiKANConv2d, 
            FastKANConv2d, 
            GRAMKANConv2d,
            C2PSA_MSDA,
            OREPA_2, 
            C3k2_OREPA_backbone, 
            C3k2_OREPA_neck,
            stem, 
            MBConvBlock,
            C2PSA_CGA, 
            LocalWindowAttention,
            C3k2_MLLABlock1, 
            C3k2_MLLABlock2,
            C2PSAMLLA,
            C2PSA_DiTBlock,
            C3k2_DiTBlock,
            C3k2_UIB,
            RepHDW,
            RepHMS,
            MANet,
            HyperComputeModule_11,
            ALSS,
            LCA,
            A2C2f,
            LDConv,
            C2PSA_HV_LCA,
            C2PSA_HV_LCA_DynamicTanh,
            Conv_DynamicTanh,
            C2f_MultiOGA,
            C2PSA_Agent,
            C2PSA_KS,
            FCM, 
            Pzconv,  
            FCM_3,
            FCM_2, 
            FCM_1, 
            Down,
            CSP_EIMS,
            MCS,
            CST,
            DySnakeConv,
            C3k2_DSConv,
            DySnakeRepNCSPELAN4,
            SFS_Conv,
            GSConvE, 
            GSConvE2, 
            ESD, 
            ESD2,
            DSC3k2,
            DSConv,
            torch.nn.ConvTranspose2d,
            MFAM,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP, 
            BottleneckCSP2, 
            BottleneckCSPA, 
            BottleneckCSPB, 
            BottleneckCSPC, 
            C1, 
            C2, 
            C2f, 
            RTMBlock, 
            C3k2, 
            C2fAttn, 
            C3, 
            C3TR, 
            C3Ghost, 
            C3x, 
            RepC3, 
            C2fPSA, 
            C2fCIB, 
            C2PSA,
            ResCSPA, 
            ResCSPB, 
            ResCSPC, 
            ResXCSPA, 
            ResXCSPB, 
            ResXCSPC, 
            HorBlock,
            SPPCSPC, 
            CSPStage, 
            C2f_DCNV3, 
            C2f_FLA, 
            C2f_Context, 
            C2f_Dual, 
            C2f_WT, 
            C2f_DBB,
            C2f_iRMB, 
            C2f_GhostModule_DynamicConv, 
            C2f_ODConv, 
            C2f_SAConv, 
            C2f_MSBlock, 
            C2f_MSBlockv2,
            C3_OREPA, 
            C2f_OREPA, 
            C2f_DynamicConv, 
            C2f_FasterBlock, 
            C2f_FasterBlock_EMA, 
            C3_Faster_CGLU, 
            C2f_Faster_CGLU, 
            C3k2_FasterBlock, 
            C2f_SENetV1, 
            C2f_SENetV2, 
            VoVGSCSP, 
            C2f_TripletAt, 
            C2f_DLKA, 
            C3_DLKA, 
            CSPHet, 
            CSPPC, 
            C2f_DSConv, 
            C2f_DWRSeg, 
            C3_Star, 
            C3_Star_CAA, 
            C2f_Star, 
            C2f_Star_CAA, 
            C3_EMBC, 
            C2f_EMBC, 
            C3_EMSC, 
            C2f_EMSC, 
            C3_EMSCP, 
            C2f_EMSCP, 
            C3_UniRepLKNetBlock, 
            C2f_UniRepLKNetBlock, 
            C3_DRB, 
            C2f_DRB, 
            C2f_DAttention,
            C3_RetBlock, 
            C2f_RetBlock, 
            C3_REPVGGOREPA, 
            C2f_REPVGGOREPA,
            C3_RFAConv, 
            C2f_RFAConv, 
            C3_RFCBAMConv, 
            C2f_RFCBAMConv,
            C3_RFCAConv, 
            C2f_RFCAConv,
            C3_RVB, 
            C2f_RVB, 
            C3_RVB_EMA, 
            C2f_RVB_EMA, 
            C2f_UIB, 
            C3k2_ConvNeXtV2Block, 
            C3k2_WTConv, 
            C3k2_SAConv,
            C3k2_RepVGG, 
            RCSOSA, 
            C2PSA_DAT, 
            C2PSA_MSDA, 
            C2PSA_SENetV2, 
            C3k2_OREPA_backbone, 
            C3k2_OREPA_neck,
            C2PSA_CGA, 
            C3k2_MLLABlock1, 
            C3k2_MLLABlock2, 
            C2PSAMLLA, 
            C2PSA_DiTBlock, 
            C3k2_DiTBlock, 
            C3k2_UIB,
            RepHDW, 
            MANet, 
            ALSS, 
            A2C2f, 
            C2PSA_HV_LCA, 
            C2PSA_HV_LCA_DynamicTanh,
            C2f_MultiOGA, 
            C2PSA_Agent, 
            C2PSA_KS,
            CSP_EIMS, 
            CST, 
            C3k2_DSConv, 
            DSC3k2,
            PST,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        t = m
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m in {C3k2, C3k2_ConvNeXtV2Block, C3k2_FasterBlock, C3k2_WTConv, C3k2_SAConv, C3k2_RepVGG, C3k2_OREPA_backbone, C3k2_OREPA_neck,
                     C3k2_MLLABlock1, C3k2_MLLABlock2, C3k2_DiTBlock, C3k2_UIB, C3k2_DSConv, DSC3k2}:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m in {SegNext_Attention, DAttention, DAttentionBaseline, TripletAttention, deformable_LKA_Attention, 
                   ContextGuidedBlock_Down, MultiDilatelocalAttention, BiLevelRoutingAttention,
                   SEAM, MultiSEAM, FFA, HAT, OREPA, IAT, SELayerV1, SELayerV2, LSKA, RIDNET,
                   ADNet, FocalModulation, EMA, LAE, LocalWindowAttention, DiTBlock}:
            c2 = ch[f]
            args = [c2, *args]
        elif m in {Bi_FPN}:
            length = len([ch[x] for x in f])
            args = [length]
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock, Light_HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m in {HGBlock, Light_HGBlock}:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m in {EMCAD_block, MultiOrderGatedAggregation}:
            args = [ch[f]]
        elif m is iRMB:
            args = [ch[f], ch[f]]
        elif m is LSKblock:
            c1 = ch[f]
            args = [c1, *args[0:]]
        elif m is AVG:
            c2 = ch[f]
        elif m in (Concat, SimFusion_4in, AdvPoolFusion, Concat_BiFPN, LCA_Concat, LCA_DynamicTanh_Concat, MyConcat4, MyConcat6):
            c2 = sum(ch[x] for x in f)
        elif m is DASI:
            # 假设DASI的from参数为[f_low, f_mid, f_high]，对应低、中、高分辨率特征
            f_high, f_low, f_mid =  f  # 解包三个索引
            c1 = [ch[f_high], ch[f_mid], ch[f_low]]
            c2 = args[3]  # DASI模块的输出通道数（由args[0]指定，如512）
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [ch[f_high], ch[f_mid], ch[f_low], c2]
        elif m is IEMA:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, 16]
        elif m is HyperComputeModule:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, threshold]
        elif m is ADD:
            c2 = sum(ch[x] for x in f)//2
        elif m is SimFusion_3in:
            c2 = args[0]
            if c2 != nc:  # if not output
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [[ch[f_] for f_ in f], c2]
        elif m is IFM:
            c1 = ch[f]
            c2 = sum(args[0])
        elif m is InjectionMultiSum_Auto_pool:
            c1 = ch[f[0]]
            c2 = args[0]
            args = [c1, *args]
        elif m is PyramidPoolAgg:
            c2 = args[0]
            args = [sum([ch[f_] for f_ in f]), *args]
        elif m is TopBasicLayer:
            c2 = sum(args[1])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Transpose:
            c1 = ch[f]
            c2 = args[0]
            args = [c1, c2, *args[1:]]
        elif m in (MobileOne, MobileOne_origin, MobileOneBlock_origin):
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, n, *args[1:]]   
        elif m is Zoom_cat:
            c2 = sum(ch[x] for x in f)
        elif m is Add:
            c2 = ch[f[-1]]
        elif m is ScalSeq:
            c1 = [ch[x] for x in f]
            c2 = make_divisible(args[0] * width, 8)
            args = [c1, c2]
        elif m is attention_model:
            args = [ch[f[-1]]]
        elif m is SCConv:
            c1 = ch[f]
            c2 = args[0]
            args = [c1, *args]
        elif m in {SDFM}:
            c2 = ch[f[1]]
            args = [c2, *args]
        elif m in (CrissCrossAttention, ShuffleAttention, SEAttention,
                   SKAttention, ECAAttention, MHSA, CBAM):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if not output
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, *args[1:]]
        elif m is ChannelAggregationFFN:
            args = [ch[f], c2]
        elif m in (CSPResNet_CBS, CSPResNet, ConvBNLayer, ResSPP):
            c2 = args[1]
        elif m is SNI:
            c1, c2, up_f = ch[f], make_divisible(args[0] * width, 8), args[1]
            args = [c1, c2, up_f]
        elif m in frozenset(
            {Detect, MAFDetect, IDetect, DDetect, WorldDetect, YOLOEDetect, 
             Segment, MAFSegment, ISegment, DSegment, YOLOESegment, 
             Pose, MAFPose, IPose, DPose,
             OBB, MAFOBB, IOBB, DOBB, 
             ImagePoolingAttn, v10Detect}
        ):
            args.append([ch[x] for x in f])
            if m in {Segment, MAFSegment, ISegment, DSegment, YOLOESegment}:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, MAFDetect, IDetect, DDetect, YOLOEDetect, 
                     Segment, MAFSegment, ISegment, DSegment, YOLOESegment, 
                     Pose, MAFPose, IPose, DPose, 
                     OBB, MAFOBB, IOBB, DOBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is PST:
            c1, c_up, c2 = ch[f[0]], ch[f[1]], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c_up, c2, *args[1:]]
            args.insert(3, n)  # number of repeats
            n = 1
            legacy = False
            if scale in "lx":  # for L/X sizes
                args.extend((True, 1.2))
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is HyperACE:
            legacy = False
            c1 = ch[f[1]]
            c2 = args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            he = args[1] 
            if scale in "n":
                he = int(args[1] * 0.5)
            elif scale in "x":
                he = int(args[1] * 1.5)
            args = [c1, c2, n, he, *args[2:]]
            n = 1
            if scale in "lx":  # for L/X sizes
                args.append(False)
        elif m is F2SoftHG:
            c1 = ch[f[1]]
            c2 = c1
            args = [c1, c2, *args]
            if scale in "m":  
                args.append(False)
        elif m is DownsampleConv:
            c1 = ch[f]
            c2 = c1 * 2
            args = [c1]
            if scale in "lx":  # for L/X sizes
                args.append(False)
                c2 =c1
        elif m is ShapeAlignConv:
            c2 = ch[f] * 2
            args = [ch[f]]
            if scale in "m": 
                c2 = ch[f]
                args.append(False)
        elif m is FullPAD_Tunnel:
            c2 = ch[f[0]]
        elif m is MergeConv:
            c2 = ch[f[0]]
            args = [c2]
        elif m in {HRIF}:
            c1 = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m in {MSCAM, MSCAMv2, MSCAMv3, MSCAMv4, MSCAMv5}:
            c1 = c2 = ch[f]
            args = [c1, args[0]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        elif m in {pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5,
                   MobileNetV1, MobileNetV2_n, MobileNetV2_s, MobileNetV2_m, MobileNetV3_large_n, MobileNetV3_large_s, 
                   MobileNetV3_large_m, MobileNetV3_small_n, MobileNetV3_small_s, MobileNetV3_small_m, mobile_vit_xx_small, mobile_vit_x_small, mobile_vit_small,
                   MobileNetV4ConvLarge, MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4HybridMedium, MobileNetV4HybridLarge,
                   mobile_vit2_xx_small, efficient, efficientnet_v2, Ghostnetv1, GhostNetV2, convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge, 
                   convnextv2_atto, convnextv2_femto, convnext_pico, convnextv2_nano, convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge, 
                   EfficientViT_M0, EfficientViT_M1, EfficientViT_M2, EfficientViT_M3, EfficientViT_M4, EfficientViT_M5, efficientvit_backbone_b0, 
                   efficientvit_backbone_b1, efficientvit_backbone_b2, efficientvit_backbone_b3, repvit_m0_6, repvit_m0_9, repvit_m1_0, 
                   repvit_m1_1, repvit_m1_5, repvit_m2_3, starnet_s050, starnet_s100, starnet_s150, starnet_s1, starnet_s2, starnet_s3, starnet_s4,
                   fasternet_t0, fasternet_t1, fasternet_t2, fasternet_s, fasternet_m, fasternet_l, unireplknet_a, unireplknet_f, unireplknet_p, unireplknet_n, unireplknet_t, 
                   unireplknet_s, unireplknet_b, unireplknet_l, unireplknet_xl, LSKNET_T, LSKNET_S, moganet_xtiny, moganet_tiny, moganet_small, moganet_base, moganet_large, moganet_xlarge,
                   mspanet50, mspanet101, vanillanet_5, vanillanet_6, vanillanet_7, vanillanet_8, vanillanet_9, vanillanet_10, vanillanet_11, vanillanet_12, 
                   vanillanet_13, vanillanet_13_x1_5, vanillanet_13_x1_5_ada_pool, mambaout_femto, mambaout_kobe, mambaout_tiny, mambaout_small, mambaout_base,
                   RMT_T, RMT_S, RMT_B, RMT_L, revcol_tiny, revcol_small, revcol_base, revcol_large, revcol_xlarge, SwinTransformer_Tiny, SwinTransformer_Tiny_c24, SwinTransformer_Small, 
                   SwinTransformer_Base, SwinTransformer_Large, SwinTransformer_mona_Tiny, SwinTransformer_mona_Small, SwinTransformer_mona_Base, SwinTransformer_mona_Large, 
                   swin_transformer_v2_t, swin_transformer_v2_s, swin_transformer_v2_b, swin_transformer_v2_l, swin_transformer_v2_h, swin_transformer_v2_g, SlabSwinTransformer_T, SlabSwinTransformer_S, 
                   SlabSwinTransformer_B, EMO_1M, EMO_2M, EMO_5M, EMO_6M, EMO2_1M_k5_hybrid, EMO2_1M_k5_hybrid_256, EMO2_1M_k5_hybrid_512, EMO2_2M_k5_hybrid, EMO2_2M_k5_hybrid_256, EMO2_2M_k5_hybrid_512, 
                   EMO2_5M_k5_hybrid, EMO2_5M_k5_hybrid_256, EMO2_5M_k5_hybrid_512, EMO2_20M_k5_hybrid, EMO2_20M_k5_hybrid_256, EMO2_50M_k5_hybrid, ShuffleNetG1, ShuffleNetG2, ShuffleNetG3, ShuffleNetG4, 
                   ShuffleNetG8, shufflenetv2_05, shufflenetv2_10, shufflenetv2_15, shufflenetv2_20, VGG11, VGG13, VGG16, VGG19, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, resnet18_moe, resnet34_moe, 
                   resnet50_moe, resnet101_moe, resnet152_moe, overlock_xt, overlock_t, overlock_s, overlock_b, rdnet_tiny, rdnet_small, rdnet_base, rdnet_large, smt_t, smt_s, smt_b, smt_l, GroupMixFormerMiny, 
                   GroupMixFormerTiny, GroupMixFormerSmall, GroupMixFormerBase, GroupMixFormerLarge, pola_pvt_tiny, pola_pvt_small, pola_pvt_medium, pola_pvt_large, nextvit_small, nextvit_base, nextvit_large, 
                   focalnet_tiny_srf, focalnet_tiny_lrf, focalnet_small_srf, focalnet_small_lrf, focalnet_base_srf, focalnet_base_lrf, focalnet_large_fl3, focalnet_large_fl4, focalnet_xlarge_fl3, focalnet_xlarge_fl4, 
                   focalnet_huge_fl3, focalnet_huge_fl4, poolformer_s12, poolformer_s24, poolformer_s36, poolformer_m48, poolformer_m36, inceptionnext_tiny, inceptionnext_small, inceptionnext_base, inceptionnext_base_384,
                   fastvit_t8, fastvit_t12, fastvit_s12, fastvit_sa12, fastvit_sa24, fastvit_sa36, fastvit_ma36, NFNetF0, NFNetF1, NFNetF2, NFNetF3, NFNetF4, NFNetF5, NFNetF6, NFNetF7,
                   DFormerv2_S, DFormerv2_B, DFormerv2_L, dfformer_s18, dfformer_s36, dfformer_m36, dfformer_b36, gfformer_s18, cdfformer_s18, cdfformer_s36, cdfformer_m36, cdfformer_b36, 
                   dfformer_s18_gelu, dfformer_s18_relu, dfformer_s18_k2, dfformer_s18_d8, dfformer_s18_afno, GhostNet_1_0, epsanet50, epsanet101, GhostNet_Reparam,
                   efficientformerv2_s0, efficientformerv2_s1, efficientformerv2_s2, efficientformerv2_l, EdgeVitXXS, EdgeVitXS, EdgeVitS, sa_resnet50, sa_resnet101, sa_resnet152, 
                   GreedyViG_S, GreedyViG_M, GreedyViG_B, mobilevigv2_ti, mobilevigv2_s, mobilevigv2_m, mobilevigv2_b, uniformer_light_xxs, uniformer_light_xs, 
                   SwiftFormer_XS, SwiftFormer_S, SwiftFormer_L1, SwiftFormer_L3, pvtv2_b0, pvtv2_b1, pvtv2_b2, pvtv2_b2_li, pvtv2_b3, pvtv2_b4, pvtv2_b5, slab_pvt_v2_b0, slab_pvt_v2_b1, 
                   slab_pvt_v2_b2, slab_pvt_v2_b2_li, slab_pvt_v2_b3, slab_pvt_v2_b4, slab_pvt_v2_b5, conv2former_n, conv2former_t, conv2former_s, conv2former_b, conv2former_b_22k, 
                   conv2former_l, LWGANet_L0_1242_e32_k11_GELU, LWGANet_L1_1242_e64_k11_GELU, LWGANet_L2_1442_e96_k11_ReLU, hornet_tiny_7x7, hornet_tiny_gf, hornet_small_7x7, hornet_small_gf, 
                   hornet_base_7x7, hornet_base_gf, hornet_base_gf_img384, hornet_large_7x7, hornet_large_gf, hornet_large_gf_img384, EfficientViM_M1, EfficientViM_M2, 
                   EfficientViM_M3, EfficientViM_M4, EfficientViM_M5, SHViT_S1, SHViT_S2, SHViT_S3, SHViT_S4, RCViT_XS, RCViT_S, RCViT_M, RCViT_T, gc_vit_xxtiny, gc_vit_xtiny, 
                   gc_vit_tiny, gc_vit_tiny2, gc_vit_small, gc_vit_small2, gc_vit_base, gc_vit_large, gc_vit_large_224_21k, gc_vit_large_384_21k, gc_vit_large_512_21k, 
                   convit_tiny_backbone, convit_small_backbone, convit_base_backbone, RepVGG_A0, RepVGG_A1, RepVGG_A2, RepVGG_B0, RepVGG_B1, RepVGG_B1g2, RepVGG_B1g4, RepVGG_B2, RepVGG_B2g2, 
                   RepVGG_B2g4, RepVGG_B3, RepVGG_B3g2, RepVGG_B3g4, RepVGG_D2se, orthonet34, orthonet50, orthonet101, orthonet152, decouplenet_d0, decouplenet_d1, decouplenet_d2, 
                   sbcformer_xs, sbcformer_s, sbcformer_b, sbcformer_l, fanet_tiny, fanet_small, cosnet_tiny, cosnet_small, cosnet_base, wtconvnext_tiny, wtconvnext_small, wtconvnext_base, 
                   wtconvnext_large, wtconvnext_xlarge, MLLA_Tiny, MLLA_Small, MLLA_Base, pkinet_t, pkinet_s, pkinet_b, glnet_stl, glnet_stl_paramslot, glnet_4g, glnet_9g, glnet_16g,
                   RAVLT_T, RAVLT_S, RAVLT_B, RAVLT_L, slak_tiny, slak_small, slak_base, slak_large, svt_s, svt_b, svt_l, EViT_Tiny, EViT_Small, EViT_Base, EViT_Large, uni_resnet50, 
                   uni_resnet101, sgformer_s, sgformer_m, sgformer_b, spanet_s, spanet_m, spanet_mx, spanet_b, spanet_bx, kw_resnet18, kw_resnet50, StripMLPNet_LightTiny, StripMLPNet_Tiny, 
                   StripMLPNet_Small, StripMLPNet_Base, QARepVGG_A0, QARepVGGV1_A0, QARepVGGV2_A0, QARepVGGV2_A0_d01, QARepVGGV2_A0_DW, QARepVGGV6_A0, QARepVGG_A0_ReLU6, QARepVGGV2_A0_PReLU, 
                   QARepVGGV2_A1, QARepVGGV2_A2, QARepVGGV2_B0, QARepVGGV2_B1, QARepVGGV2_B1g2, QARepVGGV2_B1g4, QARepVGGV2_D2se, identityformer_s12, identityformer_s24, identityformer_s36, 
                   identityformer_m36, identityformer_m48, randformer_s12, randformer_s24, randformer_s36, randformer_m36, randformer_m48, poolformerv2_s12, poolformerv2_s24, poolformerv2_s36, 
                   poolformerv2_m36, poolformerv2_m48, convformer_s18, convformer_s36, convformer_m36, convformer_b36, caformer_s18, caformer_s36, caformer_m36, caformer_b36, iformer_small, 
                   iformer_base, iformer_large, van_b0, van_b1, van_b2, van_b3, van_b4, van_b5, van_b6, vheat_tiny, vheat_small, vheat_base, RepLKNet31B, RepLKNet31L, RepLKNetXL, 
                   LSNet_T, LSNet_S, LSNet_B, StripNet_tiny, StripNet_small, transxnet_tiny, transxnet_small, transxnet_base, parcnetv2_xt, parcnetv2_tiny, parcnetv2_small, parcnetv2_base, 
                   MALA_T, MALA_S, MALA_B, MALA_L, mpvit_tiny, mpvit_xsmall, mpvit_small, mpvit_base, uninext_t, uninext_s, uninext_b, stvit_small, stvit_base, stvit_large, fat_b0, fat_b1, 
                   fat_b2, fat_b3, debi_tiny, debi_small, debi_base, maxvit_tiny, maxvit_small, maxvit_base, maxvit_large, scalable_vit_s, scalable_vit_b, scalable_vit_l, rest_lite, rest_small, 
                   rest_base, rest_large, restv2_tiny, restv2_small, restv2_base, restv2_large}:
            m = m(*args)
            c2 = m.width_list 
            backbone = True
        else:
            c2 = ch[f]
            
        if isinstance(c2, list) and m not in {CBLinear, }:
            backbone = True
            m_ = m
            m_.backbone = True
        else:
            m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i + 4 if backbone else i, f, t  # attach index, 'from' index, type

        # m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # t = str(m)[8:-2].replace("__main__.", "")  # module type
        # m_.np = sum(x.numel() for x in m_.parameters())  # number params
        # m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        # save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        save.extend(x % (i + 4 if backbone else i) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        if isinstance(c2, list) and m not in {CBLinear, }:
            ch.extend(c2)
            for _ in range(5 - len(ch)):
                ch.insert(0, 0)
        else:
            ch.append(c2)
        # ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "ntsmblx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([ntsmblx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([ntsmblx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x).
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([ntsmblx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if m == "pose":
            return "pose"
        if m == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))
        for m in model.modules():
            if isinstance(m, (Segment, MAFSegment, ISegment, DSegment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, (Pose, MAFPose, IPose, DPose)):
                return "pose"
            elif isinstance(m, (OBB, MAFOBB, IOBB, DOBB)):
                return "obb"
            elif isinstance(m, (Detect, MAFDetect, IDetect, DDetect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
