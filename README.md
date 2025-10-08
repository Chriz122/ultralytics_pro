# 模型介紹
## YOLOv3 系列
| 模型名稱                                                                  |                                                改進模組／架構變化（簡述） | 相較原版 YOLO 改進點                                  | 專長與應用場景               |
| --------------------------------------------------------------------- | -----------------------------------------------------------: | ---------------------------------------------- | --------------------- |
| `yolov3.yaml`                                                         |                 原始 YOLOv3 基本架構（Darknet53 backbone、YOLO head） | 基礎版（基準）                                        | 通用物件檢測                |
| `yolov3-spp.yaml`                                                     |              加入 SPP（Spatial Pyramid Pooling）層於 backbone/neck | 擴大感受野、改善多尺度特徵融合與小物體表現                          | 視野/多尺度複雜場景、小目標偵測      |
| `yolov3-tiny.yaml`                                                    |                                  簡化 backbone 與 head（tiny 結構） | 推論更快、參數小、精度下降                                  | 嵌入式/即時推論資源受限場景        |
| `yolov3-rtdetr.yaml`                                                  | 整合類似 RT-DETR 的 decode/head（將 DETR-like 解碼或輕量 transformer 結合） | 嘗試改善檢測器端的定位/解碼效率、可能提升精度/穩定性                    | 需要更好分類定位一致性的場景；實時取捨優化 |
| `yolov3-spp-rtdetr.yaml`                                              |                                        同時帶 SPP 與 RT-DETR 解碼頭 | 多尺度融合 + 改良解碼，有助於小目標與邊界精準度                      | 小目標＋高定位需求場景           |
| `yolov3-tiny-rtdetr.yaml`                                             |                                        tiny + rtdetr 類型 head | 在極度輕量化上嘗試提升定位/分類品質                             | 超低資源但需稍好精度的場景         |
| `yolov3-spp-seg.yaml` / `yolov3-seg.yaml` / `yolov3-spp-seg.yaml`     |                 SPP + segmentation head（或加入 segmentation 支援） | 從 detection 延伸到 instance/semantic segmentation | 同時需要物件分割與檢測的應用（視覺分析）  |
| `yolov3-obb.yaml` / `yolov3-spp-obb.yaml` / `yolov3-tiny-obb.yaml`    |                                  加入 Oriented Bounding Box 支援 | 能處理旋轉物件（非 axis-aligned）                        | 航拍/工業檢測等旋轉物體常見場景      |
| `yolov3-pose.yaml` / `yolov3-spp-pose.yaml` / `yolov3-tiny-pose.yaml` |                                    加入姿態估計 head / keypoint 支援 | 同時做檢測與姿態(keypoint)預測                           | 人體/動物姿態估計混合場景         |

## YOLOv4 系列
| 模型名稱                                                                        |                                 改進模組／架構變化（簡述） | 相較原版 YOLO 改進點                | 專長與應用場景             |
| --------------------------------------------------------------------------- | --------------------------------------------: | ---------------------------- | ------------------- |
| `yolov4-csp.yaml`                                                           | 使用 CSPDarknet（Cross Stage Partial）作為 backbone | 減少重複計算、提升參數效率與訓練穩定性          | 大型模型訓練效率與推論平衡       |
| `yolov4-csp-rtdetr.yaml`                                                    |                             CSP + RT-DETR 解碼頭 | 兼顧 CSP 的效能與 RT-DETR 類解碼效果    | 需穩定精度與較高效率的場景       |
| `yolov4-mish.yaml`                                                          |                    使用 Mish 激活函數（取代 ReLU/SiLU） | 更平滑的梯度與更好的特徵表達，常見於高精度模型      | 精度優先場景（可接受較高計算）     |
| `yolov4-mish-rtdetr.yaml`                                                   |                                Mish + RT-DETR | 精度提升與更好的解碼/定位                | 高精度檢測場景             |
| `yolov4-p5.yaml` / `yolov4-p6.yaml` / `yolov4-p7.yaml`                      |                   調整輸出層級（P5/P6/P7 分別對應不同金字塔層） | 支援更大/更小尺度的檢測需求（P7 更適合大尺度）    | 根據目標尺度調整（大目標或小目標場景） |
| `yolov4-csp-obb.yaml` / `yolov4-mish-obb.yaml` / `yolov4-obb-p5/p6/p7.yaml` |                                加入 OBB 支援（旋轉盒） | 處理旋轉物件；在 P5~P7 不同尺度可強化特定尺寸物件 | 航拍、遙感、工業檢測          |
| `yolov4-csp-seg.yaml` / `yolov4-mish-seg.yaml`                              |                               segmentation 支援 | 檢測 + 分割整合                    | 物件分割應用              |

## YOLOv5 系列
| 模型名稱                                                                    |                                     改進模組／架構變化（簡述） | 相較原版 YOLO 改進點               | 專長與應用場景                       |
| ----------------------------------------------------------------------- | ------------------------------------------------: | --------------------------- | ----------------------------- |
| `yolov5.yaml`                                                           | YOLOv5 原始/基準（Backbone + SPPF + PANet + YOLO head） | 基準版                         | 通用檢測任務                        |
| `yolov5-p6.yaml` / `yolov5-p7.yaml`                                     |                                    調整金字塔尺度（P6/P7） | 擴大可偵測的尺度範圍                  | 大尺度/多尺度任務                     |
| `yolov5-AIFI.yaml`                                                      |                 AIFI（可能為特定 Attention / Fusion 改進） | 提升特徵融合與注意力能力（針對特定資料優化）      | 複雜背景或局部特徵重要場景                 |
| `yolov5-AKConv.yaml`                                                    |               AKConv（Adaptive Kernel Convolution） | 提升局部感受野自適應性，改善不同尺寸特徵抽取      | 小目標或結構變化大的物體                  |
| `yolov5-BoT3.yaml`                                                      |              BoT3（可能為 Bottleneck-Transformer 類結構） | 結合卷積與輕量 transformer 改善上下文關係 | 需要強全域上下文的場景                   |
| `yolov5-CAConv.yaml`                                                    | CAConv（Coordinate Attention / Context-Aware Conv） | 加強坐標感知 / 通道與空間關係            | 精細定位、背景複雜場景                   |
| `yolov5-CARAFE.yaml`                                                    |                             CARAFE 上採樣模組（內容感知上採樣） | 更好地恢復高解析特徵、提升細節             | 分割/小物體/邊緣精細場景                 |
| `yolov5-CCFM.yaml`                                                      |   CCFM（Cross-Covariance / Cross-Channel Fusion 類） | 改善多通道跨層融合，提升表徵質量            | 需要強融合訊息的場景                    |
| `yolov5-CNeB-neck.yaml`                                                 |                            CNeB-neck（自定義 Neck 模組） | 調整 neck 結構以改良融合或輕量化         | 尋求效率/精度平衡的場景                  |
| `yolov5-CoordAtt.yaml`                                                  |                                   CoordAtt（座標注意力） | 空間 + 通道注意力結合，提高定位與辨識能力      | 小物體、位置信息重要場景                  |
| `yolov5-CPCA.yaml`                                                      |        CPCA（可能為 Channel-Position Cross Attention） | 強化通道與位置交互注意力                | 複合場景下精度提升                     |
| `yolov5-CrissCrossAttention.yaml`                                       |                     Criss-Cross Attention（跨行列注意力） | 更有效的全域上下文信息捕捉               | 大範圍上下文依賴場景                    |
| `yolov5-D-LKAAttention.yaml`                                            |          D-LKA（Deformable Large Kernel Attention） | 大感受野 + 可變形注意力提升遠距離與異形物件表現   | 遠距/異形目標偵測                     |
| `yolov5-DAttention.yaml`                                                |                              DAttn（一般的自注意力或動態注意力） | 改善特徵重要性分配                   | 背景干擾較多情況                      |
| `yolov5-DCNv2.yaml`                                                     |                  DCNv2（Deformable Convolution v2） | 可學習形變採樣位置，改善對物體形狀的適應        | 非剛性/變形物體偵測                    |
| `yolov5-deconv.yaml`                                                    |                              使用 Deconv 上採樣取代一般上採樣 | 更好地恢復細節特徵                   | 分割/小目標精細重建                    |
| `yolov5-Dyample.yaml`                                                   |                                  Dyample（動態採樣/卷積） | 針對局部特徵做動態抽取，改善表示能力          | 結構複雜物體                        |
| `yolov5-ECAAttention.yaml`                                              |                  ECA（Efficient Channel Attention） | 輕量通道注意力，提升性能無大成本            | 輕量化場景希望提升精度                   |
| `yolov5-EffectiveSE.yaml`                                               |                    改良版 SE（Squeeze-and-Excitation） | 強化通道重加權，較少額外計算              | 通用精度提升                        |
| `yolov5-GAMAttention.yaml`                                              |                      GAM（Global Attention Module） | 類全域注意力，提升跨位置資訊              | 大範圍依賴情境                       |
| `yolov5-goldyolo.yaml`                                                  |                                  goldyolo（集成多種優化） | 綜合多種改進提高精度/速度比              | 目標為整體提升的場景                    |
| `yolov5-hornet-backbone.yaml` / `yolov5-hornet-neck.yaml`               |           使用 Hornet 結構（或類似 EfficientNet/Backbone） | 更高效能/吞吐量的 backbone/neck     | 高效能訓練與推論                      |
| `yolov5-l-mobilenetv3s.yaml` / `yolov5-mobile3s.yaml` / `yolov5-Lite-*` |                   MobileNet / Lite 系列 backbone/變體 | 極輕量、低算力部署                   | 手機/嵌入式裝置                      |
| `yolov5-LeakyReLU.yaml`                                                 |                         改變 activation 為 LeakyReLU | 實現較保守的激活選擇（有利於某些收斂）         | 某些資料集訓練穩定度調整                  |
| `yolov5-MobileOne*.yaml` / `yolov5-mobileone-backbone.yaml`             |                                  MobileOne 類輕量化模組 | 在推理上高度優化（NPU/手機）            | 實時邊緣部署                        |
| `yolov5-ODConvNext.yaml`                                                |                     ODConv / ODConvNext（可學習卷積核組合） | 更高表示能力的卷積層                  | 複雜特徵需求場景                      |
| `yolov5-RepVGG.yaml` / `yolov5-RepVGG-A1-backbone.yaml`                 |                               RepVGG（訓練時複雜、推理時簡化） | 訓練期高表徵能力，部署時簡潔高效            | 需要訓練精度與推理效率兼顧的場景              |
| `yolov5-rtdetr.yaml`                                                    |                            RT-DETR style 解碼頭/Head | 改良定位/分類解碼流程                 | 追求定位穩定性的場景                    |
| `yolov5-scal-zoom.yaml`                                                 |                    scale/zoom augmentation 或多尺度策略 | 對各尺度更 robust                | 多尺度資料集適配                      |
| `yolov5-SEAttention.yaml`                                               |             SE Attention (Squeeze-and-Excitation) | 加強通道間交互                     | 通用精度提升                        |
| `yolov5-SegNextAttention.yaml`                                          |                        SegNext-style attention 結構 | 強化分割與細節回復能力                 | Segmentation + Detection 混合任務 |
| `yolov5-ShuffleAttention.yaml` / `yolov5-Shufflenetv2.yaml`             |               ShuffleNet / Shuffle Attention 輕量模組 | 更低成本的注意力或 backbone          | 極限資源場景                        |
| `yolov5-SimSPPF.yaml` / `yolov5-SimSPPF-seg.yaml`                       |                            Simpler SPPF（簡化版 SPPF） | 保留 SPPF 的多尺度池化但更輕量          | 小幅提升多尺度能力，成本小                 |
| `yolov5-SKAttention.yaml`                                               |                       SKNet（Selective Kernel）型注意力 | 自適應核大小融合，增強多尺度適應            | 多尺度/形狀變化場景                    |
| `yolov5-SPPCSPC.yaml`                                                   |                             SPPCSPC（SPP + CSP 變體） | 多尺度池化 + CSP 優勢混合            | 小物體與表徵穩定性提升                   |
| `yolov5-transformer.yaml`                                               |       在 neck/head 或 backbone 引入 Transformer block | 提升全域上下文建模能力                 | 需要長距離依賴或複雜場景                  |
| `yolov5-TripletAttention.yaml` / `yolov5-Triplet-D-LKAAttention.yaml`   |                     Triplet / Triple attention 組合 | 更細緻的空間/通道信息捕捉               | 細粒度分類/定位                      |
| `yolov5-VanillaNet.yaml`                                                |                      使用更“原味”的 CNN pipeline（少額外模組） | 以穩定為主，減少複雜性                 | 需要易於部署/調試的場景                  |

## YOLOv6 系列
| 模型名稱                                                                                       |                                   改進模組／架構變化（簡述） | 相較原版 YOLO 改進點                | 專長與應用場景         |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------: | ---------------------------- | --------------- |
| `yolov6.yaml`                                                                              |                              YOLOv6 基準（v6 原生設計） | 以推理與部署優化為導向（架構/操作優化）         | 通用訓練與邊緣部署       |
| `yolov6-3.0-p2.yaml` / `yolov6-3.0-p34.yaml` / `yolov6-3.0-p6.yaml` / `yolov6-3.0-p7.yaml` |                     v6 的不同金字塔輸出設定（P2/P34/P6/P7） | 依目標尺度調整輸出層以提升不同尺度表現          | 小目標/中型/大型目標專向調整 |
| `yolov6-3.0-rtdetr.yaml`                                                                   |                            結合 RT-DETR 類型解碼/Head | 嘗試取得更穩定定位與分類                 | 需平衡速度與穩定性場景     |
| `yolov6-4.0.yaml`                                                                          |                              YOLOv6 v4 系列（架構迭代） | 可能包括 backbone/neck 的輕量化與效能改良 | 推理效率場景          |
| `yolov6-4.0-CPCA.yaml`                                                                     | CPCA（Channel-Position Cross Attention） or 類似注意力 | 強化通道與位置交互，提升辨識定位             | 複雜背景或空間資訊重要場景   |
| `yolov6-4.0-CrissCrossAttention.yaml`                                                      |                     引入 Criss-Cross Attention 結構 | 更強全域上下文連結                    | 大範圍上下文依賴的場景     |
| `yolov6-4.0-D-LKAAttention.yaml`                                                           |                                    大核心 + 可變形注意力 | 擴展感受野並自適應物體形狀                | 異形或遠距目標偵測       |
| `yolov6-4.0-DAttention.yaml`                                                               |                                         動態注意力模組 | 動態分配注意力權重，提升對雜訊的魯棒性          | 背景干擾多的場景        |
| `yolov6-4.0-GAMAttention.yaml`                                                             |                         Global Attention Module | 全域注意力改善跨位置特徵整合               | 需要全局上下文的場景      |
| `yolov6-4.0-SEAttention.yaml`                                                              |                  SE 或改良版 Squeeze-and-Excitation | 輕量通道重加權以提升精度                 | 想用低成本提升通道表現的場景  |
| `yolov6-3.0-seg.yaml` / `yolov6-3.0-seg-p2/...`                                            |                            segmentation 支援（多尺度） | 檢測 + 分割整合                    | 同時需要分割與檢測任務     |
| `yolov6-3.0-obb-*.yaml`                                                                    |                   支援 Oriented Bounding Box（OBB） | 處理旋轉物件                       | 航拍 / 遙感 / 工業檢測  |

## YOLOv7 系列
| 模型名稱                                                                    |             改進模組／架構變化（簡述） | 相較原版 YOLO 改進點            | 專長與應用場景             |
| ----------------------------------------------------------------------- | ------------------------: | ------------------------ | ------------------- |
| `yolov7.yaml` / `yolov7-x.yaml` / `yolov7-w6.yaml` / `yolov7-d6.yaml` 等 |  YOLOv7 各尺度基準與擴展（不同深度/寬度） | v7 原生改進（CSP-like、更多訓練技巧） | 通用任務，視大小選擇          |
| `yolov7-C3C2-*.yaml`                                                    | C3C2 模塊（C3 變體、跨 stage 設計） | 提升特徵流/融合效率與表徵能力          | 中型模型/精度提升場景         |
| `yolov7-DCNv2.yaml` / `yolov7-DCNv2-u6.yaml`                            |        Deformable Conv v2 | 更靈活的採樣位置、改良對變形物體的表現      | 非剛性/變形物體檢測          |
| `yolov7-RepNCSPELAN.yaml`                                               |         RepNCSPELAN（複合模塊） | 結合 Rep 設計與 NCSPELAN 類優化  | 兼顧訓練表示與推理效率         |
| `yolov7-goldyolo.yaml` / `yolov7-goldyolo-simple.yaml`                  |          goldyolo（整合優化設計） | 多種提升策略集合，改善 AP 與速度比      | 想要整體提升性能的情境         |
| `yolov7-MobileOne.yaml` / `yolov7-tiny-MobileOne.yaml`                  |    MobileOne 輕量化 backbone | 推理速度優化，適合手機/嵌入           | 邊緣設備/移動端            |
| `yolov7-tiny-*`                                                         |              Tiny 系列（極輕量） | 極低參數與運算，精度會下降            | 嚴格資源限制的即時應用         |
| `yolov7-rtdetr.yaml`                                                    |            RT-DETR 類 head | 改良定位/分類解碼穩定性             | 需要更佳定位一致性的場景        |
| `yolov7-C3C2-CPCA.yaml` / `yolov7-C3C2-GAMAttention.yaml` 等             |       在 C3C2 架構上集成不同注意力模組 | 根據任務選擇注意力強化細節            | 針對性任務優化（例如小目標/背景復雜） |
| `yolov7-seg.yaml` 等                                                     |           segmentation 變體 | 檢測 + 分割整合                | 混合視覺任務              |

## YOLOv8 系列
| 模型名稱                                                                                              |                                           改進模組／架構變化（簡述） | 相較原版 YOLO 改進點                     | 專長與應用場景           |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------: | --------------------------------- | ----------------- |
| `yolov8.yaml` / `yolov8-p2.yaml` / `yolov8-p6.yaml`                                               |                    YOLOv8 基準（C2f、decoupled head 等為常見基礎） | v8 帶來更現代化的模組（C2f, decoupled head） | 通用任務，最新YOLO流程     |
| `yolov8-ghost.yaml` / `yolov8-ghost-p2.yaml` / `yolov8-ghost-p6.yaml`                             |                             GhostModule/backbone（輕量化模組） | 更少參數但維持表示能力                       | 行動裝置/邊緣部署         |
| `yolov8-C2f-DRB.yaml` / `yolov8-C2f-EMBC.yaml` / `yolov8-C2f-EMSC.yaml` / `yolov8-C2f-EMSCP.yaml` |   C2f（Cross-Stage Fusion） 結合不同 block（DRB, EMBC, EMSC 等） | 強化跨層信息流動與局部 block 表徵              | 小物體/多尺度融合任務       |
| `yolov8-C2f-FasterBlock.yaml`                                                                     |                                      FasterBlock（改良卷積塊） | 加速的 block 設計以提升吞吐量                | 高 FPS 場景          |
| `yolov8-C2f-GhostModule-DynamicConv.yaml`                                                         |                                    Ghost + Dynamic Conv | 輕量化同時具動態卷積表示                      | 邊緣設備但需較好表徵的情境     |
| `yolov8-C2f-MSBlockv2.yaml`                                                                       |                                          多尺度 MSBlock v2 | 更好的多尺度特徵抽取                        | 小/中/大 目標皆需良好表現    |
| `yolov8-C2f-OREPA.yaml` / `yolov8-C2f-REPVGGOREPA.yaml`                                           | OREPA（可能為 re-parameterized attention / enhanced fusion） | 結合 re-param 與高效融合以改善訓練/推理效率       | 想在推理端取得加速與精度平衡的任務 |
| `yolov8-C2f-REPVGGOREPA.yaml`                                                                     |                                     結合 RepVGG 設計與 OREPA | 訓練中強表示、推理時簡化結構                    | 訓練-部署流程優化場景       |
| `yolov8-C2f-MSBlockv2.yaml`                                                                       |                                         MSBlock 強化多尺度能力 | 多尺度捕捉能力提升                         | 小物體與密集場景          |
| `yolov8-C2f-EMSCP.yaml` / `yolov8-C2f-EMSC.yaml`                                                  |    EMS 類（可能為 Efficient Multi-Scale Convolution/Pooling） | 在成本受限下提升多尺度表徵                     | 邊緣部署但需多尺度表現       |
| `yolov8-C2f-DAttention.yaml` / `yolov8-C2f-DRB.yaml`                                              |                              動態注意力 / 改良 Residual Blocks | 動態分配資源以改善表徵能力                     | 變化大或背景複雜的場景       |
| `yolov8-C2f-FasterBlock.yaml`                                                                     |                                           更快速的 block 結構 | 提升推理速度，降低計算延遲                     | 即時視覺應用            |
| `yolov8-AKConv.yaml` / `yolov8-DCNv2.yaml`                                                        |                             AKConv / DCNv2（自適應捲積/可形變捲積） | 改善對不同形狀物體的適應性                     | 非剛性、形狀多變的偵測任務     |
| `yolov8-DAttention.yaml` / `yolov8-D-LKAAttention.yaml`                                           |                                          注意力模組（含大核/可變形） | 擴展感受野與注意力的表現                      | 遠距/異形目標偵測         |
| `yolov8-EMSC.yaml` / `yolov8-EMSCP.yaml`                                                          |                                          EMS 類模組（高效多尺度） | 在輕量成本下提供多尺度能力                     | 邊緣多尺度任務           |
| `yolov8-OREPA.yaml`                                                                               |                                      OREPA（重參數化或增強融合策略） | 提升訓練表示、推理簡化                       | 訓練-部署一體化優化        |
| `yolov8-transformer.yaml`                                                                         |                    引入 Transformer Block（可在 neck 或 head） | 增強全局上下文建模                         | 複雜場景或需要長距離依賴的任務   |
| `yolov8-rtdetr.yaml`                                                                              |                                          RT-DETR 類 head | 提升定位穩定度                           | 需要更準確邊框的情況        |
| `yolov8-seg.yaml` / `yolov8-seg-p6.yaml`                                                          |                                    segmentation variant | 檢測 + 分割整合                         | 視覺分析 / 產業檢測需分割的場景 |

## YOLOv9 系列

# models 檔案結構樹
ultralytics_pro\cfg\models
│  README.md
│  
├─11
│      yolo11-cls-resnet18.yaml
│      yolo11-cls.yaml
│      yolo11-obb.yaml
│      yolo11-pose.yaml
│      yolo11-RGBIR.yaml
│      yolo11-seg.yaml
│      yolo11.yaml
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
│  │  rtdetr-l.yaml
│  │  rtdetr-resnet101.yaml
│  │  rtdetr-resnet50.yaml
│  │  rtdetr-x.yaml
│  │  
│  └─Detect
│          rtdetr-l.yaml
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
│          
├─v10
│  │  yolov10b.yaml
│  │  yolov10l.yaml
│  │  yolov10m.yaml
│  │  yolov10n.yaml
│  │  yolov10s.yaml
│  │  yolov10x.yaml
│  │  
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
│  │      yolov11-cls.yaml
│  │      
│  ├─Detect
│  │      yolov11-ASF.yaml
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
│  │      yolov11-MLLA.yaml
│  │      yolov11-MobileNetv4.yaml
│  │      yolov11-OREPA-C2PSA-DAT-v10Detect.yaml
│  │      yolov11-OREPA-v10Detect.yaml
│  │      yolov11-OREPA.yaml
│  │      yolov11-OverLoCK.yaml
│  │      yolov11-PKINet.yaml
│  │      yolov11-PoolFormerv2.yaml
│  │      yolov11-QARepVGG.yaml
│  │      yolov11-RandFormer.yaml
│  │      yolov11-RepGFPN.yaml
│  │      yolov11-RepLKNet.yaml
│  │      yolov11-ResNet_MoE.yaml
│  │      yolov11-RFAConv.yaml
│  │      yolov11-SFSCNet.yaml
│  │      yolov11-SGFormer.yaml
│  │      yolov11-SlabPVTv2.yaml
│  │      yolov11-SlabSwinTransformer.yaml
│  │      yolov11-SlimNeck.yaml
│  │      yolov11-SMT.yaml
│  │      yolov11-SPANet.yaml
│  │      yolov11-StripMLPNet.yaml
│  │      yolov11-StripNet-sn2.yaml
│  │      yolov11-VAN.yaml
│  │      yolov11-vHeat.yaml
│  │      yolov11-WTConvNeXt.yaml
│  │      yolov11.yaml
│  │      新文字文件.txt
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
├─v3
│  │  yolov3-spp.yaml
│  │  yolov3-tiny.yaml
│  │  yolov3.yaml
│  │  
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
│  │  yolov5-p6.yaml
│  │  yolov5.yaml
│  │  
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
│  │  yolov6.yaml
│  │  
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
│  │  yolov8-cls-resnet101.yaml
│  │  yolov8-cls-resnet50.yaml
│  │  yolov8-cls.yaml
│  │  yolov8-ghost-p2.yaml
│  │  yolov8-ghost-p6.yaml
│  │  yolov8-ghost.yaml
│  │  yolov8-obb.yaml
│  │  yolov8-p2.yaml
│  │  yolov8-p6.yaml
│  │  yolov8-pose-p6.yaml
│  │  yolov8-pose.yaml
│  │  yolov8-rtdetr.yaml
│  │  yolov8-seg-p6.yaml
│  │  yolov8-seg.yaml
│  │  yolov8-world.yaml
│  │  yolov8-worldv2.yaml
│  │  yolov8.yaml
│  │  
│  ├─Classify
│  │      yolov8-AIFI-cls.yaml
│  │      yolov8-cls-p2.yaml
│  │      yolov8-cls-p6.yaml
│  │      yolov8-cls.yaml
│  │      yolov8-RepVGG-cls.yaml
│  │      
│  ├─Detect
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
│  │  yolov9c-seg.yaml
│  │  yolov9c.yaml
│  │  yolov9e-seg.yaml
│  │  yolov9e.yaml
│  │  yolov9m.yaml
│  │  yolov9s.yaml
│  │  yolov9t.yaml
│  │  
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
└─yolo-world
        yolov8-world.yaml
        yolov8-worldv2.yaml
        
