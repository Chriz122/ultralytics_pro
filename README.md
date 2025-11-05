# 使用說明

支援版本
- 建議 ultralytics==8.3.24
- 安裝補充相依套件：
  ```powershell
  pip install -r requirements.txt
  ```
  
此專案提供經過整理與擴充的 ultralytics 模型設定與範例，方便用於本機開發或替換 site-packages 中的 ultralytics 套件設定("C:\Users\USER\AppData\Local\Programs\Python\Python310\Lib\site-packages\ultralytics")。

> [!IMPORTANT]
> 重要提醒：
> - 在替換系統套件前請務必先備份原始資料夾，並確認 Python 版本與相依套件相容。

> [!TIP]
> ## 可搭配 YOLO_tools 的使用說明
> 可以搭配 [YOLO_tools](https://github.com/Chriz122/YOLO_tools) 的 toolbox 訓練、標註處理、評估等工作。

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

## YOLOv4 系列
| 模型名稱                                                                        |                                 改進模組／架構變化（簡述） | 相較原版 YOLO 改進點                | 專長與應用場景             |
| --------------------------------------------------------------------------- | --------------------------------------------: | ---------------------------- | ------------------- |
| `yolov4-p5.yaml` / `yolov4-p6.yaml` / `yolov4-p7.yaml`                      |                   調整輸出層級（P5/P6/P7 分別對應不同金字塔層） | 支援更大/更小尺度的檢測需求（P7 更適合大尺度）    | 根據目標尺度調整（大目標或小目標場景） |
| `yolov4-csp-rtdetr.yaml`                                                    |                             CSP + RT-DETR 解碼頭 | 兼顧 CSP 的效能與 RT-DETR 類解碼效果    | 需穩定精度與較高效率的場景       |
| `yolov4-csp.yaml`                                                           | 使用 CSPDarknet（Cross Stage Partial）作為 backbone | 減少重複計算、提升參數效率與訓練穩定性          | 大型模型訓練效率與推論平衡       |
| `yolov4-mish-rtdetr.yaml`                                                   |                                Mish + RT-DETR | 精度提升與更好的解碼/定位                | 高精度檢測場景             |
| `yolov4-mish.yaml`                                                          |                    使用 Mish 激活函數（相較 ReLU/Leaky ReLU 更平滑） | 更平滑的梯度與更好的特徵表達，常見於高精度模型      | 精度優先場景（可接受較高計算）     |

## YOLOv5 系列
| 模型名稱                                                                    | 改進模組／架構變化（簡述）                                     | 相較原版 YOLO 改進點                               | 專長與應用場景                       |
| ----------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------- | ------------------------------------ |
| `yolov5.yaml`                                                           | YOLOv5 基準（C3, SPPF, PANet head）                          | Ultralytics 實現的標準 YOLOv5                      | 通用檢測任務                         |
| `yolov5-p6.yaml` / `yolov5-p7.yaml` / `yolov5-old-p6.yaml` / `yolov5-old.yaml` / `yolov5-p2.yaml` / `yolov5-p34.yaml` / `yolov5-p6.yaml` / `yolov5-p7.yaml` / `yolov5-PPLCNet.yaml`                                | YOLOv5 大小變體（不同輸入尺寸與層數）               | 不同大小與計算量的折衷               | 根據資源選擇不同大小模型                  |
| `yolov5-AIFI.yaml`                                                      | AIFI（注意力與特徵交互模組）                               | 在骨幹末端引入注意力機制，強化特徵交互與篩選       | 複雜背景或需要精細特徵的場景         |
| `yolov5-AKConv.yaml`                                                    | AKConv（自適應核心卷積）                                   | 取代標準卷積，提升對不同尺寸和形狀物體的適應性     | 小目標或形狀不規則的物體             |
| `yolov5-BoT3.yaml`                                                      | BoT3（Bottleneck Transformer）                             | 在 C3 模組中結合多頭自注意力（MHSA）               | 需要捕捉全域上下文關係的場景         |
| `yolov5-CAConv.yaml`                                                    | CAConv（Coordinate Attention 卷積）                        | 整合座標注意力，強化空間位置與通道關係             | 精細定位、背景複雜場景               |
| `yolov5-CARAFE.yaml`                                                    | CARAFE（內容感知特徵重組上採樣）                           | 使用 CARAFE 模組進行上採樣，更好地恢復細節特徵     | 語義分割／小目標／邊緣精細化場景     |
| `yolov5-CCFM.yaml`                                                      | CCFM（跨通道特徵融合模組）                                 | 改善多通道跨層融合，提升表徵品質                   | 需要強力特徵融合的場景               |
| `yolov5-CNeB-neck.yaml`                                                 | CNeB-neck（跨層網路區塊 Neck）                             | 調整 Neck 結構以改良特徵融合或輕量化               | 尋求效率與精度平衡的場景             |
| `yolov5-CoordAtt.yaml`                                                  | CoordAtt（座標注意力）                                     | 在 C3 模組中加入座標注意力，捕捉方向與位置敏感資訊 | 小目標、位置資訊重要的場景           |
| `yolov5-CPCA.yaml`                                                      | CPCA（通道與位置交叉注意力）                               | 強化通道與位置的交互注意力                         | 複合場景下的精度提升                 |
| `yolov5-CrissCrossAttention.yaml`                                       | Criss-Cross Attention（跨十字注意力）                      | 在骨幹中引入，以更高效的方式捕捉全域上下文資訊     | 大範圍上下文依賴的場景               |
| `yolov5-D-LKAAttention.yaml`                                            | D-LKA（可變形大核心注意力）                                | 結合大感受野與可變形注意力，提升對遠距離與異形物件的表現 | 遠距／異形目標偵測                   |
| `yolov5-DAttention.yaml`                                                | DAttention（動態注意力）                                   | 改善特徵重要性的動態分配                           | 背景干擾較多的情況                   |
| `yolov5-DCNv2.yaml`                                                     | DCNv2（可變形卷積 v2）                                     | 使用可學習的形變採樣位置，改善對物體形狀的適應性   | 非剛性／變形物體偵測                 |
| `yolov5-deconv.yaml`                                                    | Deconv（反卷積上採樣）                                     | 使用反卷積層取代 `nn.Upsample`，可學習上採樣參數   | 需要恢復精細特徵的場景               |
| `yolov5-Dyample.yaml`                                                   | Dyample（動態採樣）                                        | 針對局部特徵做動態抽取，改善表示能力               | 結構複雜物體                         |
| `yolov5-ECAAttention.yaml`                                              | ECA（高效通道注意力）                                      | 在 C3 模組中加入輕量通道注意力，低成本提升性能     | 輕量化且希望提升精度的場景           |
| `yolov5-EffectiveSE.yaml`                                               | EffectiveSE（改良版 SE 注意力）                            | 在 C3 模組中加入，強化通道重加權，計算成本低       | 通用精度提升                         |
| `yolov5-GAMAttention.yaml`                                              | GAM（全域注意力模組）                                      | 類全域注意力，提升跨位置資訊                       | 大範圍依賴情境                       |
| `yolov5-goldyolo.yaml`                                                  | GoldYOLO（Gather-and-Distribute 機制）                     | 引入 `goldyolo` 模組，提升多尺度特徵融合效率       | 追求整體性能提升的場景               |
| `yolov5-hornet-backbone.yaml` / `yolov5-hornet-neck.yaml`               | Hornet 結構（遞迴門控卷積）                               | 使用 `HorNet` 區塊建構骨幹或頸部，提升效率與性能   | 高效能訓練與推論                     |
| `yolov5-l-mobilenetv3s.yaml` / `yolov5-mobile3s.yaml` / `yolov5-Lite-*` |                   MobileNet / Lite 系列 backbone/變體 | 極輕量、低算力部署                   | 手機/嵌入式裝置                      |
| `yolov5-LeakyReLU.yaml`                                                 |                         改變 activation 為 LeakyReLU | 實現較保守的激活選擇（有利於某些收斂）         | 某些資料集訓練穩定度調整                  |
| `yolov5-mobile3s.yaml` / `yolov5-mobilv3l.yaml`                         |                   MobileNetv3 backbone/變體               | 極輕量、低算力部署                   | 手機/嵌入式裝置                      |
| `yolov5-mobileone-backbone.yaml` / `yolov5-MobileOne-Lite-g.yaml` / `yolov5-MobileOne.yaml` | MobileOne 系列 backbone/變體               | 極輕量、低算力部署                   | 手機/嵌入式裝置                      |
| `yolov5-ODConvNext.yaml`                                                | ODConvNext（動態卷積 Next）                                | 引入 `ODConv`，一種對卷積核進行多維度學習的動態卷積 | 複雜特徵需求場景                     |
| `yolov5-RepVGG.yaml` / `yolov5-RepVGG-A1-backbone.yaml`                 | RepVGG（重參數化 VGG）                                     | 訓練時多分支，推理時融合成單一卷積，兼顧精度與速度 | 需要訓練精度與推理效率兼顧的場景     |
| `yolov5-rtdetr.yaml`                                                    | RT-DETR 混合模型                                           | 結合 YOLOv5 骨幹與 RT-DETR 的混合編碼器及解碼頭    | 追求更佳定位穩定性的場景             |
| `yolov5-scal-zoom.yaml`                                                 |                    scale/zoom augmentation 或多尺度策略 | 對各尺度更 robust                | 多尺度資料集適配                      |
| `yolov5-SEAttention.yaml`                                               | SE Attention（Squeeze-and-Excitation）                     | 在 C3 模組中加入 SE 注意力，加強通道間交互         | 通用精度提升                         |
| `yolov5-SegNextAttention.yaml`                                          |                        SegNext-style attention 結構 | 強化分割與細節回復能力                 | Segmentation + Detection 混合任務 |
| `yolov5-ShuffleAttention.yaml` / `yolov5-Shufflenetv2.yaml`             | ShuffleNetV2 / ShuffleAttention 輕量模組                   | 使用 `ShuffleNetV2` 或 `ShuffleAttention` 降低計算成本 | 極限資源場景                         |
| `yolov5-SimSPPF.yaml`                                                   | SimSPPF（簡化版 SPPF）                                     | 結構更簡單的 SPPF，保留多尺度池化但更輕量          | 小幅提升多尺度能力，成本較小         |
| `yolov5-SKAttention.yaml`                                               | SKAttention（選擇性核心注意力）                            | 在 C3 中引入，使網路能自適應地選擇不同大小的卷積核 | 多尺度／形狀變化顯著的場景           |
| `yolov5-SPPCSPC.yaml`                                                   | SPPCSPC（SPP + CSP 變體）                                  | 將 SPP 與 CSP 結合，強化多尺度池化與特徵表徵       | 小物體與表徵穩定性提升               |
| `yolov5-transformer.yaml`                                               | Transformer 模組                                           | 在骨幹末端引入 `TransformerBlock`，提升全域上下文建模 | 需要長距離依賴或複雜場景             |
| `yolov5-TripletAttention.yaml`                                          | TripletAttention（三重注意力）                             | 捕捉跨維度的通道與空間交互                         | 細粒度分類／定位                     |
| `yolov5-VanillaNet.yaml`                                                | VanillaNet（極簡化網路）                                   | 使用 `VanillaNet` 作為骨幹，追求最簡化的架構       | 需要易於部署／調試的場景             |

## YOLOv6 系列
| 模型名稱                                                                                       | 改進模組／架構變化（簡述）                                     | 相較原版 YOLO 改進點                                       | 專長與應用場景                       |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| `yolov6.yaml`                                                                              | YOLOv6 基準（RepOptimizer-style backbone, ConvTranspose upsample） | 以推理和部署優化為導向的架構，使用可重參數化的設計理念     | 通用訓練與邊緣部署                   |
| `yolov6-3.0-p2.yaml` / `yolov6-3.0-p34.yaml` / `yolov6-3.0-p6.yaml` / `yolov6-3.0-p7.yaml` | v6 的不同金字塔輸出設定（P2/P34/P6/P7）                      | 依目標尺度調整輸出層以提升不同尺度表現                     | 小目標／中型／大型目標專向調整       |
| `yolov6-3.0-rtdetr.yaml` / `yolov6-4.0-rtdetr.yaml`                                          | 結合 RT-DETR 解碼頭                                        | 將檢測頭替換為 `RTDETRDecoder`，追求更穩定的定位與分類     | 需要平衡速度與穩定性的場景           |
| `yolov6-4.0-CPCA.yaml`                                                                     | CPCA（通道與位置交叉注意力）                               | 強化通道與位置交互，提升辨識定位                           | 複雜背景或空間資訊重要場景           |
| `yolov6-4.0-CrissCrossAttention.yaml`                                                      | Criss-Cross Attention（跨十字注意力）                      | 引入 Criss-Cross 注意力結構，以更高效的方式捕捉全域上下文  | 大範圍上下文依賴的場景               |
| `yolov6-4.0-D-LKAAttention.yaml`                                                           | D-LKA（可變形大核心注意力）                                | 擴展感受野並自適應物體形狀                                 | 異形或遠距目標偵測                   |
| `yolov6-4.0-DAttention.yaml`                                                               | 動態注意力模組                                             | 動態分配注意力權重，提升對雜訊的魯棒性                     | 背景干擾多的場景                     |
| `yolov6-4.0-GAMAttention.yaml`                                                             | GAM（全域注意力模組）                                      | 全域注意力改善跨位置特徵整合                               | 需要全域上下文的場景                 |
| `yolov6-4.0-SEAttention.yaml`                                                              | SE（Squeeze-and-Excitation）注意力                         | 輕量級通道重加權以提升精度                                 | 想用低成本提升通道表現的場景         |
| `yolov6-4.0-SegNextAttention-obb.yaml`                                                     | SegNeXt 注意力 + 旋轉框（OBB）                             | 結合 SegNeXt 注意力與旋轉框預測，提升旋轉目標檢測性能      | 旋轉目標檢測（如遙感影像）           |
| `yolov6-4.0-ShuffleAttention-obb.yaml`                                                     | ShuffleAttention + 旋轉框（OBB）                           | 輕量級的 ShuffleAttention 結合旋轉框預測                   | 輕量化的旋轉目標檢測                 |
| `yolov6-4.0-SKAttention-obb.yaml`                                                          | SKAttention + 旋轉框（OBB）                                | 多尺度核心注意力結合旋轉框預測                             | 多尺度旋轉目標檢測                   |
| `yolov6-4.0-TripletAttention-obb.yaml`                                                     | TripletAttention + 旋轉框（OBB）                           | 三重注意力結合旋轉框預測，捕捉跨維度交互                   | 細粒度的旋轉目標檢測                 |

## YOLOv7 系列
| 模型名稱                                                                    | 改進模組／架構變化（簡述）                                     | 相較原版 YOLO 改進點                                       | 專長與應用場景                       |
| ----------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| `yolov7.yaml` / `yolov7-x.yaml` / `yolov7-w6.yaml` /  `yolov7-tiny.yaml` / `yolov7-tiny-silu.yaml` / `yolov7-e6e.yaml` / `yolov7-e6.yaml` / `yolov7-d6.yaml`|  YOLOv7 各尺度基準與擴展（不同深度/寬度） | v7 原生改進（CSP-like、更多訓練技巧） | 通用任務，視大小選擇          |
| `yolov7-af-i.yaml`                                                      |         AF-I（輕量化模組） | 更少參數但維持表示能力                       | 行動裝置/邊緣部署         |
| `yolov7-af.yaml`                                                        |         AF（輕量化模組）   | 更少參數但維持表示能力                       | 行動裝置/邊緣部署         |
| `yolov7-C3C2-CPCA.yaml` / `yolov7-C3C2-CPCA-u6.yaml`                                                |       C3C2 模塊（C3 變體、跨 stage 設計） | 提升特徵流/融合效率與表徵能力          | 中型模型/精度提升場景         |
| `yolov7-C3C2-CrissCrossAttention.yaml` / `yolov7-C3C2-CrissCrossAttention-u6.yaml`                                 |       在 C3C2 架構上集成 CrissCrossAttention | 根據任務選擇注意力強化細節            | 針對性任務優化（例如小目標/背景復雜） |
| `yolov7-C3C2-GAMAttention.yaml` / `yolov7-C3C2-GAMAttention-u6.yaml`                                        |       在 C3C2 架構上集成 GAMAttention | 根據任務選擇注意力強化細節            | 針對性任務優化（例如小目標/背景復雜） |
| `yolov7-C3C2-RepVGG.yaml` / `yolov7-C3C2-RepVGG-u6.yaml`                                             |       在 C3C2 架構上集成 RepVGG 模組 | 根據任務選擇注意力強化細節            | 針對性任務優化（例如小目標/背景復雜） |
| `yolov7-C3C2-ResNet.yaml` / `yolov7-C3C2-ResNet-u6.yaml`                                            |       在 C3C2 架構上集成 ResNet 模組 | 根據任務選擇注意力強化細節            | 針對性任務優化（例如小目標/背景復雜） |
| `yolov7-C3C2-SegNextAttention.yaml` / `yolov7-C3C2-SegNextAttention-u6.yaml`                                   |       在 C3C2 架構上集成 SegNextAttention | 根據任務選擇注意力強化細節            | 針對性任務優化（例如小目標/背景復雜） |
| `yolov7-C3C2.yaml` / `yolov7-C3C2-u6.yaml`                                                |       C3C2 模塊（C3 變體、跨 stage 設計） | 提升特徵流/融合效率與表徵能力          | 中型模型/精度提升場景         |
| `yolov7-DCNv2.yaml` / `yolov7-DCNv2-u6.yaml`                           |        Deformable Conv v2 | 更靈活的採樣位置、改良對變形物體的表現      | 非剛性/變形物體檢測          |
| `yolov7-goldyolo.yaml` / `yolov7-goldyolo-u6.yaml` / `yolov7-goldyolo-simple.yaml`                 |          goldyolo（整合優化設計） | 多種提升策略集合，改善 AP 與速度比      | 想要整體提升性能的情境         |
| `yolov7-MobileOne.yaml` / `yolov7-MobileOne-u6.yaml` / `yolov7-tiny-MobileOne.yaml`                 |    MobileOne 輕量化 backbone | 推理速度優化，適合手機/嵌入           | 邊緣設備/移動端            |
| `yolov7-RepNCSPELAN.yaml` / `yolov7-RepNCSPELAN-u6.yaml`                                              |         RepNCSPELAN（複合模塊） | 結合 Rep 設計與 NCSPELAN 類優化  | 兼顧訓練表示與推理效率         |
| `yolov7-rtdetr.yaml` / `yolov7-rtdetr-u6.yaml`                                                   |            RT-DETR 類 head | 改良定位/分類解碼穩定性             | 需要更佳定位一致性的場景        |
| `yolov7-simple.yaml`                                                   |          簡化版 YOLOv7 | 減少參數與計算量，提升速度              | 需要極速推理的場景            |
| `yolov7-tiny-AKConv.yaml`                                               |        AKConv（Adaptive Kernel Convolution） | 自適應卷積核的特徵提取               | 需要靈活卷積核的任務          |
| `yolov7-tiny-goldyolo-simple.yaml`                                      |          goldyolo（整合優化設計） | 多種提升策略集合，改善 AP 與速度比      | 想要整體提升性能的情境         |
| `yolov7-tiny-goldyolo.yaml`                                             |          goldyolo（整合優化設計） | 多種提升策略集合，改善 AP 與速度比      | 想要整體提升性能的情境         |
| `yolov7-tiny-MobileNetv3.yaml`                                          |       MobileNetv3 輕量化 backbone | 輕量化設計，適合手機/嵌入              | 邊緣設備/移動端            |
| `yolov7-tiny-MobileOne.yaml`                                           |    MobileOne 輕量化 backbone | 推理速度優化，適合手機/嵌入           | 邊緣設備/移動端            |
| `yolov7-tiny-PPLCNet.yaml`                                             |        PPLCNet 輕量化 backbone | 輕量化設計，適合手機/嵌入              | 邊緣設備/移動端            |
| `yolov7-tiny-RepNCSPELAN.yaml`                                          |         RepNCSPELAN（複合模塊） | 結合 Rep 設計與 NCSPELAN 類優化  | 兼顧訓練表示與推理效率         |
| `yolov7-tiny-rtdetr.yaml`                                              |            RT-DETR 類 head | 改良定位/分類解碼穩定性             | 需要更佳定位一致性的場景        |
| `yolov7-tiny-simple.yaml`                                              |          簡化版 YOLOv7-tiny | 減少參數與計算量，提升速度              | 需要極速推理的場景            |
| `yolov7-u6.yaml`                                                      |        YOLOv7-u6（大尺度輸入） | 適合高解析度輸入，提升小目標檢測         | 高解析度影像/小目標檢測         |

## YOLOv8 系列
| 模型名稱                                                                                              |                                           改進模組／架構變化（簡述） | 相較原版 YOLO 改進點                     | 專長與應用場景           |
| ------------------------------------------------------------------------------------------------- | ------------------------------------------------------: | --------------------------------- | ----------------- |
| `yolov8.yaml`             |                    YOLOv8 基準（C2f、anchor-free、decoupled head 為常見基礎） | v8 引入 anchor-free 與 decoupled head 等現代化設計      | 通用任務                |
| `yolov8-cls-resnet101.yaml` / `yolov8-cls-resnet50.yaml` / `yolov8-cls.yaml`                       |                                 ResNet101/50 backbone | 強化分類任務的特徵提取                     | 圖像分類任務           |
| `yolov8-ghost.yaml` / `yolov8-ghost-p2.yaml` / `yolov8-ghost-p6.yaml`                             |                             GhostModule/backbone（輕量化模組） | 更少參數但維持表示能力                       | 行動裝置/邊緣部署         |
| `yolov8-rtdetr.yaml`                                                                              |                                          RT-DETR 類 head | 提升定位穩定度                           | 需要更準確邊框的情況        |
| `yolov8-world.yaml` / `yolov8-worldv2.yaml`                                                       |                                 World模型 backbone | 結合多種模組以提升表示能力                   | 高精度任務             |
| `yolov8-AIFI.yaml` | AIFI（Attention-In-Focus Integration） backbone | 注意力引導的特徵提取                       | 需要強注意力機制的任務       |
| `yolov8-AKConv.yaml` | AKConv（Adaptive Kernel Convolution） backbone | 自適應卷積核的特徵提取                     | 需要靈活卷積核的任務       |
| `yolov8-BoT3.yaml` | BoT3（Bottom-up Top-down Transformer） backbone | 自底向上的特徵提取                       | 需要強大的上下文理解能力的任務       |
| `yolov8-C2f-DAttention.yaml` | C2f + DAttention backbone | 結合C2f與動態注意力機制                       | 需要強注意力機制的任務       |
| `yolov8-C2f-DRB.yaml` | C2f + DRB (Dynamic Residual Block) backbone | 結合C2f與動態殘差塊                       | 需要靈活特徵提取的任務       |
| `yolov8-C2f-EMBC.yaml` | C2f + EMBC (Efficient Multi-Branch Convolution) backbone | 結合C2f與高效多分支卷積                       | 需要高效特徵提取的任務       |
| `yolov8-C2f-EMSC.yaml` | C2f + EMSC (Efficient Multi-Scale Convolution) backbone | 結合C2f與高效多尺度卷積                       | 需要多尺度特徵提取的任務       |
| `yolov8-C2f-EMSCP.yaml` | C2f + EMSCP (Efficient Multi-Scale Convolution with Pooling) backbone | 結合C2f與高效多尺度卷積及池化                       | 需要多尺度特徵提取的任務       |
| `yolov8-C2f-FasterBlock.yaml` | C2f + FasterBlock backbone | 結合C2f與FasterBlock                       | 需要高效特徵提取的任務       |
| `yolov8-C2f-GhostModule-DynamicConv.yaml` | C2f + GhostModule + DynamicConv backbone | 結合C2f、GhostModule與動態卷積                       | 行動裝置/邊緣部署         |
| `yolov8-C2f-MSBlockv2.yaml` | C2f + MSBlockv2 backbone | 結合C2f與MSBlockv2                       | 需要多尺度特徵提取的任務       |
| `yolov8-C2f-OREPA.yaml` | C2f + OREPA (Optimized Re-Parameterization) backbone | 結合C2f與OREPA                       | 訓練-部署一體化優化       |
| `yolov8-C2f-REPVGGOREPA.yaml` | C2f + RepVGG + OREPA backbone | 結合C2f、RepVGG與OREPA                       | 訓練-部署一體化優化       |
| `yolov8-C2f-RetBlock.yaml` | C2f + RetBlock backbone | 結合C2f與RetBlock                       | 需要高效特徵提取的任務       |
| `yolov8-C2f-RVB-EMA.yaml` | C2f + RVB (RepVGG Block) + EMA backbone | 結合C2f、RVB與EMA                       | 訓練-部署一體化優化       |
| `yolov8-C2f-RVB.yaml` | C2f + RVB (RepVGG Block) backbone | 結合C2f與RVB                       | 需要高效特徵提取的任務       |
| `yolov8-C2f-Star-CAA.yaml` | C2f + Star-CAA (Criss-Cross Attention) backbone | 結合C2f與Criss-Cross Attention                       | 需要強注意力機制的任務       |
| `yolov8-C2f-StarNet.yaml` | C2f + StarNet backbone | 結合C2f與StarNet                       | 需要強注意力機制的任務       |
| `yolov8-C2f-UniRepLKNetBlock.yaml` | C2f + UniRepLKNetBlock backbone | 結合C2f與UniRepLKNetBlock                       | 需要高效特徵提取的任務       |
| `yolov8-CAConv.yaml` | CAConv (Channel Attention Convolution) backbone | 強化通道注意力的卷積                       | 需要強注意力機制的任務       |
| `yolov8-CNeB-neck.yaml` | CNeB (Cross-Net Block) neck | 強化特徵融合的neck                       | 需要強特徵融合的任務       |
| `yolov8-CoordAtt.yaml` | CoordAtt (Coordinate Attention) backbone | 結合坐標信息的注意力機制                       | 需要強注意力機制的任務       |
| `yolov8-CPAarch.yaml` | CPAarch (Channel and Position Attention Architecture) backbone | 結合通道與位置注意力機制                       | 需要強注意力機制的任務       |
| `yolov8-CPCA.yaml` | CPCA (Channel and Position Cross Attention) backbone | 結合通道與位置交叉注意力機制                       | 需要強注意力機制的任務       |
| `yolov8-CrissCrossAttention.yaml` | Criss-Cross Attention backbone | 強化空間注意力的機制                       | 需要強注意力機制的任務       |
| `yolov8-D-LKAAttention.yaml` | D-LKAAttention (Dynamic Large Kernel Attention) backbone | 結合動態大卷積核注意力機制                       | 需要強注意力機制的任務       |
| `yolov8-DAttention.yaml` | DAttention (Dynamic Attention) backbone | 強化動態注意力的機制                       | 需要強注意力機制的任務       |
| `yolov8-DCNv2.yaml` | DCNv2 (Deformable Convolution v2) backbone | 強化形變卷積的特徵提取                       | 需要靈活特徵提取的任務       |
| `yolov8-deconv.yaml` | Deconvolution neck | 強化上採樣的neck                       | 需要高解析度特徵的任務       |
| `yolov8-DiT-C2f-UIB-FMDI.yaml` | DiT + C2f + UIB (Unified Interaction Block) + FMDI (Feature Multi-Dimension Interaction) backbone | 結合多種模組以提升表示能力                       | 高精度任務             |
| `yolov8-ECAAttention.yaml` | ECAAttention (Efficient Channel Attention) backbone | 高效通道注意力機制                       | 需要強注意力機制的任務       |
| `yolov8-EffectiveSE.yaml` | EffectiveSE (Squeeze-and-Excitation) backbone | 高效SE注意力機制                       | 需要強注意力機制的任務       |
| `yolov8-Faster-Block-CGLU.yaml` | Faster-Block + CGLU (Convolutional Gated Linear Unit) backbone | 結合Faster-Block與CGLU                       | 需要高效特徵提取的任務       |
| `yolov8-Faster-EMA.yaml` | Faster-EMA (Exponential Moving Average) backbone | 結合Faster-Block與EMA                       | 訓練-部署一體化優化       |
| `yolov8-GAMAttention.yaml` | GAMAttention (Global Attention Mechanism) backbone | 強化全局注意力的機制                       | 需要強注意力機制的任務       |
| `yolov8-goldyolo.yaml` | GoldYOLO backbone/變體 | 綜合多種優化以提升精度-速度折衷                       | 高精度任務             |
| `yolov8-hornet-backbone.yaml` | Hornet backbone | 結合多種模組以提升表示能力                       | 高精度任務             |
| `yolov8-hornet-neck.yaml` | Hornet neck | 強化特徵融合的neck                       | 需要強特徵融合的任務       |
| `yolov8-HWD.yaml` | HWD (Hierarchical Weight Decomposition) backbone | 層次化權重分解的特徵提取                       | 需要高效特徵提取的任務       |
| `yolov8-l-mobilenetv3s.yaml` | Lite MobileNetv3s backbone | 輕量化的MobileNetv3s特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-LCDConv.yaml` | LCDConv (Lightweight Contextual Decomposition Convolution) backbone | 輕量化上下文分解卷積                       | 行動裝置/邊緣部署         |
| `yolov8-LeakyReLU.yaml` | LeakyReLU activation backbone | 使用LeakyReLU激活函數                       | 需要非線性激活的任務       |
| `yolov8-Lite-c.yaml` | Lite-c (Lightweight variant configuration) backbone | 輕量化變體                       | 行動裝置/邊緣部署         |
| `yolov8-Lite-g.yaml` | Lite-g (Lite variant with Ghost modules) backbone | 輕量化變體結合Ghost modules                       | 行動裝置/邊緣部署         |
| `yolov8-Lite-s.yaml` | Lite-s (Small lightweight variant) backbone | 輕量化小型變體                       | 行動裝置/邊緣部署         |
| `yolov8-MHSA.yaml` | MHSA (Multi-Head Self-Attention) backbone | 強化多頭自注意力的機制                       | 需要強注意力機制的任務       |
| `yolov8-mobile3s.yaml` | Mobile3s backbone | 輕量化的MobileNetv3特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-mobileone-backbone.yaml` | MobileOne backbone | 輕量化的MobileOne特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-MobileOne.yaml` | MobileOne neck | 強化特徵融合的neck                       | 需要強特徵融合的任務       |
| `yolov8-mobilev3l.yaml` | MobileV3-Large backbone | 輕量化的MobileNetv3 Large特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-MSFM.yaml` | MSFM (Multi-Scale Feature Module) backbone | 強化多尺度特徵的模組                       | 需要多尺度特徵提取的任務       |
| `yolov8-ODConvNext.yaml` | ODConvNext backbone | 強化動態卷積的特徵提取                       | 需要靈活特徵提取的任務       |
| `yolov8-p2.yaml` | P2 (Extra small variant) backbone | 超小型變體                       | 行動裝置/邊緣部署         |
| `yolov8-p34.yaml` | P3/4 (Small-medium variants) backbone | 小型至中型變體                       | 通用任務             |
| `yolov8-p6.yaml` | P6 (Large variant) backbone | 大型變體                       | 高精度任務             |
| `yolov8-p7.yaml` | P7 (Extra large variant) backbone | 超大型變體                       | 需要最高精度的任務         |
| `yolov8-PPLCNet.yaml` | PPLCNet (PaddlePaddle Lite Convolutional Network) backbone | 輕量化的PPLCNet特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-RepNCSPELAN.yaml` | RepNCSPELAN backbone | 結合Rep和NCSPELAN的特徵提取                       | 需要高效特徵提取的任務       |
| `yolov8-RepVGG-A1-backbone.yaml` | RepVGG-A1 backbone | 輕量化的RepVGG-A1特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-RepVGG.yaml` | RepVGG backbone | 輕量化的RepVGG特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-RepViTBlock.yaml` | RepViTBlock backbone | 結合Rep和ViT的特徵提取                       | 需要高效特徵提取的任務       |
| `yolov8-rtdetr.yaml` | RT-DETR head | 提升定位穩定度                       | 需要更準確邊框的情況       |
| `yolov8-SEAttention.yaml` | SEAttention (Squeeze-and-Excitation Attention) backbone | 強化SE注意力的機制                       | 需要強注意力機制的任務       |
| `yolov8-SegNextAttention.yaml` | SegNextAttention backbone | 強化語義分割注意力的機制                       | 需要強語義分割能力的任務       |
| `yolov8-ShuffleAttention.yaml` | ShuffleAttention backbone | 強化通道與空間注意力的機制                       | 需要強注意力機制的任務       |
| `yolov8-Shufflenetv2.yaml` | ShuffleNetv2 backbone | 輕量化的ShuffleNetv2特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-SimAM.yaml` | SimAM (Simple Attention Module) backbone | 簡化的注意力模組                       | 需要強注意力機制的任務       |
| `yolov8-SimSPPF.yaml` | SimSPPF (Simple Spatial Pyramid Pooling Fast) backbone | 簡化的SPPF模組                       | 需要高效特徵提取的任務       |
| `yolov8-SKAttention.yaml` | SKAttention (Selective Kernel Attention) backbone | 強化選擇性卷積核的注意力機制                       | 需要強注意力機制的任務       |
| `yolov8-SPDConv.yaml` | SPDConv (Spatially Pooled Convolution) backbone | 強化空間池化卷積的特徵提取                       | 需要高效特徵提取的任務       |
| `yolov8-SPPCSPC.yaml` | SPPCSPC (Spatial Pyramid Pooling Cross Stage Partial Connections) backbone | 強化SPP與CSP的特徵提取                       | 需要高效特徵提取的任務       |
| `yolov8-StripNet-sn2.yaml` | StripNet-sn2 backbone | 輕量化的StripNet-sn2特徵提取                       | 行動裝置/邊緣部署         |
| `yolov8-SwinTransformer.yaml` | Swin Transformer backbone | 強化局部與全局特徵的Transformer                       | 需要強上下文理解能力的任務       |
| `yolov8-TripletAttention.yaml` | TripletAttention backbone | 強化三重注意力的機制                       | 需要強注意力機制的任務       |
| `yolov8-VanillaNet.yaml` | VanillaNet backbone | 基本的卷積神經網絡                       | 通用任務             |

## YOLOv9 系列
| 模型名稱（代表）                                 | 改進模組／架構變化（簡述）                                               | 相較原版 YOLO 的改進點                      | 專長與應用場景                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov9*.yaml`                                   | v9 基本變體（架構微調）                                                  | 逐步演進的 block/neck 調整，強化穩定性         | 通用任務                     |
| `gelan-c-AKConv.yaml`                           | AKConv（自適應核卷積）                                                  | 更好的局部表示與小目標表現                  | 小目標 / 細節要求場景                 |
| `gelan-c-DCNV3RepNCSPELAN4.yaml`                | DCNv3 + RepNCSPELAN（可形變卷積 + 重參數化/複合模組）                       | 適應變形物體、訓練-推理折衝               | 非剛性物體 / 複雜形狀                 |
| `gelan-c-DualConv.yaml`                         | DualConv（雙路卷積）                                                     | 提升通道/空間訊息分離與融合                 | 背景複雜 / 多尺度                    |
| `gelan-c-FasterRepNCSPELAN.yaml`                | FasterBlock + RepNCSPELAN                                              | 加速同時保留表徵能力                       | 需要高吞吐量但不想犧牲精度            |
| `gelan-c-OREPAN.yaml`                           | OREPA（重參數化 attention/融合）                                         | 訓練強、推理簡化                            | 訓練-部署一體化優化                   |
| `gelan-c-PANet.yaml`                            | PANet（特徵金字塔網路）                                                | 強化多尺度特徵融合                          | 複雜背景 / 多尺度目標                 |
| `gelan-c-SCConv.yaml` / `gelan-c-SPDConv.yaml`  | SCConv / SPDConv（特殊卷積變體）                                         | 改善局部/多尺度特徵抽取                     | 多尺度 / 結構多變物體                 |
| `gelan-s-FasterRepNCSPELAN.yaml`                | s-FasterBlock + RepNCSPELAN（輕量版）                                    | 輕量化同時保留表徵能力                     | 行動端 / 輕量場景                      |
| `gelan-c-dpose.yaml`                             | dpose variant（結合 pose head）                                          | 同時檢測與姿態估計                         | 人體/動物姿態估計                      |
| `gelan-c-dseg.yaml`                             | dseg variant（結合 segmentation head）                                    | 同時檢測與語義分割                         | 語義分割任務                          |

## YOLOv10 系列
| 模型名稱（代表）                                 | 改進模組／架構變化（簡述）                                               | 相較原版 YOLO 的改進點                      | 專長與應用場景                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov10*.yaml`                                  | v10 基本變體（架構微調）                                                  | 逐步演進的 block/neck 調整，強化穩定性         | 通用任務                     |
| `yolov10n-ADNet.yaml`                           | ADNet（專用 attention / decoder 模組）                                   | 改善分類/定位一致性                        | 精度優先但維持一定速度                |
| `yolov10n-ADown.yaml`                           | ADown（自適應降維模組）                                                 | 降低計算量，提升速度                        | 需要速度優化的場景                  |
| `yolov10n-AIFI.yaml`                           | AIFI（自適應特徵融合模組）                                               | 改善多尺度特徵融合                         | 複雜背景 / 多尺度目標                 |
| `yolov10n-AirNet.yaml`                          | AirNet（輕量 backbone / fusion）                                          | 輕量化且保留表徵能力                       | 行動裝置 / 邊緣部署                   |
| `yolov10n-ASF.yaml` / `yolov10n-ASFF.yaml`      | ASF / ASFF（注意力融合模組）                                              | 提升特徵融合與背景抑制                      | 複雜背景 / 低對比度影像               |
| `yolov10n-BiFormer.yaml` / `yolov10n-BiFPN.yaml`| BiFormer / BiFPN（Transformer-like / BiFPN）                             | 更好上下文建模與金字塔融合                  | 需大範圍上下文或多尺度融合             |
| `yolov10n-C2f-CSPHet.yaml`                       | CSPHet（CSP + 異質注意力）                                              | 強化特徵提取與融合                          | 複雜場景 / 多尺度目標                 |
| `yolov10n-C2f-CSPPC.yaml`                        | CSPPC（CSP + 像素注意力）                                              | 提升像素級別的特徵融合                      | 需高解析度輸出的場景                   |
| `yolov10n-C2f-DLKA.yaml`                         | DLKA（深度可變形注意力）                                              | 擴展感受野、提升遠距與異形物體表現           | 遠距 / 異形目標                       |
| `yolov10n-C2f-DWRSeg.yaml`                       | DWRSeg（深度可變形分割）                                              | 提升分割精度與邊界檢測                      | 需要精細分割的場景                     |
| `yolov10n-C2f-GhostModule.yaml`                  | GhostModule（輕量化模組）                                             | 減少計算量，提升速度                        | 需要速度優化的場景                    |
| `yolov10n-C2f-iRMB.yaml`                         | iRMB（增強型重參數化模組）                                           | 提升特徵表徵能力                            | 需強化特徵提取的場景                   |
| `yolov10n-C2f-MLLABlock.yaml`                    | MLLABlock（多層次輕量化模組）                                        | 減少計算量，提升速度                        | 需要速度優化的場景                    |
| `yolov10n-C2f-MSBlock.yaml`                       | MSBlock（多尺度特徵提取模組）                                        | 強化多尺度特徵提取                          | 複雜背景 / 多尺度目標                 |
| `yolov10n-C2f-ODConv.yaml`                        | ODConv（可變形卷積模組）                                            | 擴展感受野、提升遠距與異形物體表現           | 遠距 / 異形目標                       |
| `yolov10n-C2f-OREPA.yaml`                        | OREPA（重參數化模組）                                              | 訓練強、推理簡化                            | 訓練-部署流程優化                      |
| `yolov10n-C2f-RepELAN-high.yaml`                  | RepELAN-high（高效重參數化模組）                                    | 提升特徵表徵能力                            | 需強化特徵提取的場景                   |
| `yolov10n-C2f-RepELAN-low.yaml`                   | RepELAN-low（輕量化重參數化模組）                                  | 減少計算量，提升速度                        | 需要速度優化的場景                    |
| `yolov10n-C2f-SAConv.yaml`                        | SAConv（空間注意力卷積）                                          | 強化空間特徵提取                            | 複雜背景 / 多尺度目標                 |
| `yolov10n-C2f-ScConv.yaml`                        | ScConv（空間卷積）                                                | 提升空間特徵提取                            | 複雜背景 / 多尺度目標                 |
| `yolov10n-C2f-SENetV1.yaml`                       | SENetV1（通道注意力網路 V1）                                      | 提升通道特徵提取                            | 複雜背景 / 多尺度目標                 |
| `yolov10n-C2f-SENetV2.yaml`                       | SENetV2（通道注意力網路 V2）                                      | 提升通道特徵提取                            | 複雜背景 / 多尺度目標                 |
| `yolov10n-C2f-Triple.yaml`                        | Triple（多重特徵融合模組）                                      | 強化多重特徵融合                            | 複雜場景 / 多尺度目標                 |
| `yolov10n-CCFM.yaml`                            | CCFM（Cross-Covariance / Cross-Channel Fusion）                          | 改善通道間交互與表徵質量                    | 複雜場景需強通道融合                   |
| `yolov10n-DAT.yaml`                            | DAT（雙向注意力變體）                                                  | 改善特徵融合與背景抑制                      | 複雜背景 / 低對比度影像               |
| `yolov10n-DLKA.yaml`                           | DLKA（大核 + 可變形注意力）                                              | 擴展感受野、提升遠距與異形物體表現           | 遠距 / 異形目標                       |
| `yolov10n-DynamicConv.yaml`                     | DynamicConv（動態卷積）                                                   | 針對局部特徵自適應卷積                        | 需要自適應局部表徵的場景                |
| `yolov10n-EVC.yaml`                             | EVC（高效卷積）                                                           | 提升卷積運算效率                              | 需要高效運算的場景                      |
| `yolov10n-FFA.yaml`                            | FFA（特徵融合模組）                                                      | 強化特徵融合與表徵能力                        | 複雜場景 / 多尺度目標                   |
| `yolov10n-FocalModulation.yaml`                | FocalModulation（聚焦調製）                                            | 提升對於關鍵區域的特徵提取                    | 需要強調特定區域的場景                  |
| `yolov10n-HAT.yaml`                            | HAT（高效注意力模組）                                                  | 減少計算量，提升速度                          | 需要速度優化的場景                      |
| `yolov10n-HGNet-l.yaml`                        | HGNet-l（輕量級高階特徵網路）                                          | 減少計算量，提升速度                          | 需要速度優化的場景                      |
| `yolov10n-HGNet-x.yaml`                        | HGNet-x（高階特徵網路）                                                | 強化特徵提取與融合                            | 複雜場景 / 多尺度目標                   |
| `yolov10n-IAT.yaml`                            | IAT（影像注意力模組）                                                  | 提升影像特徵提取能力                          | 需要強調影像特徵的場景                  |
| `yolov10n-iRMB.yaml`                           | iRMB（輕量級反向模組）                                                | 減少計算量，提升速度                          | 需要速度優化的場景                      |
| `yolov10n-Light-HGNet-l.yaml`                  | Light-HGNet-l（輕量級高階特徵網路）                                  | 減少計算量，提升速度                          | 需要速度優化的場景                      |
| `yolov10n-Light-HGNet-x.yaml`                  | Light-HGNet-x（輕量級高階特徵網路）                                  | 強化特徵提取與融合                            | 複雜場景 / 多尺度目標                   |
| `yolov10n-LSKA.yaml`                           | LSKA（輕量級空間注意力）                                              | 減少計算量，提升速度                          | 需要速度優化的場景                      |
| `yolov10n-MBformer.yaml`                       | MBformer（輕量級變壓器）                                             | 減少計算量，提升速度                          | 需要速度優化的場景                      |
| `yolov10n-MultiSEAM.yaml`                      | MultiSEAM（多尺度自適應模組）                                       | 強化多尺度特徵提取                            | 複雜場景 / 多尺度目標                   |
| `yolov10n-OREPA.yaml`                          | OREPA（物體重識別與再定位模組）                                     | 提升物體重識別與再定位能力                    | 需要強調物體識別的場景                  |
| `yolov10n-RCSOSA.yaml`                         | RCSOSA（重參數化交叉注意力模組）                                   | 改善特徵融合與表徵質量                        | 複雜場景需強通道融合                   |
| `yolov10n-RepGFPN.yaml`                        | RepGFPN（重參數化特徵金字塔網路）                                   | 提升特徵金字塔的表徵能力                      | 複雜場景 / 多尺度目標                   |
| `yolov10n-RIDNet.yaml`                         | RIDNet（輕量級重識別網路）                                         | 減少計算量，提升速度                          | 需要速度優化的場景                      |
| `yolov10n-SEAM.yaml`                           | SEAM（自適應特徵融合模組）                                       | 強化特徵融合與表徵能力                        | 複雜場景 / 多尺度目標                   |
| `yolov10n-SENetV2.yaml`                        | SENetV2（改進版SENet）                                            | 提升特徵提取與融合能力                        | 複雜場景 / 多尺度目標                   |
| `yolov10n-SlimNeck.yaml`                       | SlimNeck（輕量級頸部網路）                                       | 減少計算量，提升速度                          | 需要速度優化的場景                      |
| `yolov10n-SPDConv.yaml`                        | SPDConv（空間注意力卷積）                                        | 提升卷積運算效率                              | 需要高效運算的場景                      |
| `yolov10n-SPPELAN.yaml`                       | SPPELAN（空間像素級特徵融合模組）                              | 強化空間像素級特徵融合                        | 複雜場景 / 多尺度目標                   |

## YOLOv11 系列
| 模型名稱（代表）                                 | 改進模組／架構變化（簡述）                                               | 相較原版 YOLO 的改進點                      | 專長與應用場景                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov11.yaml`                                   | v11 基本變體（架構微調）                                                  | 逐步演進的 block/neck 調整，強化穩定性         | 通用任務                     |
| `yolov11-ASF.yaml`                               | ASF（自適應融合/注意力）                                                  | 改善多尺度融合與背景抑制                    | 複雜背景 / 小目標                      |
| `yolov11-BiFPN.yaml`                             | BiFPN（雙向特徵金字塔網絡）                                                | 強化多尺度特徵融合                         | 多尺度目標檢測                        |
| `yolov11-C2PSA-CGA.yaml`                         | C2PSA（通道-位置自適應）結合 CGA（通道引導注意力）                       | 強化通道與位置的交互                       | 複雜背景、多物體場景                   |
| `yolov11-C2PSA-DAT.yaml`                         | C2PSA 結合 DAT（雙重注意力 Transformer）                                   | 結合局部與全局注意力提升表示能力             | 複雜背景、多物體場景                   |
| `yolov11-C2PSA-DiT-CCFM.yaml` / `yolov11-C2PSA-DiT.yaml`| C2PSA 結合 DiT（Dual Transformer）( 與 CCFM（Cross-Channel Fusion Module）) | 強化通道與位置的交互，提升全局上下文建模能力 | 複雜背景、多物體場景                   |
| `yolov11-C2PSA-SENetV2-LightHGNetV2-l.yaml` / `yolov11-C2PSA-SENetV2-LightHGNetV2-l-CCFM.yaml`| C2PSA 結合 SENetV2 與 LightHGNetV2-l（輕量化骨幹網絡）( 與 CCFM（Cross-Channel Fusion Module）)         | 輕量化設計，提升通道與位置交互               | 行動端 / 輕量場景                      |
| `yolov11-C3k2-ConvNeXtV2Block-BiFPN.yaml` / `yolov11-C3k2-ConvNeXtV2Block-BiFPN.yaml`| C3k2（新型 block）搭配 ConvNeXtV2 Block 與 BiFPN                      | 提升表徵流動與多尺度融合                    | 需要更強表示能力的中大型模型           |
| `yolov11-C3K2-DiTBlock.yaml` | C3K2 搭配 DiT Block（Dual Transformer Block） | 提升表徵流動與全局上下文建模                | 需要更強表示能力的中大型模型           |
| `yolov11-C3k2-FasterBlock-OREPA-v10Detect.yaml` | C3k2 搭配 FasterBlock 與 OREPA（重參數化注意力） | 提升表徵流動與推理效率                      | 需要更強表示能力的中大型模型           |
| `yolov11-C3k2-MLLABlock-2-SlimNeck.yaml` / `yolov11-C3k2-MLLABlock-2.yaml` | C3k2 搭配 MLLABlock-2 ( 與 SlimNeck ) | 提升表徵流動與推理效率                      | 需要更強表示能力的中大型模型           |
| `yolov11-C3k2-OREPA-backbone-v10Detect.yaml` / `yolov11-C3k2-OREPA-backbone.yaml` | C3k2 搭配 OREPA backbone（重參數化注意力） | 提升表徵流動與推理效率                      | 需要更強表示能力的中大型模型           |
| `yolov11-C3k2-UIB-CCFM.yaml` / `yolov11-C3k2-UIB-FMDI.yaml` / `yolov11-C3k2-UIB.yaml` | C3k2 搭配 UIB（統一交互塊）| 提升表徵流動與通道/位置交互                  | 需要更強表示能力的中大型模型           |
| `yolov11-C3k2-WTConv.yaml` | C3k2 搭配 WTConv（加權卷積）| 提升表徵流動與融合效率                      | 需要更強表示能力的中大型模型           |
| `yolov11-CAFormer.yaml`    | CAFormer（通道注意力 Transformer）                                           | 結合通道注意力與全局上下文建模               | 複雜背景、多物體場景                   |
| `yolov11-CCFM.yaml` / `yolov11-CCFM-C2PSA-DAT.yaml` / `yolov11-CCFM-C2PSA-DAT-v10Detect.yaml`| CCFM 結合 C2PSA / DAT 等複合注意力                                           | 強化通道/位置交互與跨層融合                  | 複雜背景、多物體場景                   |
| `yolov11-ConvFormer.yaml`      | ConvFormer（卷積 + Transformer 混合）                                       | 結合局部卷積與全局注意力                     | 複雜背景、多物體場景                   |
| `yolov11-COSNet.yaml`          | COSNet（通道注意力 + 空間注意力）                                         | 結合通道與空間注意力，提升特徵表徵能力       | 複雜背景、多物體場景                   |
| `yolov11-DecoupleNet.yaml`      | DecoupleNet（解耦頭設計）                                                | 分離分類與回歸任務，提升精度                  | 高精度需求場景                        |
| `yolov11-DiT-C3k2-UIB-CCFM.yaml` | DiT 結合 C3k2、UIB 與 CCFM                                               | 強化通道/位置交互與全局上下文建模             | 複雜背景、多物體場景                   |
| `yolov11-DiT-C3k2-UIB-FMDI-IDetect.yaml` / `yolov11-DiT-C3k2-UIB-FMDI.yaml`  | DiT 結合 C3k2、UIB 與 FMDI（特徵多尺度雙向交互）| 強化通道/位置交互與全局上下文建模             | 複雜背景、多物體場景                   |
| `yolov11-DiT-C3k2-WTConv-CCFM.yaml` | DiT 結合 C3k2、WTConv 與 CCFM                                               | 強化通道/位置交互與全局上下文建模             | 複雜背景、多物體場景                   |
| `yolov11-DiT-CCFM-IDetect.yaml` / `yolov11-DiT-CCFM.yaml`  | DiT 結合 CCFM（Cross-Channel Fusion Module）                                | 強化通道/位置交互與全局上下文建模             | 複雜背景、多物體場景                   |
| `yolov11-DiT.yaml`               | DiT（Dual Transformer）                                                   | 提升全局上下文建模能力                       | 複雜背景、多物體場景                   |
| `yolov11-DySnakeConv.yaml`               | DySnakeConv（動態蛇形卷積）                                               | 提升特徵表徵與流動能力                       | 複雜背景、多物體場景                   |
| `yolov11-EfficientNet-CCFM-v10Detect.yaml` | EfficientNet 結合 CCFM（Cross-Channel Fusion Module）                    | 強化通道/位置交互與全局上下文建模             | 複雜背景、多物體場景                   |
| `yolov11-EfficientNet-OREPA-v10Detect.yaml` | EfficientNet 結合 OREPA（重參數化注意力）                               | 提升表徵流動與推理效率                      | 需要更強表示能力的中大型模型           |
| `yolov11-EfficientNet.yaml`               | EfficientNet                                                               | 提升表徵流動與推理效率                      | 需要更強表示能力的中大型模型           |
| `yolov11-EfficientViM.yaml` / `yolov11-EfficientViT_MIT.yaml` | EfficientViM / EfficientViT_MIT（高效 Transformer）                        | 提升全局上下文建模與效率                     | 複雜背景、多物體場景                   |
| `yolov11-EMOv2.yaml`               | EMOv2（情境感知模組）                                                   | 結合情境感知提升表示能力                     | 複雜背景、多物體場景                   |
| `yolov11-EViT.yaml`               | EViT（Efficient Vision Transformer）                                   | 提升全局上下文建模與效率                     | 複雜背景、多物體場景                   |
| `yolov11-FloraNet.yaml`           | FloraNet（輕量化骨幹網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-FMDI.yaml`               | FMDI（特徵多尺度雙向交互）                                               | 強化多尺度特徵融合                         | 多尺度目標檢測                        |
| `yolov11-GLNet.yaml`             | GLNet（全局-局部網絡）                                                | 結合全局與局部特徵                         | 複雜背景、多物體場景                   |
| `yolov11-hyper.yaml`             | hyper（超參數或特殊結構整合）                                              | 模型架構/訓練策略調整以提升穩定性            | 特定資料集優化                         |
| `yolov11-IdentityFormer.yaml`    | IdentityFormer（輕量化 Transformer）                                      | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-iFormer.yaml`           | iFormer（混合卷積與 Transformer）                                       | 結合局部卷積與全局注意力                     | 複雜背景、多物體場景                   |
| `yolov11-KW_ResNet.yaml`         | KW_ResNet（鍵重參數化 ResNet）                                            | 提升表徵流動與推理效率                      | 需要更強表示能力的中大型模型           |
| `yolov11-LAE.yaml`               | LAE（輕量化注意力增強）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-LAUDNet.yaml`           | LAUDNet（輕量化骨幹網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-LightHGNetV2-l.yaml`    | LightHGNetV2-l（輕量化骨幹網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-LSNet.yaml`             | LSNet（輕量化骨幹網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-Mamba-v10Detect.yaml` / `yolov11-Mamba.yaml`  | Mamba（多尺度注意力模組）                                               | 強化多尺度特徵融合                         | 多尺度目標檢測                        |
| `yolov11-MASF.yaml`              | MASF（多尺度自適應特徵）                                               | 強化多尺度特徵融合                         | 多尺度目標檢測                        |
| `yolov11-MLLA.yaml`             | MLLA（多層次輕量化注意力）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-MobileNetv4.yaml`      | MobileNetv4（輕量化骨幹網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-OverLoCK.yaml`         | OverLoCK（跨層次注意力模組）                                               | 強化跨層次特徵融合                         | 多尺度目標檢測                        |
| `yolov11-PKINet.yaml`           | PKINet（位置關鍵交互網絡）                                               | 強化位置交互表徵                           | 複雜背景、多物體場景                   |
| `yolov11-PoolFormerv2.yaml`    | PoolFormerv2（輕量化池化 Transformer）                                     | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-pst.yaml`              | PST（Pyramid Sparse Transformer，金字塔稀疏 Transformer） | 提升多尺度特徵融合與全局上下文建模效率 | 複雜背景、多物體場景                   |
| `yolov11-QARepVGG.yaml`         | QARepVGG    （量化重參數化 VGG）                                             | 訓練強、推理簡化                            | 需要部署效率且保持高表示能力            |
| `yolov11-RandFormer.yaml`       | RandFormer（隨機注意力 Transformer）                                      | 提升全局上下文建模與效率                     | 複雜背景、多物體場景                   |
| `yolov11-RepLKNet.yaml`         | RepLKNet（重參數化大卷積網絡）                                             | 訓練強、推理簡化                            | 需要部署效率且保持高表示能力            |
| `yolov11-ResNet_MoE.yaml`       | ResNet_MoE（專家混合模型）                                               | 提升模型容量與表徵能力                       | 複雜背景、多物體場景                   |
| `yolov11-RFAConv.yaml`          | RFAConv（重參數化注意力卷積）                                               | 提升表徵流動與推理效率                      | 需要更強表示能力的中大型模型           |
| `yolov11-SFSCNet.yaml`          | SFSCNet（空間頻率選擇卷積網絡）                                           | 提升特徵表徵與流動能力                       | 複雜背景、多物體場景                   |
| `yolov11-SGFormer.yaml`         | SGFormer（稀疏全局注意力 Transformer）                                   | 提升全局上下文建模與效率                     | 複雜背景、多物體場景                   |
| `yolov11-SlabPVTv2.yaml`        | SlabPVTv2（輕量化 Pyramid Vision Transformer）                        | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-SlabSwinTransformer.yaml` | SlabSwinTransformer（輕量化 Swin Transformer）                        | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-SlimNeck.yaml`         | SlimNeck（輕量化頸部設計）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-SMT.yaml`              | SMT（稀疏混合 Transformer）                                               | 提升全局上下文建模與效率                     | 複雜背景、多物體場景                   |
| `yolov11-SPANet.yaml`           | SPANet（空間注意力網絡）                                               | 結合空間注意力提升表示能力                   | 複雜背景、多物體場景                   |
| `yolov11-StripMLPNet.yaml`      | StripMLPNet（條帶化 MLP 網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-StripNet-sn2.yaml`     | StripNet-sn2（條帶化骨幹網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-StripNet.yaml`         | StripNet（條帶化骨幹網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-VAN.yaml`              | VAN（視覺注意力網絡）                                               | 結合通道注意力與全局上下文建模               | 複雜背景、多物體場景                   |
| `yolov11-vHeat.yaml`            | vHeat（輕量化骨幹網絡）                                               | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-WTConvNeXt.yaml`       | WTConvNeXt（加權卷積 + ConvNeXt 混合）                                       | 結合局部卷積與全局注意力                     | 複雜背景、多物體場景                   |
| `yolov11-C2PSA-DiT-C3k2-WTConv-CCFM-pose.yaml` | C2PSA-DiT-C3k2-WTConv-CCFM（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-CoordConv-BiFPN-pose.yaml` | CoordConv-BiFPN（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-EfficientViM-CCFM-pose.yaml` | EfficientViM-CCFM（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-FasterNet-CCFM-pose.yaml` | FasterNet-CCFM（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-GroupMixFormer-CCFM-pose.yaml` | GroupMixFormer-CCFM（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-GSConv-BiFPN-pose.yaml` | GSConv-BiFPN（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-LightHGNetV2-l-CCFM-pose.yaml` | LightHGNetV2-l-CCFM（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-LSNet-CCFM-pose.yaml` | LSNet-CCFM（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-MobileOne-BiFPN-Lite-g-(i)pose.yaml` | MobileOne-BiFPN-Lite-g（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-SlimNeck-BiFPN-pose.yaml` | SlimNeck-BiFPN（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-SwinTransformer-C2PSA-DAT-pose.yaml` | SwinTransformer-C2PSA-DAT（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-SwinTransformer-DiT-pose.yaml` | SwinTransformer-DiT（輕量化姿態估計模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-C3k2-RepVGG-CCFM-seg.yaml` / `yolov11-C3k2-RepVGG-seg.yaml`| C3k2-RepVGG(-CCFM)（輕量化語義分割模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-C3k2-SAConv-seg.yaml` | C3k2-SAConv（輕量化語義分割模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-C3k2-WTConv-CCFM-seg.yaml` | C3k2-WTConv-CCFM（輕量化語義分割模型） | 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |
| `yolov11-Haar-seg.yaml` | Haar（輕量化骨幹網絡）| 輕量化設計，提升效率                         | 行動端 / 輕量場景                      |

## YOLOv12 系列
| 模型名稱（代表）                                 | 改進模組／架構變化（簡述）                                               | 相較原版 YOLO 的改進點                      | 專長與應用場景                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov12.yaml`                                   | v12 基本變體（架構微調）                                                  | 逐步演進的 block/neck 調整，強化穩定性         | 通用任務                     |
| `yolov12-ASF.yaml`                               | ASF（自適應融合/注意力）                                                  | 改善多尺度融合與背景抑制                    | 複雜背景 / 小目標                      |
| `yolov12-CCFM.yaml`                              | CCFM（通道-位置交互）                                                     | 提升通道間交互表徵                         | 需要強通道融合的情境                    |
| `yolov12-hyper.yaml`                             | hyper（超參數或特殊結構整合）                                              | 模型架構/訓練策略調整以提升穩定性            | 特定資料集優化                         |
| `yolov12-ShuffleAttention-CCFM.yaml`             | ShuffleAttention + CCFM                                                   | 輕量注意力提升通道/位置交互                  | 行動端 / 輕量場景                      |
| `yolov12-EMOv2-CCFM-pose.yaml`                   | EMOv2 + CCFM + pose head                                                  | 結合情境感知與姿態估計                       | 人體/動物姿態估計                      |
| `yolov12-TransXNet-CCFM-pose.yaml`               | TransXNet + CCFM + pose head                                              | 結合 Transformer 與姿態估計                  | 高精度姿態估計                         |
| `yolov12-MobileNetv4-CCFM-seg.yaml`              | MobileNetv4 + CCFM + segmentation head                                     | 輕量化與語義分割結合                         | 行動端 / 語義分割                       |
| `yolov12-MobileNetv4-ShuffleAttention-seg.yaml` | MobileNetv4 + ShuffleAttention + segmentation head                        | 輕量注意力與語義分割結合                     | 行動端 / 輕量語義分割                   |

## YOLOv13 系列
| 模型名稱（代表）                                 | 改進模組／架構變化（簡述）                                               | 相較原版 YOLO 的改進點                      | 專長與應用場景                         |
|------------------------------------------------|---------------------------------------------------------------------:|-----------------------------------------|--------------------------------------|
| `yolov13.yaml`                                   | v13 基本變體（架構微調）                                                  | 逐步演進的 block/neck 調整，強化穩定性         | 通用任務，最新迭代                     |
| `yolov13-sn2.yaml`                                | sn2 / 特定 block 變體                                                     | 調整深寬度或 block 類型                    | 根據需求選擇不同深度/效能折衝           |
| `yolov13-pose.yaml`                               | pose variant（結合 pose head）                                            | 同時檢測與姿態估計                         | 人體/動物姿態估計                      |

# models 檔案結構
```
ultralytics_pro\ultralytics\cfg\models
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
│  │      yolov11-cls-pst.yaml
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
│  │           yolov3-spp.yaml
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
│  │      yolov5-LeakyReLU-pose.yaml
│  │      yolov5-Lite-c-pose.yaml
│  │      yolov5-Lite-g-pose.yaml
│  │      yolov5-Lite-s-pose.yaml
│  │      yolov5-mobile3s-pose.yaml
│  │      yolov5-mobileone-backbone-pose.yaml
│  │      yolov5-MobileOne-pose.yaml
│  │      yolov5-mobilev3l-pose.yaml
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
│          yolov5-LeakyReLU-seg.yaml
│          yolov5-Lite-c-seg.yaml
│          yolov5-Lite-e-seg.yaml
│          yolov5-Lite-g-seg.yaml
│          yolov5-Lite-s-seg.yaml
│          yolov5-mobile3s-seg.yaml
│          yolov5-mobileone-backbone-seg.yaml
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
