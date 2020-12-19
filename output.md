この記事は私, wataokaが一人で2020年の**1年間をかけて**作り続けた論文要約の**超大作記事**です.

# 論文100本解説

## Disparate Interactions: An Algorithm-in-the-Loop Analysis of Fiarness in Risk Assessments

wataokaの日本語訳「異なる相互作用: リスク評価に関するFairnessの分析ループ内のアルゴリズム」
- 種類: fairness
- 学会: FAT2020 Best paper
- 日付: 2018/11/20
- URL: [https://scholar.harvard.edu/files/19-fat.pdf](https://scholar.harvard.edu/files/19-fat.pdf)


### 概要

機械学習の決定に関するFairnessではなく, 機械学習の出力を人間がどう解釈するかが重要. 参加者を雇って, criminal judgeをしてもらった. そして, disparate interactionsを観測した. (リスク評価を使用することで黒人にリスク予測が高くなる)

### 手法

disparate interactionsに避けるためにalgorithm-in-the-loopを提案. human-in-the-loopはアルゴリズを向上させるために人を加えることだが, algorithm-in-the-loopは逆.

## Equality of Epportunity in Supervised Learning

wataokaの日本語訳「教師あり学習の機会平等」
- 種類: fairness
- 日付: 2016/10/07
- URL: [https://ttic.uchicago.edu/~nati/Publications/HardtPriceSrebro2016.pdf](https://ttic.uchicago.edu/~nati/Publications/HardtPriceSrebro2016.pdf)


### 概要

Equal Opportunityを提案した論文.

### 手法

「異なるセンシティブ属性間においてprecisionが等しい」がequal opportunity

## Learning Non-Discriminatory Predictors

wataokaの日本語訳「無差別な予測器の学習」
- 種類: fairness
- 日付: 2017/11/01
- URL: [https://arxiv.org/pdf/1702.06081.pdf](https://arxiv.org/pdf/1702.06081.pdf)


### 概要

公平性の汎化バウンドはモデルの複雑さに依存しなくさせることができることを示した論文.

## Flexibly Fair Representation Learning by Disentanglement

wataokaの日本語訳「Disentanglementによる柔軟な表現学習」
- 種類: fairness
- 学会: ICML2019
- 日付: 2019/06/06
- URL: [https://arxiv.org/abs/1906.02589](https://arxiv.org/abs/1906.02589)


### 概要

VAEでdisentanglementした表現を獲得. その表現を用いて, flexibly fairな応用を行う.

### 手法

非センシティブな表現zとセンシティブな表現bに切り分ける. 任意のbiとbjは独立, bとzは独立, bとセンシティブ属性は相互情報量を最大化させるようにobjectを設計した.

### wataokaのコメント

埋め込まれたsensitive latent, bはsensitive attribute, aを予測できるように埋め込まれている. うまくdisentangleされたbのうち, 特定のsensitive attributeを消すということはその属性を予測されないようにしているということ. つまりdemographic parity最適化.

## Contional Learning of Fair Representations

wataokaの日本語訳「公平表現の条件付き学習」
- 種類: fairness
- 学会: ICLR2020
- 日付: 2019/10/16
- URL: [https://arxiv.org/abs/1910.07162](https://arxiv.org/abs/1910.07162)


### 概要

BERがaccuracy parityとequalized oddsを同時に最適化できることを証明.さらに, EOが満たされた時, BERはそれぞれのグループのupper boudとなることを証明. しかもdemographic parityも守られる. 

## Learning Certified Individually Fair Representations

wataokaの日本語訳「個人公平性が保証された表現の学習」
- 種類: fairness
- 日付: 2020/02/24
- URL: [https://arxiv.org/abs/2002.10312](https://arxiv.org/abs/2002.10312)


## NestedVAE: Isolating Common Factors via Weak Supervision

wataokaの日本語訳「NestedVAE: 半教師あり学習による共通因子の分離」
- 種類: fairness
- 日付: 2020/02/26
- URL: [https://arxiv.org/abs/2002.11576](https://arxiv.org/abs/2002.11576)


## Fair Division of Mixed Divisible and Indivisible Goods

wataokaの日本語訳「割り切れるグッズと割り切れないグッズの公平な区分」
- 種類: fairness
- 学会: AAAI2020


### 概要

Fair division問題におけるEnvy-Freenessを拡張したEnvy-Freeness for Mixed goodsを提案し, その割り当てアルゴリズムも提案している.

## Price of Fairness in Budget Division and Probabilistic Social Choice

wataokaの日本語訳「予算区分と確率的社会選択における公平の価格」
- 種類: fairness
- 学会: AAAI2020


## Learning Fair Naive Bayes Classifiers by Discovering and Eliminating Discrimination Patterns

wataokaの日本語訳「差別パターンを発見し, 除去することによる公平なナイーブベイズ分類器の学習」
- 種類: fairness
- 学会: AAAI2020


### 概要

この研究では部分的観測をするナイーブベイズ分類器を考える. あるセンシティブ属性が観測されるかどうかで生まれる差別を紹介. 

### 手法

modelがfairになるまで差別パターンを発見し, 除去していくアプローチをした. 結果, 簡単な制約を加えるだけで多くの差別パターンを削除することができた.

## Fairness in Network Representation by Latent Structural Heterogeneity in Observational Data

wataokaの日本語訳「観測データにある潜在的な構造の不均一性によるネットワーク表現の公平性」
- 種類: fairness
- 学会: AAAI2020


### 概要

network representation learningにおけるfairnessの論文.

### 手法

Mean Latent Similarity Discrepancy(MLSD)という測度を提案. MLSDは構造的不均一性に対して敏感であるノード表現における差異を計算する.  (Figure1. みたいなのが構造的不一致に対して敏感なノード表現)

## Ren´yi Fair Inference

wataokaの日本語訳「レーニ公平推論」
- 種類: fairness
- 学会: AAAI2020


## White-box Fairness Testing through Adversarial Sampling

wataokaの日本語訳「敵対的サンプリングを用いたホワイトボックス公平性テスト」
- 種類: fainress
- 学会: ICSE2020
- URL: [https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5635&context=sis_research](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5635&context=sis_research)


### 概要

個人差別を探索するための手法を提案. 勾配を用いた探索で, 既存手法よりスケーラブルで軽量な探索方法となっている.

## Π-nets: Deep Polynomial Neural Networks

wataokaの日本語訳「Π-nets: ディープ多項式ニューラルネットワーク」
- 種類: general
- 学会: CVPR2020
- URL: [https://arxiv.org/abs/2003.03828](https://arxiv.org/abs/2003.03828)


### 概要

pi-netというCNNに変わる手法を提案. 出力は入力の高次元多項式. pi-netは日線形関数無しにDCNNよりいい表現を獲得でいる. 活性化関数を用いると画像生成でSOTA. また, このフレームワークでなぜStyleGANがうまくいったかがわかる.

### 手法

pi-netは特別な種類のskip connectionで実装され, それらのパラメータは高いオーダーのテンソルを通して表現される.

## Cost-Sensitive BERT for Generalisable Sentence Classification with Imbalanced Data

wataokaの日本語訳「不均衡データの一般化可能な文書分類のためのコストセンシティブBERT」
- 種類: nlp, imbalance
- 学会: NLP4IF2019
- 日付: 20200316
- URL: [https://arxiv.org/abs/2003.11563](https://arxiv.org/abs/2003.11563)


### 概要

BERTはdata augmentation無しに不均衡なクラスを処理できるが, trainとtestでデータが似ていない場合は一般化できないことを示し, その場合の対処法を提案.

### 手法

ウィルコクソンの符合順位検定を用いてp値を計算することで二つのsetの類似度を比較し, 一定値以上は不均衡データに対応できていないことを示した. コストセンシティブな手法は, misclassificationしたクラスの重要度をますように設計.

### 結果

propaganda fragment-level identificationとpropagandadistic sentence-level identificationの2つのタスクに適応させた.

## Domain Adaptation by Class Centroid Matching and Local Manifold Self-Learning

wataokaの日本語訳「クラス重心マッチングと局所多様体自己学習によるドメイン適応」
- 種類: domain adaptation, manifold learning
- 日付: 20200320
- URL: [https://arxiv.org/abs/2003.09391](https://arxiv.org/abs/2003.09391)


### 概要

ドメインのデータ分布構造を強調するためにsourceとtargetドメインそれぞれのクラスの重心を使用して, target dataに擬似ラベルを割り当てるドメイン適応を提案した. 

### 手法

domain adaptationをclass centoidマッチング問題に置き換えた.

## Deep Nets for Local Manifold Learning

wataokaの日本語訳「局所多様体学習のためのディープネット」
- 種類: manifold learning
- 日付: 20180529
- URL: [https://www.frontiersin.org/articles/10.3389/fams.2018.00012/pdf](https://www.frontiersin.org/articles/10.3389/fams.2018.00012/pdf)


### 概要

局所多様体学習に関する論文.

## When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks

wataokaの日本語訳「NASが堅牢性を満たす時: 敵対的攻撃に対して堅牢なアーキテクチャ探索」
- 種類: nas, adversarial attacks
- 学会: CVPR2020
- 日付: 20191125
- URL: [https://arxiv.org/abs/1911.10695](https://arxiv.org/abs/1911.10695)


### 概要

敵対的攻撃に堅牢なアーキテクチャを研究したかったので, アーキテクチャ探索を用いた.

### 結果

密に結合されたパターン, 計算力が少ない場合はdirect connection edgeにCNNを追加する, flow of solution procedure(FSP) matrixなどが効果的だった.

## Fairness Is Not Static:
Deeper Understanding of Long Term Fairness
via Simulation Studies

wataokaの日本語訳「Fairnessは静的では無い: シミュレーション研究による長期公平性のより深い理解」
- 種類: fairness
- 学会: ACM FAT2020


### 概要

機械学習の研究に用いられるデータセットは静的だが, 実際は長期にわたる動的な問題であると主張. 

### 手法

シミュレーションでagent, environment, metricsの関係をタイムステップ毎に解析した.

### wataokaのコメント

ソースコードが公開されている. https://github.com/google/ml-fairness-gym

## Wasserstein Fair Classification

wataokaの日本語訳「ワッサースタイン公平分類」
- 種類: fairness
- 日付: 20190728


### 概要

ワッサースタイン距離の最小化がStrong Demographic Parityの最小化と等しいことを証明.

## Obtraining Fairness using Optimal Transport Theory

wataokaの日本語訳「最適輸送理論を用いた公平性の獲得」
- 種類: fairness, statistics
- 日付: 20180608


## Counterfactual Fairness

wataokaの日本語訳「反実仮想公平性」
- 種類: fairness, counterfactual
- 学会: NIPS2017
- 日付: 20170320
- URL: [https://arxiv.org/abs/1703.06856](https://arxiv.org/abs/1703.06856)


### 概要

Counterfactual Fairnessを定義した論文. 
1つの観測に対してセンシティブ属性を介入した時のモデルの出力分布が同じならCounterfactually Fair.
また, それを実現するためのアルゴリズムも提案.

### 手法

非観測な潜在背景変数UとAの非子孫なXのみで学習を行う. (因果モデルは完璧に与えられているので, UをMCMCサンプリングすることができる)

### 結果

LowScroolでの成績を予測するデータセットを使用. knowledgeというUを仮定して, Uをサンプリングし, 学習した. 因果モデルが正しければCFは達成できるからか, 提案アルゴリズムの介入毎の分布は記載されていないが, 精度が落ちてしまっていることは記載されている.

### wataokaのコメント

提案アルゴリズムはCounterfactual Fairnessの十分条件を満たしてる感じがしてやりすぎ感がある. というかそもそも因果モデルが完璧に与えられてるのが無理がある. (しかし, 定義論文なので理想形から論じるのは仕方がない)

## Avoiding Discrimination through Causal Reasoning

wataokaの日本語訳「因果関係を用いた差別の回避」
- 種類: fairness, causal inference
- 学会: NIPS2017
- 日付: 20170608
- URL: [https://arxiv.org/abs/1706.02744](https://arxiv.org/abs/1706.02744)


### 概要

観測データに基づく公平指標には限界があるので, 因果ベースで考える必要がある. 因果ベースで考えることによって, 観測データに基づく指標がいつ,なぜ失敗するのかを明らかできる. 因果に基づく指標を提案し, それを守るアルゴリズムを開発した.

## When Worlds Cllide: Integrating Different Counterfactual Assumptions in Fairness

wataokaの日本語訳「世界が衝突するとき: 公平性における異なる反事実な仮定の統合」
- 種類: fairness, counterfactual
- 学会: NIPS2017
- 日付: 20171204
- URL: [https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf](https://papers.nips.cc/paper/7220-when-worlds-collide-integrating-different-counterfactual-assumptions-in-fairness.pdf)


### 概要

全ての因果関係を検証することは不可能なので, 複数の因果モデルに対して, どちらの因果の世界が正しいかではなく, どちらが公平な判断を提供するかを考えることが望ましい. 本論文では, 同時に一度の因果モデルに対して公平な予測ができる方法を提案する.

## Causal Reasoning for Algorithmic Fairness

wataokaの日本語訳「アルゴリズムの公平性のための因果関係」
- 種類: fairness, causal inference
- 日付: 20180315
- URL: [https://arxiv.org/abs/1805.05859](https://arxiv.org/abs/1805.05859)


### 概要

既存のfairness手法のレビューし, あらゆるfairアプローチにとって因果的アプローチが必要であることを議論した. また, 近年の因果ベース公平性へのアプローチを詳細に解析した.

## Path-Specific Counterfactual Fairness

wataokaの日本語訳「パス特有の反実仮想公平性」
- 種類: fairness, counterfactual
- 学会: AAAI2019
- 日付: 20180222
- URL: [https://arxiv.org/abs/1802.08139](https://arxiv.org/abs/1802.08139)


### 概要

Path-Specific Counterfactual Fairnessを提案. unfairなpathwayに沿った影響を無視する因果手法 

### 手法

センシティブ属性によって悪影響を及ぼす観測を修正し, 使用する. ディープラーニングと近似推論で非線形なシナリオに適用.

### 結果

AdultとGermanで実験.

## Counterfactual Fairness: Unidentification, Bound and Algorithm

wataokaの日本語訳「反実仮想公平性: Unidentificationとboundとアルゴリズム」
- 種類: fairness, counterfactual
- 学会: IJCAI2019
- 日付: 20190810
- URL: [https://www.ijcai.org/Proceedings/2019/0199.pdf](https://www.ijcai.org/Proceedings/2019/0199.pdf)


### 概要

反事実の値が特定的でない場合, counterfactual fairnessは限界がある(計算しにくい). 非特定的な反事実の値を数学的にboundした. また, counterfactually fairな分類器を構築するための理論的に正しいアルゴリズムを開発した.

### 手法

τ-Counterfactual Fairnessを定義し, これが閾値を超えないように学習する.

## Counterfactual fairness: removing direct effects through regularization

wataokaの日本語訳「反実仮想公平性: 正則化による直接的影響の削除」
- 種類: fairness, counterfactual
- 日付: 20200226
- URL: [https://arxiv.org/pdf/2002.10774.pdf](https://arxiv.org/pdf/2002.10774.pdf)


### 概要

Controlled Direct Effect (CDE)を用いて因果関係を取り入れた新しい公平性の定義を提案.

## Deep Learning for Causal Inference

wataokaの日本語訳「因果推論のためのディープラーニング」
- 種類: causal inference


## Estimating Causal Effects Using Weighting-Based Estimators

wataokaの日本語訳「重み付けベース推定器を用いた因果効果の推定」
- 種類: causal inference
- 学会: AAAI2020
- 日付: 20191100
- URL: [https://causalai.net/r54.pdf](https://causalai.net/r54.pdf)


## WHERE IS THE INFORMATION IN A DEEP NETWORK?

wataokaの日本語訳「ディープネットワークの情報はどこにある?」
- 種類: general
- 学会: ICLR2019
- URL: [https://openreview.net/pdf?id=BkgHWkrtPB](https://openreview.net/pdf?id=BkgHWkrtPB)


### 概要

DNNが過去のデータから収集した情報は全て重みでエンコードされる. その情報は未知データに対するDNNの反応にどう影響するのかは未解決問題. 実際, DNN内の情報の定義の仕方や測り方でさえ曖昧.


## Estimating individual treatment effect: generalization bounds and algorithms

wataokaの日本語訳「個人介入効果の推定: 汎化誤差とアルゴリズム」
- 種類: causal inference
- 学会: ICML2017
- 日付: 20160613
- URL: [https://arxiv.org/abs/1606.03976](https://arxiv.org/abs/1606.03976)


### 概要

TARNetを提案. ITE(個人介入効果)を推定する.  さらに, 介入分布とコントロール分布の距離と汎化誤差を用い, 推定されたITEの誤差のboundをとった.

### 手法

φ(x)から介入t=0 or t=1で分岐し, それぞれで損失をとる. mata

### wataokaのコメント

CEVAEに参考にされている.

## Causal Generative Neural Networks

wataokaの日本語訳「因果生成ニューラルネットワーク」
- 種類: causal inference
- 日付: 20171124
- URL: [https://arxiv.org/abs/1711.08936](https://arxiv.org/abs/1711.08936)


### 概要

CGNNを提案. 因果構造を発見する.

### 手法

観測データ分布と生成データ分布のMMDを最小化する.

### wataokaのコメント

code: https://github.com/GoudetOlivier/CGNN

## CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training

wataokaの日本語訳「CausalGAN: 敵対的学習を用いた明示的な因果生成モデルの学習」
- 種類: causal inference, gan
- 学会: ICLR2018
- 日付: 20170914
- URL: [https://arxiv.org/abs/1709.02023](https://arxiv.org/abs/1709.02023)


### 概要

CGAN (CausalGAN)を提案

### wataokaのコメント

code: https://github.com/mkocaoglu/CausalGAN

## FairGAN: Fairness-aware Generative Adversarial Networks

wataokaの日本語訳「公平なGAN」
- 種類: fairness, gan
- 学会: IEEE ICBD2019
- 日付: 20180328
- URL: [https://arxiv.org/abs/1805.11202](https://arxiv.org/abs/1805.11202)


### 概要

FGAN (FairGAN)を提案

### 手法

G:(z, s)→(x, y)
D1:(x, y)→real or fake
D2:(x, y)→s=0 or s=1
という構成で敵対的学習するだけ.

## Achieving Causal Fairness through Generative Adversarial Networks

wataokaの日本語訳「GANを用いた因果公平性の達成」
- 種類: causal inference, gan, fairness
- 学会: IJICAI2019
- 日付: 20190816
- URL: [https://pdfs.semanticscholar.org/1846/bb80fbd235bcf3316b5ffb09a6d3e22ebeab.pdf](https://pdfs.semanticscholar.org/1846/bb80fbd235bcf3316b5ffb09a6d3e22ebeab.pdf)


### 概要

CFGAN (Causal Fairness GAN)を提案. 与えられた因果関係グラフに基づいて様々な因果関係の公平性を確保しながら分布を学習できる.

### 手法

CFGANは因果グラフ, 介入グラフの構造を作る2つのgを持つ. 2つのgは因果モデルと介入後の因果モデルを生成できる.  2つのdはgが実際に近い分布を生成するため, そしてgによってシミュレートされた因果量が公平になるために使用される.

### 結果

Adultを用いて実験.

## The Variational Fair Autoencoder

wataokaの日本語訳「変分公平オートエンコーダー」
- 種類: fairness, vae
- 学会: ICLR2016
- 日付: 20151103
- URL: [https://arxiv.org/abs/1511.00830](https://arxiv.org/abs/1511.00830)


### 概要

VFAEを提案.

## Causal effect inference with deep latent-variable models

wataokaの日本語訳「ディープ潜在変数モデルを用いた因果効果推論」
- 種類: causal inference, vae
- 学会: NIPS2017
- 日付: 20170324
- URL: [https://arxiv.org/abs/1705.08821](https://arxiv.org/abs/1705.08821)


### 概要

CEVAE (Causal Effect VAE)を提案. Individual Causal Effectを推論するタスクにおいてSOTA.

### 手法

xからq(t|x)を推論し, q(y|t,x)を推論し, q(z|t,y,x)を推論する. zからp(x|z)を推論し, p(t|z)を推論し, p(y|t,z)を推論する. 精度が上がったp(y|t,z)で介入による差異を計算することで, 介入効果を計算する.

### 結果

人工データと二つのread data(IHDPとJobs)で検証. 

### wataokaのコメント

FCVAEに引用されている. 理論的保証はないが, empiricalにSOTAを達成して黙らせた感じ. 

## Fairness Through Causal Awareness: Learning Latent-Variable Models for Biased Data

wataokaの日本語訳「因果レベルでの公平性: バイアスデータのための潜在変数モデルの学習」
- 種類: fairness, causal inference, vae
- 学会: FAT2019
- 日付: 20180907
- URL: [https://arxiv.org/abs/1809.02519](https://arxiv.org/abs/1809.02519)


### 概要

CFVAE (Causal Fairness VAE)を提案.
センシティブ属性を交絡因子として考えたことで, ヒストリカルバイアスデータセットにおける因果効果の推論精度が上がった.

## CausalVAE: Structured Causal Disentanglement in Variational Autoencoder

wataokaの日本語訳「因果VAE: VAEによる構造化因果のDisentanglement」
- 種類: cusal inference, vae
- 日付: 20200418
- URL: [https://arxiv.org/abs/2004.08697](https://arxiv.org/abs/2004.08697)


### 概要

CVAE(Causal VAE)を提案. 潜在空間上のzの因果関係を探索する. 識別可能性にまで言及している.

### 手法

z上で因果関係を組みたい. z=Az+ε. εの部分をencoderで推論する. Aの探索はcontinuous constraint functionを用いる.

### 結果

振り子の人工データセットと水槽の人工データセットとCelebAで実験.

## DAGs with NO TEARS: Continuous Optimization for Structure Learning

wataokaの日本語訳「NO TEARSのDAGs: 構造学習のための連続最適化」
- 種類: causal inference
- 学会: NIPS2018
- 日付: 20180304
- URL: [https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf](https://papers.nips.cc/paper/8157-dags-with-no-tears-continuous-optimization-for-structure-learning.pdf)


### 概要

NP困難であるDAGの構造学習を実数行列上の連続最適化問題に定式化した.

### 手法

h=tf(e^{W○W})-1とすると, G(W)∈DAGs <=> h(W)=0となる. hは簡単に微分可能なので連続最適化ができる.

## Visual Causal Feature Learning

wataokaの日本語訳「視覚的な因果関係の学習」
- 種類: causal inference
- 学会: UAI2015
- 日付: 20141207
- URL: [https://arxiv.org/abs/1412.2309](https://arxiv.org/abs/1412.2309)


### 概要
- 行動の視覚的因果の定義を提案する.
- 行動の視覚的因果は人間, 動物, ニューロン, ロボット, その他の知覚システムにおける視覚的に駆動される行動に応用可能.
- 提案フレームワークは因果変数がミクロ変数から構成されている必要があるような標準的な因果学習を一般化する.
- Causal Coarsening(因果関係の粗大化)定理を証明した.
- この定理により最小限の実験による観測データから因果知識を得ることができる.
- この定理は標準的な因果推論のテクニックを画像特徴を識別する機械学習に接続する. (画像特徴はターゲットの行動を引き起こすわけではないがターゲットの行動と相関がある.)
- 最終的にターゲットの行動の視覚的因果を自動的に特定するために画像におけるoptimal manipulationsを行うmanipulator関数を学習するためのactive learningスキームを提供する.
- 人工データとreal dataで実験した.

### 1. 導入
Three advances
画像空間を作るミクロ変数(pixels)から構成されるマクロ変数としてのターゲットの行動の視覚的因果を定義した. 視覚的因果は画像内にターゲットの行動に関する因果情報があるという点で他のマクロ変数と異なる.
Causal Coarsening Theorem (CCT)を証明。CCTは最小限の実験的努力から視覚的因果を学習するために観測データをどのように使えるかを示してくれる.
manipulator関数を学習する方法を示す. manipulator関数は自動的に視覚的因果においてoptimal manipulationsを実行してくれる.

### 2. A Theory of visual causal features
2.2 Generative Models: from micro to macro variables
Definition 1 (Observational Partition, Observational Class)
振る舞いTに関する集合Iのobservational partition Π_o(T, I) (⊂I)は,
i~j <=> P(T|I=i)=P(T|I=j)
という等価関係に基づく分割である.

画像のobservational partitionを知ることはTの値を予測することになる.

#### Definition2 (Visual Manipulation)
visual manipulationは操作man(I=i)である. 


#### Definition 3 (Causal Partition, Causall Class)
振る舞いTに関する集合IのCausal partition Π_o(T, I) (⊂I)は,
	i~j <=> P(T|man(I=i)) = P(T|man(I=j))
という等価関係に基づく分割である.

### スライド
https://www.slideshare.net/KojinOshiba/visutl-causal-feature-learning

目的: 人間の視覚的な因果を理解すること
マクロ変数(e.g. ピクセルの集合)から因果学習するフレームワーク
ミクロ変数(e.g. 聴覚や嗅覚データ)への応用可能

C: macro-variable
I: image 
T: behavior
H: discrete variable (Iを生成する)

## Discovering Causal Signals in Images

wataokaの日本語訳「画像における因果信号の発見」
- 種類: causal inference
- 学会: CVPR2017
- 日付: 20150526
- URL: [https://arxiv.org/abs/1605.08179](https://arxiv.org/abs/1605.08179)


### 概要

現実世界ではcar→wheelという出現関係があり, 画像に現れると仮定している. そのようなcausal dispositionを明らかにするような方法を提案した.

### 手法

画像からobjectとcontextに分離, それぞれをcausal featureなのかanticausalなのかに二値分類

## When Causal Intervention Meets Adversarial Examples and Image Masking for Deep Neural Networks

wataokaの日本語訳「因果介入はいつCNNのための敵対的事例と画像マスキングを満たすのか」
- 種類: causal inference,
adversarial attacks
- 学会: IEEE ICIP 2019


## Explaining Visual Models by Causal Attribution

wataokaの日本語訳「因果属性による視覚モデルの説明」
- 種類: causal inference, gan
- 学会: ICCV2019 Workshop
- URL: [https://arxiv.org/abs/1909.08891](https://arxiv.org/abs/1909.08891)


### wataokaのコメント

上の論文を引用している.

## Inclusive FaceNet: Improving Face Attribute Detection with Race and Gender Diversity

wataokaの日本語訳「包括的なFaceNet: 人種や性別に多様な顔属性検知の向上」
- 種類: causal inference
- 学会: FAT/ML 2018
- 日付: 20171201
- URL: [https://arxiv.org/abs/1712.00193](https://arxiv.org/abs/1712.00193)


### 概要

問題設定: 性別において, 顔の属性分類の精度を合わせる.

### 結果

データセット: Faces of the World, CelebA

## Towards Fairness in Visual Recognition: Effective Strategies for Bias Mitigation

wataokaの日本語訳「視覚認識の公平性に向けて: バイアス除去のための効果的な戦略」
- 種類: causal inference


### 概要

問題設定: 性別において, 顔の属性分類の精度を合わせる.

### 結果

データセット: CIFAR-10S, CelebA

## Face Recognition Performance: Role of Demographic Information

wataokaの日本語訳「顔認識のパフォーマンス: 人口統計情報の役割」
- 種類: causal inference
- 学会: TIFS2012


### 概要

問題設定: 性別において, 顔認識の精度を合わせる.

## Balanced Datasets Are Not Enough: Estimating and Mitigating Gender Bias in Deep Image Representation

wataokaの日本語訳「バランスの取れたデータセットは十分でない: ディープ画像表現における性別バイアスの推測と除去」
- 種類: causal inference
- 学会: CVPR2019 Workshop


### 概要

問題設定: 画像内に映る人物の性別によって, ラベル分類精度を合わせる.

### 結果

データセット: MSCOCO, imSitu

## SensitiveNets: Learning Agnostic Representations with Application to Face Recognition

wataokaの日本語訳「SensitiveNets: 顔認識の応用とagnostic表現の学習」
- 種類: fairness
- 学会: CVPR2019 Workshop
- 日付: 20190201
- URL: [https://arxiv.org/abs/1902.00334](https://arxiv.org/abs/1902.00334)


### 概要

問題設定: 顔認識, 画像分類において, 潜在空間で性別情報を消した上で精度を下げない.

### 結果

データセット: CelebA, VGGFace2, LFW

## Discovering Fair Representations in the Data Domain

wataokaの日本語訳「データドメインにおける公平表現の学習」
- 種類: fairness
- 学会: CVPR2019


### 概要

問題設定: 画像から性別を判断できない画像を生成する.

### 結果

データセット: CelebA, Diversity in Faces, Adult income (タブラー)

## What does it mean to ‘solve’ the problem of discrimination in hiring?

wataokaの日本語訳「採用差別を'解決'するとはどういうことか？」
- 種類: fairness
- 学会: ACM FAT2020 Best Paper  (SSH/LAW/EDU/PE)


## Weakly Supervised Disentanglement with Guarantees

wataokaの日本語訳「保証付き弱い教師ありdisentanglement」
- 種類: disentanglement
- 学会: ICLR2020
- URL: [https://arxiv.org/pdf/1910.09772.pdf](https://arxiv.org/pdf/1910.09772.pdf)


### wataokaのコメント

disentanglementをしっかり定義したらしい. ちゃんと読んでない.

## Progressive Growing of GANs for Improved Quality, Stability, and Variation

wataokaの日本語訳「質, 安定性, 多様性を向上させるためのPG-GAN」
- 種類: GAN
- 学会: ICLR2018
- URL: [https://arxiv.org/abs/1710.10196](https://arxiv.org/abs/1710.10196)


### 概要

画像などの解像度が高くなると生成分布のランダム要素が強くなるので, 学習が不安定になる. そこでPGGANを提案した.

### 手法

Progressive Growing: 徐々に解像度を上げる. minibatch discrimination: ミニバッチ全体の統計量を計算し, それぞれの画像に反映させる. equalized learning rate: wiによって学習率を変えない. pixelwise feature vec normalization in generator: まぁfeature vecの正規化.

### wataokaのコメント

参考記事1 

## CAN: Creative Adversarial Networks, Generating "Art" by Learning About Styles and Deviating from Style Norms

wataokaの日本語訳「CAN: 敵対的創造ネット, スタイルノルムからスタイルと多様を学習することで芸術を生成する.」
- 種類: GAN
- 学会: ICCC2017
- 日付: 20170621
- URL: [https://arxiv.org/abs/1706.07068](https://arxiv.org/abs/1706.07068)


### 概要

アートを生成するCANを提案. 
与えられたart distributinoからは外れないように, styleは逸脱するように学習する.

### 結果

CANが生成したものとアーティストが作ったものを人間に見せた反応を比べた.

## Cost-Effective Incentive Allocation via Structured Counterfatual Inference

wataokaの日本語訳「反実仮想推論による費用対効果の高いインセンティブ配分」
- 種類: counterfactual
- 学会: AAAI2020
- URL: [https://aaai.org/Papers/AAAI/2020GB/AAAI-LopezR.787.pdf](https://aaai.org/Papers/AAAI/2020GB/AAAI-LopezR.787.pdf)


### 概要

従来のpolicy最適化フレームワークとは違った, 報酬構造と予算の制約を考慮に入れるという反実仮想policy最適化問題を解く手法を提案した.

## Evaluating the Disentanglement of Deep Generative Models through Manifold Topology

wataokaの日本語訳「多様体トポロジーを用いた深層生成モデルのdisentanglementの評価」
- 種類: disentanglement
- 日付: 20200605
- URL: [https://arxiv.org/abs/2006.03680](https://arxiv.org/abs/2006.03680)


### 概要

学習された表現の条件付きsubmanifoldsのトポロジカルな類似度を測定することで, disentanglementを定量化する手法を提案.

### wataokaのコメント

全然理解できてない.

## Understanding image representations by measuring their equivariance and equivalence

wataokaの日本語訳「同変性と等価性を測定することによる画像表現の理解」
- 種類: general
- 学会: CVPR2015
- 日付: 20141121
- URL: [https://arxiv.org/abs/1411.5908](https://arxiv.org/abs/1411.5908)


### 概要

equivariance:入力画像の変換がどう埋め込まれるか. invariance:その変換が影響を与えいない. equivalence:CNNの2つの異なるパラメータが同じ情報を見ているか. これらの特性を確立するための方法はあるが, どの層で達成されているかなどを見る.

## Group Equivariant Convolutional Networks

wataokaの日本語訳「群同変な畳み込みネット」
- 種類: general
- 学会: ICML2016
- 日付: 20160224
- URL: [https://arxiv.org/abs/1602.07576](https://arxiv.org/abs/1602.07576)


## Gauge Equivariant Convolutional Networks and the Icosahedral CNN

wataokaの日本語訳「ゲージ同変な畳み込みネットと20面体CNN」
- 種類: general
- 学会: ICML2019
- 日付: 20190211
- URL: [https://arxiv.org/abs/1902.04615](https://arxiv.org/abs/1902.04615)


### 概要

対象変換に対する等変量の原理により, 理論的根拠のあるニューラルネットの設計ができる. (equivariant networks). この論文では, この原理が大域的な対称性を超えて, 局所的なゲージ変換にまで拡張できることを示した.

### 手法

球体の合理的な近似でらう正20面体の表面上で定義された信号に対してゲージ等変なCNNを実装した. 単一のconv2d呼び出しを用いてゲージ等変畳み込みを実装することができ, 球形CNNに変わる非常にスケーラブルな手段となる.

### 結果

全方位画像や地球規模の気候パターンのセグメンテーションタスクにおいて従来手法を大きく上回った.

## Invertible Conditional GANs for image editing

wataokaの日本語訳「画像編集のための可逆な条件付きGAN」
- 種類: GAN
- 学会: NeurIPS2016
- 日付: 20161119
- URL: [https://arxiv.org/abs/1611.06355](https://arxiv.org/abs/1611.06355)


### 概要

cGANで条件付け生成はできたが, 実画像に対する画像編集はできない. IcGANsはencoderを利用することでそれを可能にした.

### 手法

属性付き顔画像データセットを用いて, G(z, y), zを推論するE_z, yを推論するE_yを学習させる. 学習が終わったら, 推論されたyを編集することで画像編集ができる.

## GANalyze: Toward Visual Definitions of Cognitive Image Properties

wataokaの日本語訳「GANalyze: 認知的画像特性の視覚的定義に向けて」
- 種類: GAN
- 学会: ICCV2020
- 日付: 20190624
- URL: [https://arxiv.org/abs/1906.10112](https://arxiv.org/abs/1906.10112)


### 概要

surpervised editing. 記憶性, 美的性, 感情的価値性などの認知特性を高めるように潜在空間でwalkする方法を実験した.

### 手法

AをMemNetなどの, 画像から特性を出力する評価関数とした時, 以下を最小化するθを見つける.
A(G(z+αθ)) - A(G(z))+α

### 結果

画像の記憶性が物体の大きさとして表面化したことを示した.

### wataokaのコメント

website: http://ganalyze.csail.mit.edu/

## Interpreting the Latent Space of GANs for Semantic Face Editing

wataokaの日本語訳「意味論的顔編集のためのGANの潜在空間の解釈」
- 種類: GAN
- 学会: CVPR2020
- 日付: 20190725
- URL: [https://arxiv.org/abs/1907.10786](https://arxiv.org/abs/1907.10786)


### 概要

supervised editing. InterFaceGANを提案. GANの潜在空間を解釈するために, 顔属性編集を行える.

### 手法

GANとEncoderを用意し, 属性付き顔画像を潜在空間に埋め込む. 埋め込まれたベクトルをある属性で分離する超平面をSVMで作り, その法線方向にzを移動させることで, その属性に関して編集できる.

### 結果

かなりうまく編集できている. 条件付き編集も提案しており, 固定したい属性を固定もできている.

### wataokaのコメント

Detecting Bias with Generative Counterfactual Face Attribut Augmentationの拡張というか理論というか

## Controlling generative models with continuous factors of variations

wataokaの日本語訳「変動の連続的な要因による生成モデルの制御」
- 種類: GAN
- 学会: ICLR2020
- 日付: 20200128
- URL: [https://arxiv.org/abs/2001.10238](https://arxiv.org/abs/2001.10238)


### 概要

semi-supervised editing. 鳥とかきのことかを編集している. pixel-wiseな損失では高周波成分をうまく再構成できていないと怒っていた.

### 手法

あまり理解できていないが, L(G(z), T(I))を最小化するようなzを探索し, 損失が少ないやつを逐次的に採用していく感じ. 特徴的なところとして, Lとして低周波数成分に注目する損失値を使用している.

## Unsupervised Discovery of Interpretable Directions in the GAN Latent Space

wataokaの日本語訳「GANの潜在空間における解釈可能方向の教師なし発見」
- 種類: GAN
- 日付: 20200210
- URL: [https://arxiv.org/abs/2002.03754](https://arxiv.org/abs/2002.03754)


### 概要

unsupervised editing. 背景削除の方向を見つけたらしい. sacliency detectionでSOTA.

### 手法

[1~K]のどれかkをone-hotエンコードし, ε倍する. (d×Kの行列A) * (ε倍されたK次元one-hotエンコードe)でε倍されたAの列を一つ取り出す. その列をzに足した奴がshift. zとz+A(εe)をreconstructor Rに入力し, kとεを予測. AとRの精度を上げることで, 見分けが追記やすいshiftになる.

### 結果

MNIST, AnimeFace, Imagenet, CelebA-HQで実験した. saliency detectionでSOTA.

### wataokaのコメント

code: https://github.com/anvoynov/GanLatentDiscovery, まだarXiv論文だがどこかにacceptされそう.

## PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models

wataokaの日本語訳「PULSE: 生成モデルの潜在空間探索によるself-supervised高画質化」
- 種類: GAN, superresolution
- 学会: CVPR2020
- 日付: 20200308
- URL: [https://arxiv.org/abs/2003.03808](https://arxiv.org/abs/2003.03808)


### 概要

高解像(HR)なし, 低解像(LR)のみで超解像(SR)を作成できるPULSEを提案.

### 手法

学習済みGはz→自然画像多様体というmapをしてくれるという仮定の下, ||DS(G(z)) - LR||を最小化するzを探索する. 勾配が得られるので勾配ベース. 探索の工夫としては, 超球面の表面上のみで探索を行う制約をかけてzの尤度をあげている.

### wataokaのコメント

website: http://pulse.cs.duke.edu/

## ON THE “STEERABILITY” OF
GENERATIVE ADVERSARIAL NETWORKS

wataokaの日本語訳「GANの「操縦性」について」
- 種類: GAN
- 学会: ICLR2020
- 日付: 20200311
- URL: [https://openreview.net/pdf?id=HylsTT4FvB](https://openreview.net/pdf?id=HylsTT4FvB)


### 概要

semi-supervised editing. GANはデータセットバイアス(e.g.物体が中心にくる)に影響されているが, 潜在空間で「steering」することで, 現実的な画像を作成しながら分布を移動することができる.

### 手法

edit(G(z),α) - G(z+αw)を最小化するw(walk)を勾配探索で見つける. editは計算できる編集(zoomとか)なので, 編集自体に意味はない. 論文としては, steerabilityが高い属性はその属性に対するデータセットのバイアスが少ないと言えることから, この操縦に意味を見出している.

### wataokaのコメント

paper, poster, codeがある
↓
 https://ali-design.github.io/gan_steerability/

## Detecting Bias with Generative Counterfactual Face Attribute Augmentation

wataokaの日本語訳「反実な顔属性の生成増強によるバイアスの検知」
- 種類: fairness, gan,counterfactual
- 学会: ICML2019 Workshop
- 日付: 20190618
- URL: [https://arxiv.org/abs/1906.06439](https://arxiv.org/abs/1906.06439)


### 概要

InterFaceGANの前進で, ほとんど同じ. 顔属性の編集を行う研究. InterFaceGANのように条件付き編集には言及してないが, Counterfactual imageの考え方はInterFaceGANに無い部分.

### 手法

InterFaceGANと同じ手法で属性をいじり, 生成された画像をcounterfactual imageだと見なす. 元画像と反実画像を分類器に推論させることで, counterfactual fairnessを測定できる.

### 結果

属性ベクトルで画像操作した際に, どれほど予測結果が変化するかを, 属性毎に調べた.

### wataokaのコメント

他の手法との精度の比較が全くない.

## Counterfactual Image Network

wataokaの日本語訳「反実画像ネットワーク」
- 種類: counterfactual
- 学会: ICLR2018
- 日付: 20180216
- URL: [https://openreview.net/pdf?id=SyYYPdg0-](https://openreview.net/pdf?id=SyYYPdg0-)


### 概要

「うまくセグメンテーションできる」=「選択範囲を削除しても自然な画像」という仮定の下, うまくcounterfactualな画像(オブジェクトを消した物)を生成できるようにし, セグメンテーションの精度を上げている.

### 手法

入力画像からzに埋め込み, K個のレイヤーでオブジェクトを削除. 一定確率pで結合し, Discriminatorがreal dataかどうかを判断する.

## Counterfactual Visual Explanations

wataokaの日本語訳「反実的視覚説明」
- 種類: counterfactual
- 学会: PMLR2019
- URL: [https://arxiv.org/abs/1904.07451](https://arxiv.org/abs/1904.07451)


### 概要

画像の一部を変化させると別クラスに分類されるという可視化を行うことによって画像の分類根拠を説明するという試み.

### 手法

分類器を前半fと後半gにわけ, 2つのサンプルそれぞれからfで抽出される特徴マップのどこを入れ替えたら別クラスに分類されるのかという情報からAttentionで可視化する.

### 結果

野鳥分類を用いた.

## Explaining Image Classifiers by Counterfactual Generation

wataokaの日本語訳「反実仮想生成による画像分類の説明」
- 種類: counterfactual
- 学会: ICLR2019
- URL: [https://arxiv.org/abs/1807.08024](https://arxiv.org/abs/1807.08024)


### 概要

FIDOというフレームワークを提案. 消された場所を生成的に補強する.

### wataokaのコメント

ソースコードが公開されている.

## FACE: Feasible and Actionable Counterfactual Explanations

wataokaの日本語訳「FACE: 実現可能で実用可能な反実仮想説明」
- 種類: counterfactual
- 学会: AAAI2020
- 日付: 20190920
- URL: [https://arxiv.org/abs/1909.09369](https://arxiv.org/abs/1909.09369)


### 概要

反実仮想説明システムにおいて, 実現不可能なアドバイスがされることがある. それを解決するために, FACEという反実仮想説明のアルゴリズムを提案.

### 手法

他のサンプルが近くに存在する経路を通りながら別クラスの領域に移動させていくことで, 不自然な変換を行わないようにしている.

### 結果

 人工トイデータとMNISTで実験した. 0→8への変換の時, 0でも8でもない画像が生まれない.

## Open Set Learning with Counterfactual Images

wataokaの日本語訳「反実画像によるOpenSet学習」
- 種類: counterfactual
- 学会: ECCV2019
- URL: [http://web.engr.oregonstate.edu/~lif/3090.pdf](http://web.engr.oregonstate.edu/~lif/3090.pdf)


### 概要

open set(unkownクラスあり)学習をするためにcounterfactualを考慮した.

## MemNet: A persistent Memory Network for Image Recognition

wataokaの日本語訳「MemNet: 画像認識のための永続的メモリーネットワーク」
- 種類: general
- 学会: ICCV2017
- 日付: 20170807
- URL: [https://arxiv.org/abs/1708.02209](https://arxiv.org/abs/1708.02209)


### 概要

MemNetを提案. 再起的ユニットとゲートユニットからなるメモリブロックを持ち, 前のメモリブロックからの出力と表現を連結してゲートユニットにろくり, ゲートユニットは前の状態をどれだけ保存し, 現在の状態をどれだけ保存するのかを決定する.

### 結果

超解像などに適用した.

## Examining CNN Representations With Respect To Dataset Bias

wataokaの日本語訳「データセットバイアスに関するCNN表現の検証」
- 種類: general
- 学会: AAAI2018
- 日付: 20171029
- URL: [https://arxiv.org/abs/1710.10577](https://arxiv.org/abs/1710.10577)


### 概要

CNNが共起パターンを学習し, ある属性の有無をその属性と関係のない部分から判断していることを発見する手法を提案. (例) 顔画像においてリップをしているかどうかを判断するためにアイシャドウをみている.

### 手法

属性間の関係性をデータから学習し, 真の関係と矛盾するなら表現に問題ありとする.

### wataokaのコメント

真の関係にバイアスがないことを前提としたequal opportunity的な手法.

## README: REpresentation learning by fairness-Aware Disentangling MEthod

wataokaの日本語訳「README: 公平なdisentangleによる表現学習」
- 種類: fairness
- 日付: 20200707
- URL: [https://arxiv.org/abs/2007.03775](https://arxiv.org/abs/2007.03775)


### 概要

protected attribute, target attribute(分類ラベル), mutual attributeそれぞれに関する情報に別れるように潜在変数を埋め込むFD-VAEを提案.

## Latent Space Factorisation and Manipulation via Matrix Subspace Projection

wataokaの日本語訳「行列部分空間射影を用いた潜在空間分解と編集」
- 種類: disentanglement
- 学会: ICML2020
- 日付: 20190726
- URL: [https://arxiv.org/abs/1907.12385](https://arxiv.org/abs/1907.12385)


### 概要

学習済みオートエンコーダへのプラグインとして利用できる潜在変数を分解方法MSP(Matrix Subspace Projection)を提案.

### 手法

xをEncoderでzに落とし, zをHでz^に変換し,  z^にMをかけてy^とする. また, z^にMと直交する行列をかけてs^とする. y^はyを再構成できるよう, s^は情報を持たないようにノルムを最小化するように損失関数を設計し, Mを探索する.

### wataokaのコメント

website: https://xiao.ac/proj/msp
code: https://github.com/lissomx/MSP

## Gender Slopes: Counterfactual Fairness for Computer Vision Models by Attribute Manipulation

wataokaの日本語訳「Gender Slopes: 属性編集を用いた画像モデルのための反実仮想公平性」
- 種類: counterfactual, fairness
- 日付: 20200521
- URL: [https://arxiv.org/abs/2005.10430](https://arxiv.org/abs/2005.10430)


### 概要

counterfactual fairness測定のために, 性別や人種などの属性を変えて, それ以外の属性を固定した画像を生成するオートエンコーダーを提案.

### 手法

センシティブ属性以外が変化しないように, 人物領域をセグメントしたり, 他の属性を明示的に固定したりした.

### 結果

様々なAPIに対して属性と出力がp値<0.001で相関していると結論づけた.

## Fader Networks: Manipulating Images by Sliding Attributes

wataokaの日本語訳「Fader Networks: 属性スライドによる画像編集」
- 種類: image manipulation
- 学会: NeurIPS2017
- 日付: 20170601
- URL: [https://arxiv.org/abs/1706.00409](https://arxiv.org/abs/1706.00409)


### 概要

画像の属性変換をするための手法Fader Networksを提案.

### 手法

xからEncoderでE(x)に埋め込み, そのE(x)から属性aをDiscriminatorに予測させる. 敵対的学習により, E(x)から属性aの情報を消す. 属性aとE(x)を入力とするDecoderで画像D(E(x), a)を生成する. 結果, 属性aをいじることで属性aを編集した画像を生成することができる.

### wataokaのコメント

近年の画像編集のほぼ始祖

## AttGAN: Facial Attribute Editing by Only Changing What You Want

wataokaの日本語訳「AttGAN: 変えたいものだけ変えられる顔属性編集」
- 種類: image manipulation, gan
- 学会: IEEE TIP2018
- 日付: 20171129
- URL: [https://arxiv.org/abs/1711.10678](https://arxiv.org/abs/1711.10678)


### 概要

タイトルの通り, 変えたい属性のみを編集した顔画像を生成できるGANを提案.

### 手法

入力データをzに埋める. zとある属性の真の値aをGに入力し, 生成した画像と元画像で再構成誤差. また, zからその属性の編集値bをGに入力し, 生成した画像をDとCに入力. Dはreal or fakeを見抜く. Cはその属性の値を分類. GはDには見抜かれないように学習し, Cには分類されるように学習する.

### wataokaのコメント

code: https://github.com/LynnHo/AttGAN-Tensorflow

## Inverting The Generator Of A Generative Adversarial Network

wataokaの日本語訳「GANのGeneratorの逆変換」
- 種類: GAN inversion
- 学会: NIPS2016 Workshot
- 日付: 20161117
- URL: [https://arxiv.org/abs/1611.05644](https://arxiv.org/abs/1611.05644)


### 概要

- xlog{G(z)} - (1-x)log{1-G(z)}を損失関数として, xに対する最適なzを勾配法で探索する.

### 手法

xlog()

## Invertibility of Convolutional Generative Networks from Partial Measurements

wataokaの日本語訳「部分測定からの畳み込み生成ネットの逆変換」
- 種類: GAN inversion
- 学会: NIPS2018
- 日付: 20181202
- URL: [https://papers.nips.cc/paper/8171-invertibility-of-convolutional-generative-networks-from-partial-measurements](https://papers.nips.cc/paper/8171-invertibility-of-convolutional-generative-networks-from-partial-measurements)


### 概要

CNNのinverseは非常に非凸で, 困難な計算である. この研究は2層の畳み込み生成ネットと単純な勾配降下を用いて, 潜在ベクトルを出力から効率的に推論できることを厳密に証明した. この理論敵発見は低次元潜在空間から高次元の画像空間への写像は単射であることを示唆している.

### wataokaのコメント

code: https://github.com/fangchangma/invert-generative-networks

## Disentangled Inference for GANs with Latently Invertible Autoencoder

wataokaの日本語訳「潜在的可逆オートエンコーダを用いたGANのためのdisentangledな推論」
- 種類: GAN inversion
- 学会: ICLR2020
- 日付: 20190619
- URL: [https://arxiv.org/abs/1906.08090](https://arxiv.org/abs/1906.08090)


### 概要

GAN inversion手法のLIA(Latently invertible autoencoder)を提案.

### 手法

x→[f]→y→[φ]→z→[φ^-1]→y→[g]→x~ (y:中間潜在表現, z:潜在表現, φ:可逆関数) stage1: Dを騙すようにφとgを訓練. stage2: φを取り外し, Dを騙す訓練と特徴量距離を最小化する訓練をする.

### wataokaのコメント

code: https://github.com/genforce/lia
idinvertとinterfaceganと同じgithubグループ

## In-Domain GAN Inversion for Real Image Editing

wataokaの日本語訳「実画像編集のためのIn-Domain GAN Inversion」
- 種類: GAN inversion
- 学会: ECCV2020
- 日付: 20200331
- URL: [https://arxiv.org/abs/2004.00049](https://arxiv.org/abs/2004.00049)


### 概要

GANのinverseをエンコーダ学習と最適化で行った論文. 実画像を再構成するエンコーダ(domain-guided encoder)を先に学習させておき, それを正則化として最適化する.

### 手法

まず, x→[E]→z→[G]→x'→[D]→real/recでEとDを敵対的に学習させる. そしてxからzを推論するために, 次の3つを満たすように最適化を解く. (1)生成画像G(z)がxに近い. (2)G(z)の特徴量F(G(z))がxの特徴量F(x)と近い. (3) G(z)のエンコードE(G(z))が元のzに近い.

### 結果

顔属性変換, image interpolation, semantic diffusionタスクに適応させて, 従来手法よりよかった.

### wataokaのコメント

TensorFlowのコードもPyTorchのコードもある
website: https://genforce.github.io/idinvert/
interfaceganとLIAと同じgithubグループ

## Image Processing Using Multi-Code GAN Prior

wataokaの日本語訳「複数のGANの事前コード(z)を用いた画像処理」
- 種類: GAN inversion
- 学会: CVPR2020
- 日付: 20201216
- URL: [https://arxiv.org/abs/1912.07116v2](https://arxiv.org/abs/1912.07116v2)


### 概要

既存手法は単一のzからinvertしようとしていたが限界がある. 複数のzからinvertする手法mGANpriorを提案した.

### 手法

複数のzから複数の中間表現を得る. 中間表現のチャネルに対してベクトルαで重みづけをして, 全ての中間表現を足し合わせる. その結果得られた画像を元画像に近づけるようにz達とα達を最適化する. objectはピクセルのL2と特徴量のL1.

### 結果

単一のzを最適化するorEncoderを用いて単一のzを推論する手法よりも定性的にも定量的にも良い精度になり, 単一zではやりにくかった色付けタスクや高解像タスクなどもできるようになった.

### wataokaのコメント

code: https://github.com/genforce/mganprior

## StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditinoal Continuous Normalizing Flows

wataokaの日本語訳「StyleFlow: 条件付き連続Flowを用いたStyle-GANの生成画像の属性条件付け」
- 種類: image manipulation/ Flow
- 日付: 20200806
- URL: [https://arxiv.org/abs/2008.02401](https://arxiv.org/abs/2008.02401)


### 概要

StyleFlowを提案. 属性条件付けsampleと属性編集のタスクをシンプルかつ効率的かつロバストに行える. 

### 手法

zとw(全ての重みWのサブスペース)をflowで繋ぐ. flowは属性aで条件付けできるようにする. そして, 画像からaを分類し, aで条件付けて, 初期値w0からz0を生成, z0から編集したatで条件付けてwtを生成, wtを用いてStyleGANで編集画像を生成.

### wataokaのコメント

website: https://rameenabdal.github.io/StyleFlow/

## A Linear Non-Gaussian Acyclic Model for Causal Discovery

wataokaの日本語訳「因果探索のための線形非ガウス非巡回モデル」
- 種類: causal inference
- 学会: Journal of Machine Learning Research 2006
- 日付: 20061006
- URL: [https://www.cs.helsinki.fi/group/neuroinf/lingam/JMLR06.pdf](https://www.cs.helsinki.fi/group/neuroinf/lingam/JMLR06.pdf)


### 概要

LiNGAMを提案

## Estimation of causal effects using linear non-Gaussian causal models with hidden variables

wataokaの日本語訳「隠れ変数のある線形非ガウス因果モデルを用いた因果効果の推定」
- 種類: causal inference
- 学会: International Journal of Approximate Reasoning 2008
- 日付: 20081000
- URL: [https://189568f5-a-62cb3a1a-s-sites.googlegroups.com/site/sshimizu06/ijar07.pdf?attachauth=ANoY7cpqDtq0TkopTBeV1UYzz2oXubY2uiu6V-FC8ZnvVB8ek_mwcJX3-Is8a0a_SzkgNKcxnRNrYI7j6nQn5bljXUp502hDKP9dAZJq4qZnHeYMwWUAko1Bt5z2coxAghulrT1ic-PFyDRTWNIikZyrA69pkpt0St2XOF0SA_t72skyVRceUvUvp9v38AxG2j7kQx-dQqWF8vQKNHJSFl-vjvSmWPFfBg%3D%3D&attredirects=0](https://189568f5-a-62cb3a1a-s-sites.googlegroups.com/site/sshimizu06/ijar07.pdf?attachauth=ANoY7cpqDtq0TkopTBeV1UYzz2oXubY2uiu6V-FC8ZnvVB8ek_mwcJX3-Is8a0a_SzkgNKcxnRNrYI7j6nQn5bljXUp502hDKP9dAZJq4qZnHeYMwWUAko1Bt5z2coxAghulrT1ic-PFyDRTWNIikZyrA69pkpt0St2XOF0SA_t72skyVRceUvUvp9v38AxG2j7kQx-dQqWF8vQKNHJSFl-vjvSmWPFfBg%3D%3D&attredirects=0)


### 概要

LvLiNGAMを提案.

### wataokaのコメント

LvLiNGAM, matlab code, ICAベース

## Bayesian estimation of causal direction in acyclic structual equation models with individual-specific confounder variables and non-Gaussian distributions.

wataokaの日本語訳「個別交絡因子と非ガウスを用いた非巡回構造方程式における因果方向のベイズ推定」
- 種類: causal inference
- 学会: Journal of Machine Learning Rearch 2014
- 日付: 20130000
- URL: [https://jmlr.org/papers/volume15/shimizu14a/shimizu14a.pdf](https://jmlr.org/papers/volume15/shimizu14a/shimizu14a.pdf)


### 概要

BMLiNGAMを提案.

### wataokaのコメント

BMLiNGAM, python code, 混合モデルベース (未観測共通原因を明示的にモデルに組み込まない)

## ParceLiNGAM: A causal ordering method rubust against latent confounders

wataokaの日本語訳「ParceLiNGAM: 潜在交絡に対してロバストな因果順序法」
- 種類: causal inference
- 学会: Neural Computation2014
- 日付: 20130329
- URL: [https://arxiv.org/abs/1303.7410](https://arxiv.org/abs/1303.7410)


### 概要
- 潜在交絡因子が存在しないという仮定を取っ払ったLiNGAMを提案.
- Key ideaは外生変数間の独立性をテストすることで潜在交絡因子を検知し, 潜在交絡因子によって影響を受けていない変数集合(parcels)を見つけるということ.
- 人工データと脳画像データで実験を行った.

xjと全てのiに対する残差r(j)iが独立である=xjはsource変数という性質とその逆を用いて, source変数とsink変数を排除していくアルゴリズムをベースとして, それが解けない状況にもrobustな改良もしている. 出力はcausal order matrix

### 3 A method robust against latent confounders
3.1 Identification of causal orders of variables that are not affected by latent confounders

```math
x = Bx + Λf + e (3)
```

#### Lemma 1
式(3)における潜在変数LiNGAMの仮定を満たし, サンプル数が無限である時, 以下が同値.
「全てのiに対してx_jと残差r(j)_iが独立」
「x_jは親も潜在交絡因子も持たない外生変数」

#### Lemma 2
Lemma 1の逆. 全部と依存してたらsink変数だよ的な定理.


## Causal discovery of linear non-Gaussian acyclic models in the presence of latent confounders

wataokaの日本語訳「潜在交絡因子のある線形非ガウス非巡回モデルの因果探索」
- 種類: causal inference
- 学会: AISTATS2020
- 日付: 20200113
- URL: [https://arxiv.org/abs/2001.04197](https://arxiv.org/abs/2001.04197)


### 概要

RCDを提案.

### 手法

 RCDは少数の観測変数間の因果方向の推論を繰り返し, その関係がlatent confounderの影響を受けているかを判断する. 最終的に因果グラフを作成し, 双方向矢印はlatent confounderをもつ変数ペアを示し, 有効矢印はlatent confounderに影響を受けない変数ペアの因果を表す.

### wataokaのコメント

RCD

## Causation, Prediction, and Search

wataokaの日本語訳「因果関係, 予測, 探索」
- 種類: cuasal inference
- 学会: Springer
- 日付: 19930000
- URL: [https://books.google.com/books/about/Causation_Prediction_and_Search.html?id=oUjxBwAAQBAJ&printsec=frontcover&source=kp_read_button&hl=en#v=onepage&q&f=false](https://books.google.com/books/about/Causation_Prediction_and_Search.html?id=oUjxBwAAQBAJ&printsec=frontcover&source=kp_read_button&hl=en#v=onepage&q&f=false)


### 概要

FCIを提案
(Fast Causal Inference)

### 手法

adjacency phaseとorientation phaseの2つのフェーズがある. adjacency phaseは完全無効グラフから始まり, 条件付けして独立したらエッジを削除する. orientation phaseでは, 条件付けした変数セットを用いてエッジに方向をつける.

## Learning high-dimensional directed acyclic graphs with latent and selection variables

wataokaの日本語訳「潜在選択変数のある高次元の有向非巡回グラフの学習」
- 種類: causal inference
- 日付: 20110429
- URL: [https://arxiv.org/abs/1104.5617](https://arxiv.org/abs/1104.5617)


### 概要

RFCIを提案.
(Really Fast Causal Inference)

## A Hybrid Causal Search Algorithm for Latent Variable Models

wataokaの日本語訳「潜在変数モデルのためのハイブリッド因果探索アルゴリズム」
- 種類: causal inference
- 学会: JMLR2016 workshop
- 日付: 20160000
- URL: [http://proceedings.mlr.press/v52/ogarrio16.pdf](http://proceedings.mlr.press/v52/ogarrio16.pdf)


### 概要

GFCIを提案.
(Greedy Fast Causal Inference)

## Independent Component Analysis

wataokaの日本語訳「独立成分分析」
- 種類: ICA
- 学会: Wiley Interscience 2001
- 日付: 19970000
- URL: [https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf)


### 概要

Hyvarinenの最初のICAの本

### wataokaのコメント

overcomplete ICAにも言及しているが, 簡単な尤度最大化手法とかしか書かれていない.

## Independent Factor Analysis

wataokaの日本語訳「独立要素解析」
- 種類: ICA
- 学会: Neural Computation 1999
- 日付: 19990000
- URL: [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.62&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.62&rep=rep1&type=pdf)


### 概要

IFAを提案.
(Independent Factor Analysis)
LvLiNGAMが用いたovercomplete ICA手法.

### 手法

独立成分の分布に混合ガウスを仮定し, EMアルゴリズムで尤度最大化. 

## Mean-Field Approaches to Independent Component Analysis

wataokaの日本語訳「ICAのためのMean-Fieldアプローチ」
- 種類: ICA
- 学会: Neurla Computation 2002
- 日付: 20020500
- URL: [https://www.researchgate.net/publication/11428922_Mean-Field_Approaches_to_Independent_Component_Analysis](https://www.researchgate.net/publication/11428922_Mean-Field_Approaches_to_Independent_Component_Analysis)


### 概要

MFICAを提案.
(Mean-Field Approaches to ICA)

### wataokaのコメント

LFOICAの比較手法

## ICA with Reconstruction Cost for Efficient Overcomplete Feature Learning

wataokaの日本語訳「効率的な過完備特徴学習のための再構成コスト付きICA」
- 種類: ICA
- 学会: NeurIPS2011
- 日付: 20111212
- URL: [http://www.robotics.stanford.edu/~ang/papers/nips11-ICAReconstructionCost.pdf](http://www.robotics.stanford.edu/~ang/papers/nips11-ICAReconstructionCost.pdf)


### 概要

RICAを提案.
(Reconstruction ICA)

### wataokaのコメント

LFOICAの比較手法

## Discovering Temporal Causal Relations from Subsampled Data

wataokaの日本語訳「サブサンプルデータからの時間的因果関係の発見」
- 種類: ICA
- 学会: ICML2015
- 日付: 20150000
- URL: [https://mingming-gong.github.io/papers/ICML_SUBSAMPLE.pdf](https://mingming-gong.github.io/papers/ICML_SUBSAMPLE.pdf)


### 概要

NG-EMとNG-MFを提案.
(Non-Gaussian EMとNon-Gaussian Mean-Field)

### wataokaのコメント

LFOICAの比較手法

## overcomplete Independent Component Analysis via SDP

wataokaの日本語訳「SDPを用いた過完備ICA」
- 種類: ICA
- 学会: AISTATS2019
- 日付: 20190124
- URL: [https://arxiv.org/abs/1901.08334](https://arxiv.org/abs/1901.08334)


### 概要

overcomplete ICAを半正定値計画問題に落とし込み, それをprojected accelerated gradient descent methodで解いた.

### wataokaのコメント

matlab code: https://github.com/anastasia-podosinnikova/oica

## Likelihood-Free Overcomplete ICA and Applications in Causal Discovery

wataokaの日本語訳「尤度が必要ない過完備ICAと
因果探索における応用」
- 種類: ICA
- 学会: NeurIPS2019
- 日付: 20190904
- URL: [https://arxiv.org/abs/1909.01525](https://arxiv.org/abs/1909.01525)


<iframe src="https://docs.google.com/document/d/e/2PACX-1vTxC1eyyTWwhfrg6J4QolY3HQ2KvW9b1LPepz26paT2Flm2k1TCKkXMjNggcAlCivgXjZjPPMyLBNyT/pub?embedded=true"></iframe>
## From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification

wataokaの日本語訳「SoftmaxからSparsemaxへ: attentionのスパースモデルとマルチラベル分類」
- 種類: general
- 学会: ICML2016
- 日付: 201602205
- URL: [https://arxiv.org/abs/1602.02068](https://arxiv.org/abs/1602.02068)


### 概要

softmaxに変わる出力層の関数sparsemaxを提案. アイディアとしては, 小さい値は0に, 大きい値はキープ, そしてもちろん微分可能.

### 手法

1より大きければ1, -1より小さければ0, 間の値は両端を繋ぐような線形の関数.

### 結果

[0.1, 1.1, 0.2, 0.3]とかをsoftmaxに入力すると, 
[0.165, 0.45 , 0.183, 0.202]
sparsemaxに入力すると,
[0. , 0.9, 0. , 0.1]

## GANITE: Estimation of Indeividual Treatment Effects using Generative Adversarial Nets

wataokaの日本語訳「GANITE: GANを用いた個人処置効果の推定」
- 種類: GAN, causal inference
- 学会: ICLR2018
- 日付: 20180216
- URL: [https://openreview.net/pdf?id=ByKWUeWA-](https://openreview.net/pdf?id=ByKWUeWA-)


### wataokaのコメント

公式か分からないがcode: https://github.com/jsyoon0823/GANITE,
日本語解説記事: https://cyberagent.ai/blog/research/causal_inference/10261/

## The Counterfactual chi-GAN

wataokaの日本語訳「反実仮想カイGAN」
- 種類: GAN, counterfactual
- 日付: 20200109
- URL: [https://arxiv.org/abs/2001.03115](https://arxiv.org/abs/2001.03115)


## Density estimation using Real NVP

wataokaの日本語訳「Real NVPを用いた密度推定」
- 種類: Flow
- 学会: ICLR2017
- 日付: 20160327
- URL: [https://arxiv.org/abs/1605.08803](https://arxiv.org/abs/1605.08803)


### 概要

Flowの一種のRealNVP(real-valued non-valume presering)を提案. NICEでは, detが1であるので, スケーリングができない. そこで スケールング関数sを含んだアフィンカップリングを用いた.

### 手法

次元分割し, 一方をxとし, もう一方をx'とした時, y = x, y' = x'⊙exp(s(x))+t(x')とした. これで可逆かつスケーリング可能な変換となる.

### 結果

まだ高画質画像には弱そう

## Glow: Generative Flow with Invertible 1x1 Convolutions

wataokaの日本語訳「Glow: 1x1の可逆な畳み込みを用いた生成Flow」
- 種類: Flow
- 学会: NeurIPS2018
- 日付: 20180709
- URL: [https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)


### 概要

Flowの一種のGlowを提案. RealNVPに1x1の畳み込みとactnormを加えたFLow. RealNVPに比べ, 高画質画像の生成に成功した.

### 手法

ActNorm(Activation Normalization): channel-wiseなNormalization. 1x1の畳み込み: 入力と出力のwidth, heightはかえず, channel方向に畳み込む. なので正確には1x1xcのカーネルで畳み込む. 逆変換時に計算するlogdetはLU分解などを用いて計算量をO(c)にできる.

### wataokaのコメント

code: https://github.com/openai/glow,
日本語記事: https://qiita.com/exp/items/4f562ec788f2ac5241dc#3-%E7%94%9F%E6%88%90%E7%9A%84flow%E3%81%AE%E6%8F%90%E6%A1%88

## Learning Non-Discriminatory Predictors

- URL: [https://arxiv.org/abs/1702.06081](https://arxiv.org/abs/1702.06081)


### 概要

Group Fairnessの学習理論. EOに関する汎化誤差を評価した.

### wataokaのコメント

日本語記事: https://www.slideshare.net/kazutofukuchi/ai-197863181

## Training Well-Generalizing Classifiers for Fairness Metrics and Other Data-Dependent Constraints

- URL: [https://arxiv.org/abs/1807.00028](https://arxiv.org/abs/1807.00028)


### 概要

Group Fairnessの学習理論. 上の論文を一般化. 分類誤差を目的関数, 公平性を製薬として最適化したい問題において, 汎化誤差を評価した.

### wataokaのコメント

日本語記事: https://www.slideshare.net/kazutofukuchi/ai-197863181

## Probably Approximately Metric-Fair Learning

- URL: [https://arxiv.org/abs/1803.03242](https://arxiv.org/abs/1803.03242)


### 概要

Individual Fairnessの学習理論.

### wataokaのコメント

日本語記事: https://www.slideshare.net/kazutofukuchi/ai-197863181

## Why Normalizing Flows Fail to Detect Out-of-Distribution Data

- URL: [https://arxiv.org/abs/2006.08545](https://arxiv.org/abs/2006.08545)


### 概要

Normalizing Flowがout of distributionなデータに弱い(帰納バイアスがかかっている)ことを示した.

### wataokaのコメント

code: https://github.com/PolinaKirichenko/flows_ood

## Achieving Causal Fairness in Machine Learning 

- URL: [https://scholarworks.uark.edu/cgi/viewcontent.cgi?article=5197&context=etd](https://scholarworks.uark.edu/cgi/viewcontent.cgi?article=5197&context=etd)


### 概要

Causal Fairness全般のことが書かれている. Counterfactual Fairness→Path-specific Counterfactual Fairness→複雑さとバウンドの流れが書かれていて読み応えがある.

## A causal framework for discovering and removing direct and indirect discrimination

wataokaの日本語訳「直接/間接差別を発見し, 取り除くための因果フレームワーク」
- 種類: fairness
- 学会: IJICAI2017
- 日付: 20161122
- URL: [https://arxiv.org/abs/1611.07509](https://arxiv.org/abs/1611.07509)


### 概要

直接/間接差別を因果モデルにおけるpath-specific effectであるとした. それを発見し, 取り除くアルゴリズムを提案した. 取り除いたデータを用いて予測モデルを構築すると差別しなくなる.

## Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis

wataokaの日本語訳「多様性のある画像生成のためのモードシーキングGAN」
- 種類: GAN
- 学会: CVPR2019
- 日付: 20190313
- URL: [https://arxiv.org/abs/1903.05628v6](https://arxiv.org/abs/1903.05628v6)


### 概要

zを変化させても画像が変化しないことをmode collapseといい, 逆にうまく反映できていることをmode seekingという. mode seekingするためのGANであるMSGANを提案した.

### 手法

mode collapseが起きている時, 以上に(画像空間での距離)/(潜在空間での距離)という比率が大きくなることに注目し, その比率のマイナスを損失関数に入れる正則化を行った.

### 結果

画像の品質を下げずに多様性を向上することに成功した.

### wataokaのコメント

code: https://github.com/HelenMao/MSGAN

## Counterfactual Data Augmentation using Locally Factored Dynamics

wataokaの日本語訳「局所的に分解されたダイナミクスを用いた反実仮想データ増強」
- 種類: counterfactual
- 学会: NeurIPS2020
- 日付: 20200706
- URL: [https://arxiv.org/abs/2007.02863](https://arxiv.org/abs/2007.02863)


### 概要

RLなどにおいて多くの動作プロセスはサブプロセス同士が繋がっているのに, 非常にスパースなので独立的に考えられがち. そこで, この論文では, LCMsとしてダイナミクスを定式化し, データ増強アルゴリズムCoDAを提案した.

### wataokaのコメント

code: https://github.com/spitis/mrl

## Deep Structural Causal Models for Tractable Counterfactual Inference

wataokaの日本語訳「反実仮想推論のためのディープ階層的因果モデル」
- 種類: counterfactual
- 学会: NeurIPS2020
- 日付: 20200611
- URL: [https://arxiv.org/abs/2006.06485](https://arxiv.org/abs/2006.06485)


## Information Theoretic Counterfactual Learning from Missing-Not-At-Random Feedback

wataokaの日本語訳「情報理論的なMNARフィードバックからの反実仮想学習」
- 種類: couterfactual
- 学会: NeurIPS2020
- URL: [https://proceedings.neurips.cc/paper/2020/file/13f3cf8c531952d72e5847c4183e6910-Paper.pdf](https://proceedings.neurips.cc/paper/2020/file/13f3cf8c531952d72e5847c4183e6910-Paper.pdf)


### 概要

RCTは計算コストが高いので, 代わりに情報理論的な反実仮想変分情報ボトルネックを導いた. factual/couterfactual domainsでバランスの良い学習のためのcontrastive information lossと出力値へのペナルティ方法を提案した.

## Adversarial Vertex Mixup: Toward Better Adversarially Robust Generalization

wataokaの日本語訳「敵対的頂点Mixup: より良い敵対的に頑健な生成に向けて」
- 種類: adversarial attack
- 学会: CVPR2020
- 日付: 20200305
- URL: [https://arxiv.org/abs/2003.02484v3](https://arxiv.org/abs/2003.02484v3)


### 概要

ATは最適なポイントを通り過ぎがち.(やりすぎ) これをAFO(Adversarial Feature Overfitting)とした.  AFOを防ぐために, AVmixupというソフトラベル化されたデータ増強手法を提案した.

## CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations

wataokaの日本語訳「CelebA-Spoof: リッチなアノテーション付きなりすまし防止大規模顔画像データセット」
- 種類: general
- 学会: ECCV2020 Workshop
- 日付: 20200724
- URL: [https://arxiv.org/abs/2007.12342](https://arxiv.org/abs/2007.12342)


### 概要

CelebAを作成した香港大の新しいデータセット. 62万枚という大規模な顔画像データで, 多様な場所での写真やリッチなアノテーションと謳っている.

### wataokaのコメント

data: https://github.com/Davidzhangyuanhan/CelebA-Spoof

## MAAD-Face: A Massively Annotated Attribute Dataset for Face Images

wataokaの日本語訳「MAAD-Face: 大規模にあのテートされた属性付き顔画像データセット」
- 種類: general
- 日付: 20201202
- URL: [https://arxiv.org/abs/2012.01030](https://arxiv.org/abs/2012.01030)


### 概要

-1, 0, 1の3段階のラベルがある顔画像データセット

### wataokaのコメント

data: https://github.com/pterhoer/MAAD-Face

