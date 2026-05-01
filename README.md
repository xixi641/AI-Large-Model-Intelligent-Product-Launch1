# AI-Large-Model-Intelligent-Product-Launch
一款可以发布智能商品的项目

项目介绍： 在电商平台中，商品发布通常是一个繁琐的过程，需要手动录入商品的各项信息，如品牌、品类等。为了提升效率并减轻商家的工作负担，智能商品录入系统应运而生。该系统能够根据商家提供的商品标题，自动预测并填写相关信息，从而加速商品上架过程 
本项目的核心功能是 **商品分类预测，通过自动化分析商品标题，实现品类的自动分类，帮助商家快速、准确地完成商品录入**

所需py库如下：
pytorch：深度学习框架，用于训练和推理

transformers：Hugging Face 提供的库，用于加载和微调 BERT 等预训练模型。

datasets：用于高效加载和处理大规模数据集。

scikit-learn：用于模型评估。

tensorboard：用于可视化训练过程中的损失、准确率等指标。

tqdm：用于显示训练进度条，方便监控训练过程。

jupyter：用于实验和数据分析。

FastAPI：用于构建和部署API接口。 

Uvicorn：FastAPI的服务器，用于高性能地运行FastAPI应用。


注意：
1.**本项目仅为单层分级 此外，项目需要提供接口，以便商品发布系统进行调用。**
2.**由于训练需要使用nVidia显卡的cuda，使用兼容性更高的torch_directml库，如需使用cuda进行训练，请自行修改，使用torch库。**
3.**web页面部署使用fastapi进行部署，预测模型需要进入docs页面进行查看。**

下载下方的pytorch，放在models目录下即可运行

我用夸克网盘给你分享了「best.pt」，点击链接或复制整段内容，打开「夸克APP」即可获取。
/~33723YJ81T~:/
链接：https://pan.quark.cn/s/df59a8d55547
提取码：bwCT

运行方法：
1.<br>
在命令行中输入：> python src/main.py server即可启用web页面，默认进入8000端口主页面<br>
输入http://127.0.0.1:8000/docs进入post请求页面测试预测结果<br>
2.<br>
在命令行输入python src/main.py train即可启动模型进行训练<br>
3.<br>
在命令行输入python src/main.py predict即可进行模型预测<br>
4.<br>
在命令行输入python src/main.py evaluate即可进行模型评估<br>
