## 1 train

### 1.1

完成并行训练代码，2张卡训练 (done)

分别使用matcha和deplot进行训练

https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/406250

matcha (done)

```python
model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-plotqa-v2")
processor = Pix2StructProcessor.from_pretrained("google/matcha-base")
```

deplot

```python
model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
processor = AutoProcessor.from_pretrained("google/deplot")
```

### 1.2

https://github.com/MhLiao/DB

DB-NET对图像进行文本检，检测x y轴的label，作为prompt输入 (效果一般)

```python
encoding = self.processor(images=image,
                                  # prompt
                                  text="x-axis is <x1,x2,x3,x4> y-axis is <y1,y2,y3,y4>“ +
                          						 "Generate underlying data table of the figure below:",
                                  font_path="arial.ttf",
                                  return_tensors="pt",
                                  add_special_tokens=True, max_patches=MAX_PATCHES)
```

### 1.3

500k graphs数据集

增加数据进行训练以提升模型准确率

https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/413055



## 2 infer

### 2.1

https://www.kaggle.com/code/thanhhau097a/deplot-inference

使用deplot inference代码 (done)

需要改标签形式 (done)

```python
# x1,x2 <> y1 y2
# x1 | y1 <0x0A> x2 | y2 <0x0A>
# label xxxxyyyy -> xyxyxyxy
```

### 2.2

5种图片类型 

需要使用resnet去5分类 (done) resnet50

```python
'dot': 0, 'horizontal_bar' : 1, 'vertical_bar': 2, 'line': 3, 'scatter': 4
```

参考以下代码

https://www.kaggle.com/code/thanhhau097a/simple-submission-classification-task-training

### 2.3

替换对应的权重文件并保证 train 和 infer 的 transformer 版本一致 (done)

完成线上 infer 得到第一个分数 (done, score:0.46)


## schedule

6.5 - 6.6 (done)

训完没并行的 matcha

6.6 (done)

huggingface trainer

data parallel 实现 ddp 训练

训完 有并行的 deplot

6.7

线上线下 infer (done)

===> 走通流程，得到第一个分数 0.60左右 (done,score: 0.46)

6.8

dbnet 得到xy标签 (done, 识别效果一般)

添加 prompt进行训练

6.12

添加 weight-decay 和 warmup 进行训练 (done， weight-decay = 1/10 lr, warmup-step = 0.2)

添加 1/10 500k 数据集进行训练

添加 ICDAR 2023 数据集进行训练

修改线上inference得到提交结果部分的代码

使用 deplot 进行 finetune

=======>  6.20 end
