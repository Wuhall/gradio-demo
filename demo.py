import gradio as gr
import pandas as pd
import numpy as np
import random
import PIL.Image as Image
import matplotlib.pyplot as plt
from ultralytics import ASSETS, YOLO
import matplotlib
matplotlib.use('Agg')

model = YOLO("./weights/construction_safe_site.pt")

def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLOv5 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

# construction_safe datasets
df = pd.read_csv('./static/construction_safe_instance.csv')
# telecomNet datasets
df2 = pd.read_csv('./static/telecomNet_instance.csv')

# Define a function to plot a bar chart
def plot_bar_chart():
    plt.figure(figsize=(7, 5))
    plt.bar(df['Class'], df['Instances'], color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Instances')
    plt.title('Number of Instances per Class')
    plt.xticks(rotation=20)
    plt.tight_layout()
    return plt

def plot_bar_chart2():
    plt.figure(figsize=(7, 5))
    plt.bar(df2['Class'], df2['Instances'], color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Instances')
    plt.title('Number of Instances per Class')
    plt.xticks(rotation=20)
    plt.tight_layout()
    return plt

# Gradio Blocks UI
# footer {visibility: hidden}
with gr.Blocks(theme=gr.themes.Monochrome(), css="footer {visibility: hidden}", title=" AI Platform") as demo:
    gr.Markdown("#  AI Platform")

    with gr.Tab("评价指标"):
        metix_dataset = gr.Accordion("评价指标", open=False)
        with metix_dataset:
            gr.Markdown("""
                ## 数据集
                - 建议每类≥ 10000 个实例（标记对象）
                - 图像种类。必须有代表性。对于真实世界的案例，建议使用不同时间、不同季节、不同天气、不同光线、不同角度、不同来源（网上搜索、本地收集、不同摄像头）等的图像。
                - 标签一致性。所有图像中所有类别的所有实例都必须贴上标签。部分标注将不起作用。
                - 标签准确性。标签必须紧密包围每个对象。对象与边界框之间不得有空隙。任何对象都不能缺少标签。
                - 背景图像。背景图片是没有物体的图像，添加到数据集中可以减少误报（FP）。建议使用约 0-10% 的背景图片来帮助减少误报率，背景图片不需要标签。               
                ## 评价指标
                - P（精确度）：检测物体的精确度，表示有多少检测是正确的。
                - R（召回率）：模型识别图像中所有物体实例的能力。
                - F1分数曲线:该曲线表示不同阈值下的 F1 分数。曲线走向代表模型在不同阈值下假阳性和假阴性之间的平衡。
                - mAP50：按 0.50 的交集大于联合（IoU）阈值计算的平均精度。
                """)
        gr.Markdown("""
                    ## construction-site-safety240818
                    - 训练集：2605张（6张背景）
                    - 验证集：114张（10张背景）
                    - 测试集：82张
                    """)
        data_output = gr.DataFrame(label="类别实例表格", headers=["Class", "Instances"], value=df)
        plot_output = gr.Plot(label="柱状图")
        demo.load(fn=plot_bar_chart, inputs=[], outputs=plot_output) 

        # gr.Markdown("""
        #     ## 质量(telecomNet240909))
        #     - 训练集：2605张（6张背景）
        #     - 验证集：114张（10张背景）
        #     - 测试集：82张
        #     """)
        data_output = gr.DataFrame(label="类别实例表格", headers=["Class", "Instances"], value=df2)
        # plot_output2 = gr.Plot(label="柱状图")
        # demo.load(fn=plot_bar_chart2, inputs=[], outputs=plot_output2) 
        
    

    with gr.Tab("检测"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="上传图像", sources=['upload'], width=300, height=300)
            image_output = gr.Image(type="pil", label="结果", width=300, height=300)
            conf_slider = gr.Slider(minimum=0, maximum=1, value=0.25, label="置信度")
            iou_slider = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU")
                
        example_img_select = gr.Accordion("示例图片选择", open=True)
        with example_img_select:
            gr.Examples(
                examples=[
                    ['static/img/employee.jpg', 0.25, 0.45],
                    [ASSETS / "zidane.jpg", 0.25, 0.45],
                ],
                inputs=[image_input, conf_slider, iou_slider],
                label="示例图片"
            )

        predict_button = gr.Button("预测")
        predict_button.click(
            fn=predict_image,
            inputs=[image_input, conf_slider, iou_slider],
            outputs=image_output
        )

if __name__ == "__main__":
    demo.launch(share=False)